"""
llama.cpp Inference Handler

Handles inference using llama.cpp for CPU/edge device inference with GGUF models.
Supports Metal acceleration on macOS and optimized CPU inference.
"""

from typing import List, Dict, Optional, AsyncIterator
import logging
from dataclasses import dataclass
import asyncio
import platform
import os
import time

logger = logging.getLogger(__name__)


@dataclass
class InferenceRequest:
    """Request for model inference."""
    messages: List[Dict[str, str]]
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 0.9
    stop: Optional[List[str]] = None
    stream: bool = False


@dataclass
class InferenceResponse:
    """Response from model inference."""
    content: str
    model: str
    usage: Dict[str, int]
    latency_ms: float
    finish_reason: str = "stop"


class LlamaCppHandler:
    """
    llama.cpp inference handler for edge devices and CPU inference.

    Supports:
    - GGUF quantized models (Q4_K_M, Q5_K_M, etc.)
    - Metal acceleration for M-series Mac
    - Optimized CPU inference with multi-threading
    - Streaming generation
    - Low memory footprint
    - Context window management
    """

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 8192,
        n_threads: Optional[int] = None,
        use_metal: bool = True,
        verbose: bool = False
    ):
        """
        Initialize llama.cpp handler.

        Args:
            model_path: Path to GGUF model file
            n_ctx: Context window size (default: 8192)
            n_threads: Number of CPU threads (None = auto-detect)
            use_metal: Auto-detect and use Metal acceleration on macOS
            verbose: Enable verbose logging
        """
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.verbose = verbose

        # Auto-detect Metal availability on macOS
        self.is_macos = platform.system() == "Darwin"
        self.use_metal = use_metal and self.is_macos

        # Auto-detect optimal thread count
        if n_threads is None:
            cpu_count = os.cpu_count() or 8
            self.n_threads = max(4, cpu_count // 2)  # Use half of available cores
        else:
            self.n_threads = n_threads

        # Determine GPU layers based on Metal availability
        if self.use_metal:
            # For Metal, offload most layers to GPU
            self.n_gpu_layers = 35  # Good default for 4B-8B models
            logger.info(f"Metal acceleration enabled: offloading {self.n_gpu_layers} layers to GPU")
        else:
            # CPU-only mode
            self.n_gpu_layers = 0
            logger.info(f"CPU-only mode: using {self.n_threads} threads")

        self._llama = None
        self._loaded = False

    def load(self):
        """
        Load GGUF model using llama.cpp.

        Initializes the model with appropriate settings for Metal/CPU.
        Handles missing llama-cpp-python with realistic mock.
        """
        logger.info(f"Loading llama.cpp model: {self.model_path}")
        logger.info(f"Settings: n_ctx={self.n_ctx}, n_threads={self.n_threads}, "
                   f"n_gpu_layers={self.n_gpu_layers}, use_metal={self.use_metal}")

        try:
            # Try to import llama-cpp-python
            from llama_cpp import Llama

            # Verify model file exists
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")

            # Verify GGUF format
            if not self.model_path.endswith('.gguf'):
                logger.warning(f"Model file may not be in GGUF format: {self.model_path}")

            # Initialize llama.cpp
            self._llama = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers,
                n_threads=self.n_threads,
                use_mlock=True,   # Keep model in RAM to prevent swapping
                use_mmap=True,    # Use memory mapping for efficiency
                verbose=self.verbose,
            )

            self._loaded = True
            logger.info(f"Successfully loaded model: {self.model_path}")

        except ImportError:
            logger.warning("llama-cpp-python not installed, using mock implementation")
            self._llama = self._create_mock_llama()
            self._loaded = True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Failed to load llama.cpp model: {e}")

    def _create_mock_llama(self):
        """Create realistic mock for llama.cpp when library not available."""
        class MockLlama:
            def __init__(self, model_path):
                self.model_path = model_path

            def __call__(self, prompt, **kwargs):
                """Mock generation."""
                return {
                    "choices": [{
                        "text": f"[Mock llama.cpp response from {self.model_path}]\n\n"
                                f"This is a simulated response. Install llama-cpp-python for real inference."
                    }]
                }

            def create_chat_completion(self, messages, **kwargs):
                """Mock chat completion."""
                stream = kwargs.get('stream', False)

                response_text = (
                    f"[Mock llama.cpp chat response from {self.model_path}]\n\n"
                    f"This is a simulated chat response. Install llama-cpp-python for real inference."
                )

                if stream:
                    # Mock streaming response
                    def stream_generator():
                        words = response_text.split()
                        for word in words:
                            yield {
                                "choices": [{
                                    "delta": {"content": word + " "}
                                }]
                            }
                    return stream_generator()
                else:
                    return {
                        "choices": [{
                            "message": {"content": response_text}
                        }],
                        "usage": {
                            "prompt_tokens": 50,
                            "completion_tokens": 20,
                            "total_tokens": 70
                        }
                    }

        return MockLlama(self.model_path)

    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        """
        Generate response using llama.cpp.

        Args:
            request: Inference request with messages and parameters

        Returns:
            InferenceResponse with generated content and metadata

        Raises:
            RuntimeError: If model not loaded or generation fails
        """
        if not self._loaded or self._llama is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        start_time = time.time()

        try:
            # Format messages to prompt
            messages = request.messages

            # Run generation in thread pool to avoid blocking event loop
            loop = asyncio.get_event_loop()

            def _generate():
                try:
                    # Use chat completion API for better message handling
                    output = self._llama.create_chat_completion(
                        messages=messages,
                        temperature=request.temperature,
                        max_tokens=request.max_tokens,
                        top_p=request.top_p,
                        stop=request.stop or ["User:", "Human:", "<|endoftext|>"],
                        stream=False
                    )

                    # Extract response
                    content = output["choices"][0]["message"]["content"]
                    usage = output.get("usage", {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0
                    })

                    return content, usage

                except Exception as e:
                    logger.error(f"Generation failed: {e}")
                    raise RuntimeError(f"llama.cpp generation failed: {e}")

            # Execute in thread pool
            generated_text, usage = await loop.run_in_executor(None, _generate)

            latency_ms = (time.time() - start_time) * 1000

            return InferenceResponse(
                content=generated_text,
                model=self.model_path,
                usage=usage,
                latency_ms=latency_ms,
                finish_reason="stop"
            )

        except Exception as e:
            logger.error(f"Error during generation: {e}")
            raise

    async def generate_stream(
        self,
        request: InferenceRequest
    ) -> AsyncIterator[str]:
        """
        Generate streaming response using llama.cpp.

        Args:
            request: Inference request with messages and parameters

        Yields:
            String chunks of the generated response

        Raises:
            RuntimeError: If model not loaded or streaming fails
        """
        if not self._loaded or self._llama is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        try:
            # Format messages
            messages = request.messages

            # Create streaming generator
            loop = asyncio.get_event_loop()

            # llama.cpp streaming returns a generator, we need to wrap it
            def _stream_generate():
                try:
                    stream = self._llama.create_chat_completion(
                        messages=messages,
                        temperature=request.temperature,
                        max_tokens=request.max_tokens,
                        top_p=request.top_p,
                        stop=request.stop or ["User:", "Human:", "<|endoftext|>"],
                        stream=True
                    )

                    for chunk in stream:
                        if "choices" in chunk and len(chunk["choices"]) > 0:
                            delta = chunk["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content

                except Exception as e:
                    logger.error(f"Streaming generation failed: {e}")
                    raise RuntimeError(f"llama.cpp streaming failed: {e}")

            # Yield chunks from the generator
            for chunk in await loop.run_in_executor(None, lambda: list(_stream_generate())):
                yield chunk

        except Exception as e:
            logger.error(f"Error during streaming: {e}")
            raise

    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """
        Format messages into Qwen chat template prompt.

        This method is kept for compatibility but create_chat_completion
        handles message formatting internally.

        Args:
            messages: List of message dicts with 'role' and 'content'

        Returns:
            Formatted prompt string
        """
        prompt_parts = []

        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")

            if role == "system":
                prompt_parts.append(f"<|im_start|>system\n{content}<|im_end|>")
            elif role == "user":
                prompt_parts.append(f"<|im_start|>user\n{content}<|im_end|>")
            elif role == "assistant":
                prompt_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")

        # Add assistant prompt
        prompt_parts.append("<|im_start|>assistant\n")

        return "\n".join(prompt_parts)

    def unload(self):
        """
        Unload model and free resources.

        Clears model from memory to free RAM/VRAM.
        """
        logger.info(f"Unloading llama.cpp model: {self.model_path}")

        # Free the model
        self._llama = None
        self._loaded = False

        # Force garbage collection to free memory
        import gc
        gc.collect()

        logger.info("Model unloaded successfully")

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded and self._llama is not None

    def get_model_info(self) -> Dict[str, any]:
        """
        Get model information and settings.

        Returns:
            Dictionary with model configuration
        """
        return {
            "model_path": self.model_path,
            "n_ctx": self.n_ctx,
            "n_threads": self.n_threads,
            "n_gpu_layers": self.n_gpu_layers,
            "use_metal": self.use_metal,
            "is_macos": self.is_macos,
            "loaded": self._loaded
        }
