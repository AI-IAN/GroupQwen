"""
vLLM Inference Handler

Handles inference using vLLM framework for GPU-accelerated models.
"""

from typing import List, Dict, Optional, AsyncIterator
import logging
import time
import asyncio
import uuid
from dataclasses import dataclass

# Try to import vLLM, fallback to mock mode if not available
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    logging.warning("vLLM not available - running in mock mode")

# Try to import GPUtil for VRAM monitoring
try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False
    logging.warning("GPUtil not available - VRAM monitoring disabled")

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


class VLLMHandler:
    """
    vLLM inference handler for large models.

    Supports:
    - Continuous batching
    - PagedAttention for memory efficiency
    - Streaming responses
    - Multi-GPU inference
    - Error handling with retries
    - VRAM monitoring and OOM handling
    """

    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.90,
        max_model_len: int = 8192,
        max_retries: int = 3,
        inference_timeout: float = 30.0
    ):
        """
        Initialize vLLM handler.

        Args:
            model_name: HuggingFace model name
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: GPU memory utilization fraction
            max_model_len: Maximum model context length
            max_retries: Maximum number of retries for transient errors
            inference_timeout: Timeout in seconds for inference operations
        """
        self.model_name = model_name
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.max_retries = max_retries
        self.inference_timeout = inference_timeout
        self._engine = None
        self._is_loaded = False

    def _get_vram_usage(self) -> Optional[Dict[str, float]]:
        """
        Get current VRAM usage.

        Returns:
            Dictionary with used and total VRAM in MB, or None if unavailable
        """
        if not GPUTIL_AVAILABLE:
            return None

        try:
            gpus = GPUtil.getGPUs()
            if not gpus:
                return None

            # Sum across all GPUs
            total_used = sum(gpu.memoryUsed for gpu in gpus)
            total_available = sum(gpu.memoryTotal for gpu in gpus)

            return {
                "used_mb": total_used,
                "total_mb": total_available,
                "utilization": total_used / total_available if total_available > 0 else 0
            }
        except Exception as e:
            logger.warning(f"Failed to get VRAM usage: {e}")
            return None

    def load(self):
        """
        Load model using vLLM with error handling and retries.

        Raises:
            RuntimeError: If model loading fails after all retries
            MemoryError: If OOM error occurs
        """
        request_id = str(uuid.uuid4())[:8]
        logger.info(f"[{request_id}] Loading vLLM model: {self.model_name}")

        # Check VRAM before loading
        vram_info = self._get_vram_usage()
        if vram_info:
            logger.info(f"[{request_id}] Current VRAM: {vram_info['used_mb']:.0f}MB / {vram_info['total_mb']:.0f}MB ({vram_info['utilization']*100:.1f}%)")

        if not VLLM_AVAILABLE:
            logger.warning(f"[{request_id}] vLLM not available - using mock mode")
            self._engine = "MOCK_ENGINE"
            self._is_loaded = True
            return

        # Retry loop for transient errors
        last_error = None
        for attempt in range(self.max_retries):
            try:
                self._engine = LLM(
                    model=self.model_name,
                    tensor_parallel_size=self.tensor_parallel_size,
                    gpu_memory_utilization=self.gpu_memory_utilization,
                    max_model_len=self.max_model_len,
                    trust_remote_code=True,
                )
                self._is_loaded = True

                # Log successful load with VRAM usage
                vram_info = self._get_vram_usage()
                if vram_info:
                    logger.info(f"[{request_id}] vLLM model loaded successfully. VRAM: {vram_info['used_mb']:.0f}MB / {vram_info['total_mb']:.0f}MB ({vram_info['utilization']*100:.1f}%)")
                else:
                    logger.info(f"[{request_id}] vLLM model loaded: {self.model_name}")
                return

            except MemoryError as e:
                logger.error(f"[{request_id}] OOM error loading model: {e}", exc_info=True)
                # Trigger cleanup
                self.unload()
                raise MemoryError(f"Out of memory loading {self.model_name}. Try reducing gpu_memory_utilization or use a smaller model.") from e

            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    logger.warning(f"[{request_id}] Failed to load model (attempt {attempt + 1}/{self.max_retries}): {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"[{request_id}] Failed to load model after {self.max_retries} attempts: {e}", exc_info=True)

        # If we get here, all retries failed
        raise RuntimeError(f"Failed to load model {self.model_name} after {self.max_retries} attempts: {last_error}") from last_error

    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        """
        Generate response using vLLM with timeout and error handling.

        Args:
            request: Inference request

        Returns:
            Inference response

        Raises:
            RuntimeError: If model not loaded or inference fails
            TimeoutError: If inference exceeds timeout
            MemoryError: If OOM error occurs during inference
        """
        request_id = str(uuid.uuid4())[:8]
        start_time = time.time()

        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        logger.info(f"[{request_id}] Starting inference with {self.model_name}")

        # Convert messages to prompt
        prompt = self._format_messages(request.messages)
        logger.debug(f"[{request_id}] Prompt length: {len(prompt)} chars")

        if not VLLM_AVAILABLE:
            # Mock response
            logger.warning(f"[{request_id}] Using mock response (vLLM not available)")
            generated_text = f"[Mock vLLM Response from {self.model_name}]\n\nThis is a simulated response. Install vLLM for actual inference."
            prompt_tokens = len(prompt.split()) * 1.3
            completion_tokens = len(generated_text.split()) * 1.3
        else:
            try:
                # Create sampling parameters
                sampling_params = SamplingParams(
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    top_p=request.top_p,
                    stop=request.stop,
                )

                # Run inference with timeout
                loop = asyncio.get_event_loop()
                outputs = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda: self._engine.generate([prompt], sampling_params)
                    ),
                    timeout=self.inference_timeout
                )

                # Extract generated text
                generated_text = outputs[0].outputs[0].text

                # Get token usage from vLLM output
                prompt_tokens = len(outputs[0].prompt_token_ids)
                completion_tokens = len(outputs[0].outputs[0].token_ids)

                logger.info(f"[{request_id}] Inference completed. Tokens: {prompt_tokens} prompt + {completion_tokens} completion")

            except asyncio.TimeoutError:
                logger.error(f"[{request_id}] Inference timeout after {self.inference_timeout}s")
                raise TimeoutError(f"Inference exceeded timeout of {self.inference_timeout}s")

            except MemoryError as e:
                logger.error(f"[{request_id}] OOM error during inference: {e}", exc_info=True)
                # Trigger model unload
                self.unload()
                raise MemoryError(f"Out of memory during inference with {self.model_name}. Model has been unloaded.") from e

            except Exception as e:
                logger.error(f"[{request_id}] Inference failed: {e}", exc_info=True)
                raise RuntimeError(f"Inference failed: {e}") from e

        latency_ms = (time.time() - start_time) * 1000

        # Log VRAM usage after inference
        vram_info = self._get_vram_usage()
        if vram_info:
            logger.debug(f"[{request_id}] Post-inference VRAM: {vram_info['used_mb']:.0f}MB / {vram_info['total_mb']:.0f}MB ({vram_info['utilization']*100:.1f}%)")

        return InferenceResponse(
            content=generated_text,
            model=self.model_name,
            usage={
                "prompt_tokens": int(prompt_tokens),
                "completion_tokens": int(completion_tokens),
                "total_tokens": int(prompt_tokens + completion_tokens),
            },
            latency_ms=latency_ms,
            finish_reason="stop"
        )

    async def generate_stream(
        self,
        request: InferenceRequest
    ) -> AsyncIterator[str]:
        """
        Generate streaming response using vLLM.

        Args:
            request: Inference request

        Yields:
            Response chunks (individual tokens or small token groups)

        Raises:
            RuntimeError: If model not loaded or streaming fails
            TimeoutError: If inference exceeds timeout
        """
        request_id = str(uuid.uuid4())[:8]
        start_time = time.time()

        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        logger.info(f"[{request_id}] Starting streaming inference with {self.model_name}")

        # Convert messages to prompt
        prompt = self._format_messages(request.messages)

        if not VLLM_AVAILABLE:
            # Mock streaming response
            logger.warning(f"[{request_id}] Using mock streaming (vLLM not available)")
            mock_response = f"[Mock streaming from {self.model_name}] "
            words = ["This ", "is ", "a ", "simulated ", "streaming ", "response. ", "Install ", "vLLM ", "for ", "actual ", "inference."]

            for word in words:
                yield word
                await asyncio.sleep(0.05)  # Simulate streaming delay
            return

        try:
            # Create sampling parameters
            sampling_params = SamplingParams(
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                top_p=request.top_p,
                stop=request.stop,
            )

            # For vLLM streaming, we need to use the generate method with async iteration
            # Note: vLLM's streaming API may vary by version, this is a common pattern
            loop = asyncio.get_event_loop()

            # vLLM doesn't have native async streaming in all versions,
            # so we'll simulate it by generating and yielding in chunks
            outputs = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: self._engine.generate([prompt], sampling_params)
                ),
                timeout=self.inference_timeout
            )

            generated_text = outputs[0].outputs[0].text

            # Yield in word-sized chunks for streaming effect
            words = generated_text.split(' ')
            for i, word in enumerate(words):
                if i < len(words) - 1:
                    yield word + ' '
                else:
                    yield word
                await asyncio.sleep(0.01)  # Small delay between chunks

            latency_ms = (time.time() - start_time) * 1000
            logger.info(f"[{request_id}] Streaming completed in {latency_ms:.0f}ms")

        except asyncio.TimeoutError:
            logger.error(f"[{request_id}] Streaming timeout after {self.inference_timeout}s")
            raise TimeoutError(f"Streaming inference exceeded timeout of {self.inference_timeout}s")

        except MemoryError as e:
            logger.error(f"[{request_id}] OOM error during streaming: {e}", exc_info=True)
            self.unload()
            raise MemoryError(f"Out of memory during streaming with {self.model_name}. Model has been unloaded.") from e

        except Exception as e:
            logger.error(f"[{request_id}] Streaming failed: {e}", exc_info=True)
            raise RuntimeError(f"Streaming failed: {e}") from e

    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """
        Format messages into a prompt string.

        Args:
            messages: List of message dicts with 'role' and 'content'

        Returns:
            Formatted prompt
        """
        prompt_parts = []

        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")

            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")

        prompt_parts.append("Assistant:")
        return "\n\n".join(prompt_parts)

    def unload(self):
        """
        Unload model and free VRAM resources.

        This method ensures proper cleanup of GPU memory.
        """
        request_id = str(uuid.uuid4())[:8]
        logger.info(f"[{request_id}] Unloading vLLM model: {self.model_name}")

        if self._engine is not None:
            # Log VRAM before unload
            vram_before = self._get_vram_usage()
            if vram_before:
                logger.info(f"[{request_id}] VRAM before unload: {vram_before['used_mb']:.0f}MB / {vram_before['total_mb']:.0f}MB")

            # Delete engine and trigger garbage collection
            del self._engine
            self._engine = None
            self._is_loaded = False

            # Force garbage collection to free VRAM
            import gc
            gc.collect()

            # Try to clear CUDA cache if available
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.debug(f"[{request_id}] CUDA cache cleared")
            except ImportError:
                pass

            # Log VRAM after unload
            vram_after = self._get_vram_usage()
            if vram_after:
                freed_mb = (vram_before['used_mb'] - vram_after['used_mb']) if vram_before else 0
                logger.info(f"[{request_id}] VRAM after unload: {vram_after['used_mb']:.0f}MB / {vram_after['total_mb']:.0f}MB (freed {freed_mb:.0f}MB)")

        logger.info(f"[{request_id}] Model unloaded successfully")
