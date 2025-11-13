"""
llama.cpp Inference Handler

Handles inference using llama.cpp for CPU/edge device inference.
"""

from typing import List, Dict, Optional, AsyncIterator
import logging
from dataclasses import dataclass
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class LlamaCppRequest:
    """Request for llama.cpp inference."""
    messages: List[Dict[str, str]]
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    stream: bool = False


@dataclass
class LlamaCppResponse:
    """Response from llama.cpp inference."""
    content: str
    model: str
    usage: Dict[str, int]
    latency_ms: float
    finish_reason: str = "stop"


class LlamaCppHandler:
    """
    llama.cpp inference handler for edge devices and CPU inference.

    Supports:
    - GGUF quantized models
    - CPU and Metal (M-series Mac) acceleration
    - Low memory footprint
    - Optimized for edge deployment
    """

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 2048,
        n_gpu_layers: int = 0,
        n_threads: Optional[int] = None,
        use_metal: bool = False
    ):
        """
        Initialize llama.cpp handler.

        Args:
            model_path: Path to GGUF model file
            n_ctx: Context window size
            n_gpu_layers: Number of layers to offload to GPU
            n_threads: Number of CPU threads (None = auto)
            use_metal: Use Metal acceleration (for M-series Mac)
        """
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.n_threads = n_threads
        self.use_metal = use_metal
        self._llm = None

    def load(self):
        """
        Load model using llama.cpp.

        This is a placeholder - actual implementation would use:
        from llama_cpp import Llama
        """
        logger.info(f"Loading llama.cpp model: {self.model_path}")

        # Placeholder for actual llama.cpp initialization
        # self._llm = Llama(
        #     model_path=self.model_path,
        #     n_ctx=self.n_ctx,
        #     n_gpu_layers=self.n_gpu_layers,
        #     n_threads=self.n_threads,
        #     use_mlock=True,  # Keep model in RAM
        #     use_mmap=True,   # Use memory mapping
        #     verbose=False,
        # )

        logger.info(f"llama.cpp model loaded: {self.model_path}")

    async def generate(self, request: LlamaCppRequest) -> LlamaCppResponse:
        """
        Generate response using llama.cpp.

        Args:
            request: Inference request

        Returns:
            Inference response
        """
        import time
        start_time = time.time()

        # Format messages to prompt
        prompt = self._format_messages(request.messages)

        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()

        def _generate():
            # Placeholder for actual llama.cpp inference
            # output = self._llm(
            #     prompt,
            #     max_tokens=request.max_tokens,
            #     temperature=request.temperature,
            #     top_p=request.top_p,
            #     top_k=request.top_k,
            #     repeat_penalty=request.repeat_penalty,
            #     stop=["User:", "Human:"],
            # )
            # return output["choices"][0]["text"]

            return f"[llama.cpp Response from {self.model_path}]"

        generated_text = await loop.run_in_executor(None, _generate)

        latency_ms = (time.time() - start_time) * 1000

        return LlamaCppResponse(
            content=generated_text,
            model=self.model_path,
            usage={
                "prompt_tokens": len(prompt.split()) * 1.3,
                "completion_tokens": len(generated_text.split()) * 1.3,
                "total_tokens": len((prompt + generated_text).split()) * 1.3,
            },
            latency_ms=latency_ms
        )

    async def generate_stream(
        self,
        request: LlamaCppRequest
    ) -> AsyncIterator[str]:
        """
        Generate streaming response.

        Args:
            request: Inference request

        Yields:
            Response chunks
        """
        # Placeholder for streaming
        # In actual implementation, would use llama.cpp's streaming API

        response = await self.generate(request)

        # Simulate streaming by yielding chunks
        words = response.content.split()
        for i in range(0, len(words), 5):
            chunk = " ".join(words[i:i+5]) + " "
            yield chunk
            await asyncio.sleep(0.01)  # Simulate streaming delay

    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """
        Format messages into a prompt string for llama.cpp.

        Args:
            messages: List of message dicts

        Returns:
            Formatted prompt
        """
        prompt_parts = []

        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")

            if role == "system":
                prompt_parts.append(f"<<SYS>>\n{content}\n<</SYS>>")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")

        prompt_parts.append("Assistant:")
        return "\n\n".join(prompt_parts)

    def unload(self):
        """Unload model and free resources."""
        logger.info(f"Unloading llama.cpp model: {self.model_path}")
        self._llm = None
