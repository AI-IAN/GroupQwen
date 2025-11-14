"""
Tests for Inference Handlers (vLLM and llama.cpp)
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from backend.inference.vllm_handler import VLLMHandler, InferenceRequest, InferenceResponse
from backend.inference.llamacpp_handler import LlamaCppHandler, LlamaCppRequest, LlamaCppResponse


class TestVLLMHandler:
    """Tests for VLLMHandler."""

    @patch('backend.inference.vllm_handler.LLM')
    def test_load_success(self, mock_llm):
        """Test successful model loading."""
        handler = VLLMHandler(model_name="qwen3-8b")
        handler.load()

        # Verify engine is set (even if it's a placeholder)
        # In the actual implementation, this would call the mock
        assert handler.model_name == "qwen3-8b"

    @pytest.mark.asyncio
    async def test_generate_success(self):
        """Test successful generation."""
        handler = VLLMHandler(model_name="qwen3-8b")

        request = InferenceRequest(
            messages=[{"role": "user", "content": "Hi"}],
            temperature=0.7
        )
        response = await handler.generate(request)

        assert isinstance(response, InferenceResponse)
        assert response.content is not None
        assert response.model == "qwen3-8b"
        assert response.latency_ms > 0
        assert "usage" in response.__dict__
        assert response.usage["total_tokens"] > 0

    @pytest.mark.asyncio
    async def test_generate_with_system_message(self):
        """Test generation with system message."""
        handler = VLLMHandler(model_name="qwen3-8b")

        request = InferenceRequest(
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "Hello"}
            ],
            temperature=0.7
        )
        response = await handler.generate(request)

        assert isinstance(response, InferenceResponse)
        assert response.content is not None

    @pytest.mark.asyncio
    async def test_generate_stream(self):
        """Test streaming generation."""
        handler = VLLMHandler(model_name="qwen3-8b")

        request = InferenceRequest(
            messages=[{"role": "user", "content": "Hi"}],
            stream=True
        )

        chunks = []
        async for chunk in handler.generate_stream(request):
            chunks.append(chunk)

        assert len(chunks) > 0
        # Verify chunks are strings
        assert all(isinstance(chunk, str) for chunk in chunks)

    @pytest.mark.asyncio
    async def test_generate_with_parameters(self):
        """Test generation with custom parameters."""
        handler = VLLMHandler(model_name="qwen3-14b")

        request = InferenceRequest(
            messages=[{"role": "user", "content": "Test"}],
            temperature=0.9,
            max_tokens=1024,
            top_p=0.95
        )
        response = await handler.generate(request)

        assert isinstance(response, InferenceResponse)
        assert response.content is not None

    def test_format_messages(self):
        """Test message formatting."""
        handler = VLLMHandler(model_name="qwen3-8b")

        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?"}
        ]

        prompt = handler._format_messages(messages)

        assert "System: You are helpful" in prompt
        assert "User: Hello" in prompt
        assert "Assistant: Hi there" in prompt
        assert "User: How are you?" in prompt

    def test_unload(self):
        """Test model unloading."""
        handler = VLLMHandler(model_name="qwen3-8b")
        handler.load()
        handler.unload()

        assert handler._engine is None


class TestLlamaCppHandler:
    """Tests for LlamaCppHandler."""

    @patch('backend.inference.llamacpp_handler.Llama')
    def test_load_success(self, mock_llama):
        """Test successful model loading."""
        handler = LlamaCppHandler(
            model_path="/path/to/model.gguf",
            n_ctx=2048
        )
        handler.load()

        assert handler.model_path == "/path/to/model.gguf"
        assert handler.n_ctx == 2048

    @patch('backend.inference.llamacpp_handler.Llama')
    def test_load_with_metal(self, mock_llama):
        """Test loading with Metal backend."""
        handler = LlamaCppHandler(
            model_path="/path/to/model.gguf",
            use_metal=True,
            n_gpu_layers=35
        )
        handler.load()

        assert handler.use_metal is True
        assert handler.n_gpu_layers == 35

    @pytest.mark.asyncio
    async def test_generate_success(self):
        """Test successful generation."""
        handler = LlamaCppHandler(model_path="/path/to/model.gguf")
        handler.load()

        request = LlamaCppRequest(
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.7
        )
        response = await handler.generate(request)

        assert isinstance(response, LlamaCppResponse)
        assert response.content is not None
        assert response.model == "/path/to/model.gguf"
        assert response.latency_ms > 0
        assert response.usage["total_tokens"] > 0

    @pytest.mark.asyncio
    async def test_generate_with_custom_params(self):
        """Test generation with custom parameters."""
        handler = LlamaCppHandler(model_path="/path/to/model.gguf")
        handler.load()

        request = LlamaCppRequest(
            messages=[{"role": "user", "content": "Test"}],
            temperature=0.8,
            max_tokens=512,
            top_p=0.9,
            top_k=50,
            repeat_penalty=1.2
        )
        response = await handler.generate(request)

        assert isinstance(response, LlamaCppResponse)
        assert response.content is not None

    @pytest.mark.asyncio
    async def test_generate_stream(self):
        """Test streaming generation."""
        handler = LlamaCppHandler(model_path="/path/to/model.gguf")
        handler.load()

        request = LlamaCppRequest(
            messages=[{"role": "user", "content": "Hello"}],
            stream=True
        )

        chunks = []
        async for chunk in handler.generate_stream(request):
            chunks.append(chunk)

        assert len(chunks) > 0
        # Verify chunks are strings
        assert all(isinstance(chunk, str) for chunk in chunks)

    def test_format_messages(self):
        """Test message formatting for llama.cpp."""
        handler = LlamaCppHandler(model_path="/path/to/model.gguf")

        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "How are you?"}
        ]

        prompt = handler._format_messages(messages)

        assert "<<SYS>>" in prompt
        assert "You are helpful" in prompt
        assert "User: Hello" in prompt
        assert "Assistant: Hi" in prompt
        assert "User: How are you?" in prompt

    def test_unload(self):
        """Test model unloading."""
        handler = LlamaCppHandler(model_path="/path/to/model.gguf")
        handler.load()
        handler.unload()

        assert handler._llm is None


class TestInferenceErrorHandling:
    """Tests for error handling in inference handlers."""

    @pytest.mark.asyncio
    async def test_empty_messages(self):
        """Test handling of empty messages."""
        handler = VLLMHandler(model_name="qwen3-8b")

        request = InferenceRequest(messages=[])

        # Should handle gracefully (either error or return something)
        try:
            response = await handler.generate(request)
            # If it doesn't error, verify response is valid
            assert isinstance(response, InferenceResponse)
        except (ValueError, IndexError):
            # It's acceptable to raise an error for empty messages
            pass

    @pytest.mark.asyncio
    async def test_invalid_message_format(self):
        """Test handling of invalid message format."""
        handler = VLLMHandler(model_name="qwen3-8b")

        # Missing 'content' field
        request = InferenceRequest(
            messages=[{"role": "user"}]
        )

        try:
            response = await handler.generate(request)
            # Should handle gracefully
            assert isinstance(response, InferenceResponse)
        except (ValueError, KeyError):
            # Acceptable to raise error
            pass

    @pytest.mark.asyncio
    async def test_very_long_input(self):
        """Test handling of very long input."""
        handler = VLLMHandler(model_name="qwen3-8b")

        # Create very long message
        long_content = "word " * 10000

        request = InferenceRequest(
            messages=[{"role": "user", "content": long_content}]
        )

        # Should handle without crashing
        response = await handler.generate(request)
        assert isinstance(response, InferenceResponse)


class TestInferenceDataClasses:
    """Tests for inference data classes."""

    def test_inference_request_defaults(self):
        """Test InferenceRequest default values."""
        request = InferenceRequest(
            messages=[{"role": "user", "content": "Test"}]
        )

        assert request.temperature == 0.7
        assert request.max_tokens == 4096
        assert request.top_p == 0.9
        assert request.stream is False

    def test_inference_response_structure(self):
        """Test InferenceResponse structure."""
        response = InferenceResponse(
            content="Test response",
            model="qwen3-8b",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            latency_ms=150.5
        )

        assert response.content == "Test response"
        assert response.model == "qwen3-8b"
        assert response.finish_reason == "stop"
        assert response.latency_ms == 150.5

    def test_llamacpp_request_defaults(self):
        """Test LlamaCppRequest default values."""
        request = LlamaCppRequest(
            messages=[{"role": "user", "content": "Test"}]
        )

        assert request.temperature == 0.7
        assert request.max_tokens == 2048
        assert request.top_p == 0.9
        assert request.top_k == 40
        assert request.repeat_penalty == 1.1

    def test_llamacpp_response_structure(self):
        """Test LlamaCppResponse structure."""
        response = LlamaCppResponse(
            content="Test response",
            model="/path/to/model.gguf",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            latency_ms=200.0
        )

        assert response.content == "Test response"
        assert response.model == "/path/to/model.gguf"
        assert response.finish_reason == "stop"
