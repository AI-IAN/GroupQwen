"""
Tests for Inference Handlers (vLLM and llama.cpp)
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from backend.inference.vllm_handler import VLLMHandler, InferenceRequest, InferenceResponse
from backend.inference.llamacpp_handler import LlamaCppHandler, LlamaCppRequest, LlamaCppResponse


class TestVLLMHandler:
    """Test suite for VLLMHandler."""

    @patch('backend.inference.vllm_handler.logger')
    def test_load_success(self, mock_logger):
        """Test successful model loading."""
        handler = VLLMHandler(model_name="qwen3-8b")
        handler.load()

        # Verify logging was called
        assert mock_logger.info.called
        mock_logger.info.assert_any_call("Loading vLLM model: qwen3-8b")
        mock_logger.info.assert_any_call("vLLM model loaded: qwen3-8b")

    @pytest.mark.asyncio
    async def test_generate_success(self):
        """Test successful generation."""
        handler = VLLMHandler(model_name="qwen3-8b")

        request = InferenceRequest(
            messages=[{"role": "user", "content": "Hi"}],
            temperature=0.7
        )
        response = await handler.generate(request)

        # Verify response structure
        assert isinstance(response, InferenceResponse)
        assert response.content is not None
        assert response.model == "qwen3-8b"
        assert response.latency_ms > 0
        assert "usage" in response.__dict__
        assert response.usage["prompt_tokens"] > 0
        assert response.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_generate_with_custom_params(self):
        """Test generation with custom parameters."""
        handler = VLLMHandler(model_name="qwen3-14b")

        request = InferenceRequest(
            messages=[{"role": "user", "content": "Explain quantum computing"}],
            temperature=0.9,
            max_tokens=2048,
            top_p=0.95
        )
        response = await handler.generate(request)

        assert response is not None
        assert response.model == "qwen3-14b"
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

        # Verify at least one chunk was generated
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)

    @pytest.mark.asyncio
    async def test_generate_with_system_message(self):
        """Test generation with system message."""
        handler = VLLMHandler(model_name="qwen3-8b")

        request = InferenceRequest(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"}
            ]
        )
        response = await handler.generate(request)

        assert response is not None
        assert response.content is not None

    @pytest.mark.asyncio
    async def test_generate_with_conversation_history(self):
        """Test generation with conversation history."""
        handler = VLLMHandler(model_name="qwen3-8b")

        request = InferenceRequest(
            messages=[
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4"},
                {"role": "user", "content": "What about 2+3?"}
            ]
        )
        response = await handler.generate(request)

        assert response is not None

    def test_format_messages_simple(self):
        """Test message formatting with simple user message."""
        handler = VLLMHandler(model_name="qwen3-8b")

        messages = [{"role": "user", "content": "Hello"}]
        formatted = handler._format_messages(messages)

        assert "User: Hello" in formatted
        assert "Assistant:" in formatted

    def test_format_messages_with_system(self):
        """Test message formatting with system message."""
        handler = VLLMHandler(model_name="qwen3-8b")

        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hi"}
        ]
        formatted = handler._format_messages(messages)

        assert "System: You are helpful" in formatted
        assert "User: Hi" in formatted

    def test_format_messages_conversation(self):
        """Test message formatting with full conversation."""
        handler = VLLMHandler(model_name="qwen3-8b")

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"}
        ]
        formatted = handler._format_messages(messages)

        assert "User: Hello" in formatted
        assert "Assistant: Hi there!" in formatted
        assert "User: How are you?" in formatted

    def test_unload(self):
        """Test model unloading."""
        handler = VLLMHandler(model_name="qwen3-8b")
        handler._engine = Mock()

        handler.unload()

        assert handler._engine is None

    def test_initialization_parameters(self):
        """Test handler initialization with custom parameters."""
        handler = VLLMHandler(
            model_name="qwen3-32b",
            tensor_parallel_size=2,
            gpu_memory_utilization=0.85,
            max_model_len=16384
        )

        assert handler.model_name == "qwen3-32b"
        assert handler.tensor_parallel_size == 2
        assert handler.gpu_memory_utilization == 0.85
        assert handler.max_model_len == 16384


class TestLlamaCppHandler:
    """Test suite for LlamaCppHandler."""

    @patch('backend.inference.llamacpp_handler.logger')
    def test_load_success(self, mock_logger):
        """Test successful model loading."""
        handler = LlamaCppHandler(model_path="/path/to/model.gguf")
        handler.load()

        # Verify logging
        assert mock_logger.info.called
        mock_logger.info.assert_any_call("Loading llama.cpp model: /path/to/model.gguf")

    @patch('backend.inference.llamacpp_handler.logger')
    def test_load_with_metal(self, mock_logger):
        """Test loading with Metal backend."""
        handler = LlamaCppHandler(
            model_path="/path/to/model.gguf",
            use_metal=True,
            n_gpu_layers=32
        )
        handler.load()

        assert handler.use_metal is True
        assert handler.n_gpu_layers == 32

    @pytest.mark.asyncio
    async def test_generate_success(self):
        """Test successful generation."""
        handler = LlamaCppHandler(model_path="/path/to/model.gguf")

        request = LlamaCppRequest(
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.7
        )
        response = await handler.generate(request)

        # Verify response structure
        assert isinstance(response, LlamaCppResponse)
        assert response.content is not None
        assert response.model == "/path/to/model.gguf"
        assert response.latency_ms > 0
        assert "usage" in response.__dict__
        assert response.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_generate_with_custom_params(self):
        """Test generation with custom parameters."""
        handler = LlamaCppHandler(model_path="/path/to/model.gguf")

        request = LlamaCppRequest(
            messages=[{"role": "user", "content": "Test"}],
            temperature=0.8,
            max_tokens=1024,
            top_p=0.9,
            top_k=50,
            repeat_penalty=1.2
        )
        response = await handler.generate(request)

        assert response is not None
        assert response.content is not None

    @pytest.mark.asyncio
    async def test_generate_stream(self):
        """Test streaming generation."""
        handler = LlamaCppHandler(model_path="/path/to/model.gguf")

        request = LlamaCppRequest(
            messages=[{"role": "user", "content": "Hello"}],
            stream=True
        )

        chunks = []
        async for chunk in handler.generate_stream(request):
            chunks.append(chunk)

        # Verify streaming produced chunks
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)

    @pytest.mark.asyncio
    async def test_generate_with_conversation(self):
        """Test generation with conversation history."""
        handler = LlamaCppHandler(model_path="/path/to/model.gguf")

        request = LlamaCppRequest(
            messages=[
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
                {"role": "user", "content": "How are you?"}
            ]
        )
        response = await handler.generate(request)

        assert response is not None

    def test_format_messages_simple(self):
        """Test message formatting."""
        handler = LlamaCppHandler(model_path="/path/to/model.gguf")

        messages = [{"role": "user", "content": "Test"}]
        formatted = handler._format_messages(messages)

        assert "User: Test" in formatted
        assert "Assistant:" in formatted

    def test_format_messages_with_system(self):
        """Test message formatting with system message."""
        handler = LlamaCppHandler(model_path="/path/to/model.gguf")

        messages = [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Hi"}
        ]
        formatted = handler._format_messages(messages)

        assert "<<SYS>>" in formatted
        assert "Be helpful" in formatted
        assert "User: Hi" in formatted

    def test_unload(self):
        """Test model unloading."""
        handler = LlamaCppHandler(model_path="/path/to/model.gguf")
        handler._llm = Mock()

        handler.unload()

        assert handler._llm is None

    def test_initialization_parameters(self):
        """Test handler initialization with custom parameters."""
        handler = LlamaCppHandler(
            model_path="/custom/path.gguf",
            n_ctx=4096,
            n_gpu_layers=16,
            n_threads=8,
            use_metal=True
        )

        assert handler.model_path == "/custom/path.gguf"
        assert handler.n_ctx == 4096
        assert handler.n_gpu_layers == 16
        assert handler.n_threads == 8
        assert handler.use_metal is True


# Error Handling Tests
class TestInferenceErrorHandling:
    """Test error handling scenarios."""

    @pytest.mark.asyncio
    async def test_empty_messages(self):
        """Test handling of empty messages."""
        handler = VLLMHandler(model_name="qwen3-8b")

        request = InferenceRequest(messages=[])
        response = await handler.generate(request)

        # Should handle gracefully even with empty messages
        assert response is not None

    @pytest.mark.asyncio
    async def test_very_long_input(self):
        """Test handling of very long input."""
        handler = VLLMHandler(model_name="qwen3-32b")

        # Create a very long message
        long_content = "test " * 10000
        request = InferenceRequest(
            messages=[{"role": "user", "content": long_content}]
        )
        response = await handler.generate(request)

        assert response is not None

    @pytest.mark.asyncio
    async def test_special_characters(self):
        """Test handling of special characters."""
        handler = LlamaCppHandler(model_path="/path/to/model.gguf")

        request = LlamaCppRequest(
            messages=[{"role": "user", "content": "Test with 特殊字符 and émojis 🚀"}]
        )
        response = await handler.generate(request)

        assert response is not None

    def test_missing_role_in_message(self):
        """Test handling of message without role."""
        handler = VLLMHandler(model_name="qwen3-8b")

        # Message with missing role should still be formatted
        messages = [{"content": "Hello"}]
        formatted = handler._format_messages(messages)

        # Should default to user role
        assert "User: Hello" in formatted
