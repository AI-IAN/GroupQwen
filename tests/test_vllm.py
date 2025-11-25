"""
Unit Tests for vLLM Handler

Tests model loading, generation, streaming, error handling, and integration.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any

from backend.inference.vllm_handler import (
    VLLMHandler,
    InferenceRequest,
    InferenceResponse,
    ModelConfig,
    MockVLLMEngine,
    create_vllm_handler_from_config
)


# Test Fixtures
@pytest.fixture
def basic_model_config():
    """Basic model configuration for testing."""
    return ModelConfig(
        model_name="Qwen/Qwen3-8B-AWQ",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90,
        max_model_len=4096,
        quantization="awq"
    )


@pytest.fixture
def qwen_handler(basic_model_config):
    """VLLMHandler instance with Qwen model."""
    handler = VLLMHandler(basic_model_config)
    return handler


@pytest.fixture
def olmo_config():
    """OLMo3 model configuration."""
    return ModelConfig(
        model_name="allenai/OLMo3-7B-Instruct",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,
        max_model_len=4096
    )


@pytest.fixture
def sample_messages():
    """Sample chat messages for testing."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ]


@pytest.fixture
def inference_request(sample_messages):
    """Basic inference request."""
    return InferenceRequest(
        messages=sample_messages,
        temperature=0.7,
        max_tokens=100,
        top_p=0.9
    )


# Model Loading Tests
class TestModelLoading:
    """Test model loading functionality."""

    def test_handler_initialization(self, basic_model_config):
        """Test handler initializes with correct configuration."""
        handler = VLLMHandler(basic_model_config)

        assert handler.model_name == "Qwen/Qwen3-8B-AWQ"
        assert handler.config == basic_model_config
        assert not handler.is_loaded
        assert handler._total_requests == 0

    def test_model_load(self, qwen_handler):
        """Test model loads successfully."""
        qwen_handler.load()

        assert qwen_handler.is_loaded
        assert qwen_handler._engine is not None
        assert isinstance(qwen_handler._engine, MockVLLMEngine)

    def test_double_load_warning(self, qwen_handler, caplog):
        """Test loading already loaded model logs warning."""
        qwen_handler.load()
        qwen_handler.load()  # Second load

        assert "already loaded" in caplog.text.lower()

    def test_model_unload(self, qwen_handler):
        """Test model unloads and frees resources."""
        qwen_handler.load()
        assert qwen_handler.is_loaded

        qwen_handler.unload()

        assert not qwen_handler.is_loaded
        assert qwen_handler._engine is None

    def test_unload_not_loaded(self, qwen_handler, caplog):
        """Test unloading non-loaded model logs warning."""
        qwen_handler.unload()

        assert "not loaded" in caplog.text.lower()


# Generation Tests
class TestGeneration:
    """Test model generation functionality."""

    @pytest.mark.asyncio
    async def test_basic_generation(self, qwen_handler, inference_request):
        """Test basic text generation."""
        qwen_handler.load()

        response = await qwen_handler.generate(inference_request)

        assert isinstance(response, InferenceResponse)
        assert response.content
        assert response.model == qwen_handler.model_name
        assert response.usage["prompt_tokens"] > 0
        assert response.usage["completion_tokens"] > 0
        assert response.latency_ms > 0

    @pytest.mark.asyncio
    async def test_generation_not_loaded(self, qwen_handler, inference_request):
        """Test generation fails if model not loaded."""
        with pytest.raises(RuntimeError, match="not loaded"):
            await qwen_handler.generate(inference_request)

    @pytest.mark.asyncio
    async def test_generation_empty_messages(self, qwen_handler):
        """Test generation fails with empty messages."""
        qwen_handler.load()

        request = InferenceRequest(messages=[], temperature=0.7)

        with pytest.raises(ValueError, match="cannot be empty"):
            await qwen_handler.generate(request)

    @pytest.mark.asyncio
    async def test_generation_updates_stats(self, qwen_handler, inference_request):
        """Test generation updates handler statistics."""
        qwen_handler.load()

        initial_requests = qwen_handler._total_requests
        await qwen_handler.generate(inference_request)

        assert qwen_handler._total_requests == initial_requests + 1
        assert qwen_handler._total_tokens_generated > 0

    @pytest.mark.asyncio
    async def test_generation_with_custom_params(self, qwen_handler, sample_messages):
        """Test generation with custom parameters."""
        qwen_handler.load()

        request = InferenceRequest(
            messages=sample_messages,
            temperature=0.3,
            max_tokens=50,
            top_p=0.95,
            stop=["END"]
        )

        response = await qwen_handler.generate(request)

        assert response.content
        assert response.latency_ms > 0


# Streaming Tests
class TestStreaming:
    """Test streaming generation functionality."""

    @pytest.mark.asyncio
    async def test_streaming_generation(self, qwen_handler, inference_request):
        """Test streaming text generation."""
        qwen_handler.load()

        chunks = []
        async for chunk in qwen_handler.generate_stream(inference_request):
            chunks.append(chunk)

        assert len(chunks) > 0
        full_text = "".join(chunks)
        assert full_text

    @pytest.mark.asyncio
    async def test_streaming_not_loaded(self, qwen_handler, inference_request):
        """Test streaming fails if model not loaded."""
        with pytest.raises(RuntimeError, match="not loaded"):
            async for _ in qwen_handler.generate_stream(inference_request):
                pass

    @pytest.mark.asyncio
    async def test_streaming_empty_messages(self, qwen_handler):
        """Test streaming fails with empty messages."""
        qwen_handler.load()

        request = InferenceRequest(messages=[], temperature=0.7)

        with pytest.raises(ValueError, match="cannot be empty"):
            async for _ in qwen_handler.generate_stream(request):
                pass


# Prompt Formatting Tests
class TestPromptFormatting:
    """Test message-to-prompt formatting."""

    def test_qwen_format_single_message(self, qwen_handler):
        """Test Qwen3 chat format with single message."""
        messages = [{"role": "user", "content": "Hello"}]

        prompt = qwen_handler._format_messages(messages)

        assert "<|im_start|>user" in prompt
        assert "Hello" in prompt
        assert "<|im_end|>" in prompt
        assert "<|im_start|>assistant" in prompt

    def test_qwen_format_multi_message(self, qwen_handler, sample_messages):
        """Test Qwen3 chat format with multiple messages."""
        prompt = qwen_handler._format_messages(sample_messages)

        assert "<|im_start|>system" in prompt
        assert "helpful assistant" in prompt
        assert "<|im_start|>user" in prompt
        assert "capital of France" in prompt

    def test_olmo_format(self, olmo_config):
        """Test OLMo3 chat format."""
        handler = VLLMHandler(olmo_config)
        messages = [
            {"role": "system", "content": "System message"},
            {"role": "user", "content": "User query"}
        ]

        prompt = handler._format_messages(messages)

        assert "<|system|>" in prompt
        assert "<|user|>" in prompt
        assert "<|assistant|>" in prompt

    def test_format_with_assistant_message(self, qwen_handler):
        """Test formatting with assistant message in history."""
        messages = [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
            {"role": "user", "content": "Second question"}
        ]

        prompt = qwen_handler._format_messages(messages)

        assert prompt.count("<|im_start|>user") == 2
        assert prompt.count("<|im_start|>assistant") == 2  # 1 in history + 1 final


# Statistics Tests
class TestStatistics:
    """Test handler statistics tracking."""

    def test_get_stats_initial(self, qwen_handler):
        """Test statistics for unloaded handler."""
        stats = qwen_handler.get_stats()

        assert stats["model_name"] == "Qwen/Qwen3-8B-AWQ"
        assert stats["is_loaded"] is False
        assert stats["total_requests"] == 0
        assert stats["avg_latency_ms"] == 0.0

    @pytest.mark.asyncio
    async def test_get_stats_after_generation(self, qwen_handler, inference_request):
        """Test statistics after generation."""
        qwen_handler.load()
        await qwen_handler.generate(inference_request)

        stats = qwen_handler.get_stats()

        assert stats["is_loaded"] is True
        assert stats["total_requests"] == 1
        assert stats["total_tokens_generated"] > 0
        assert stats["avg_latency_ms"] > 0
        assert stats["avg_tokens_per_request"] > 0


# Error Handling Tests
class TestErrorHandling:
    """Test error handling and recovery."""

    @pytest.mark.asyncio
    async def test_retry_on_failure(self, qwen_handler, inference_request):
        """Test retry logic on generation failure."""
        qwen_handler.load()

        # Mock engine to fail first time, succeed second time
        original_generate = qwen_handler._engine.generate
        call_count = {"count": 0}

        async def mock_generate_with_failure(*args, **kwargs):
            call_count["count"] += 1
            if call_count["count"] == 1:
                raise RuntimeError("Simulated failure")
            return await original_generate(*args, **kwargs)

        qwen_handler._engine.generate = mock_generate_with_failure

        # Should succeed after retry
        response = await qwen_handler.generate(inference_request)

        assert response.content
        assert call_count["count"] == 2  # Failed once, succeeded on retry

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self, qwen_handler, inference_request):
        """Test failure after max retries."""
        qwen_handler.load()

        # Mock engine to always fail
        async def mock_generate_fail(*args, **kwargs):
            raise RuntimeError("Persistent failure")

        qwen_handler._engine.generate = mock_generate_fail

        # Should fail after max retries
        with pytest.raises(RuntimeError, match="after 3 attempts"):
            await qwen_handler.generate(inference_request)


# Factory Function Tests
class TestFactoryFunction:
    """Test factory function for creating handlers."""

    def test_create_from_config_basic(self):
        """Test creating handler from config dict."""
        config = {
            "name": "Qwen/Qwen3-8B-AWQ",
            "quantization": "awq",
            "max_tokens": 4096,
            "vllm_config": {
                "gpu_memory_utilization": 0.90,
                "max_model_len": 4096
            }
        }

        handler = create_vllm_handler_from_config("qwen3-8b", config)

        assert isinstance(handler, VLLMHandler)
        assert handler.model_name == "Qwen/Qwen3-8B-AWQ"
        assert handler.config.gpu_memory_utilization == 0.90

    def test_create_from_config_with_rope_scaling(self):
        """Test creating handler with rope scaling config."""
        config = {
            "name": "Qwen/Qwen3-32B-AWQ",
            "quantization": "awq",
            "max_tokens": 16384,
            "vllm_config": {
                "gpu_memory_utilization": 0.95,
                "max_model_len": 16384,
                "rope_scaling": {
                    "type": "yarn",
                    "factor": 4.0
                }
            }
        }

        handler = create_vllm_handler_from_config("qwen3-32b", config)

        assert handler.config.rope_scaling is not None
        assert handler.config.rope_scaling["type"] == "yarn"

    def test_create_from_config_defaults(self):
        """Test creating handler with default values."""
        config = {
            "name": "allenai/OLMo3-7B",
            "max_tokens": 4096
        }

        handler = create_vllm_handler_from_config("olmo3-7b", config)

        assert handler.config.gpu_memory_utilization == 0.90  # Default
        assert handler.config.tensor_parallel_size == 1  # Default


# Mock Engine Tests
class TestMockEngine:
    """Test MockVLLMEngine functionality."""

    @pytest.mark.asyncio
    async def test_mock_generate(self):
        """Test mock engine generation."""
        engine = MockVLLMEngine("test-model")

        prompt = "Test prompt"
        params = {"temperature": 0.7, "max_tokens": 100}

        text, prompt_tokens, completion_tokens = await engine.generate(prompt, params)

        assert isinstance(text, str)
        assert "Mock Response" in text
        assert prompt_tokens > 0
        assert completion_tokens > 0

    @pytest.mark.asyncio
    async def test_mock_stream(self):
        """Test mock engine streaming."""
        engine = MockVLLMEngine("test-model")

        prompt = "Test prompt"
        params = {"temperature": 0.7, "max_tokens": 100}

        chunks = []
        async for chunk in engine.stream_generate(prompt, params):
            chunks.append(chunk)

        assert len(chunks) > 0
        full_text = "".join(chunks)
        assert "Mock Response" in full_text


# Integration Tests
class TestIntegration:
    """Integration tests for complete workflows."""

    @pytest.mark.asyncio
    async def test_full_workflow(self, qwen_handler, sample_messages):
        """Test complete workflow: load -> generate -> unload."""
        # Load
        qwen_handler.load()
        assert qwen_handler.is_loaded

        # Generate
        request = InferenceRequest(
            messages=sample_messages,
            temperature=0.7,
            max_tokens=100
        )
        response = await qwen_handler.generate(request)

        assert response.content
        assert response.usage["total_tokens"] > 0

        # Check stats
        stats = qwen_handler.get_stats()
        assert stats["total_requests"] == 1

        # Unload
        qwen_handler.unload()
        assert not qwen_handler.is_loaded

    @pytest.mark.asyncio
    async def test_multiple_generations(self, qwen_handler, sample_messages):
        """Test multiple sequential generations."""
        qwen_handler.load()

        request = InferenceRequest(messages=sample_messages, max_tokens=50)

        # Generate 3 times
        for i in range(3):
            response = await qwen_handler.generate(request)
            assert response.content

        # Check stats
        stats = qwen_handler.get_stats()
        assert stats["total_requests"] == 3
        assert stats["avg_latency_ms"] > 0

    @pytest.mark.asyncio
    async def test_concurrent_generations(self, qwen_handler, sample_messages):
        """Test concurrent generation requests."""
        qwen_handler.load()

        request = InferenceRequest(messages=sample_messages, max_tokens=50)

        # Generate 5 requests concurrently
        tasks = [qwen_handler.generate(request) for _ in range(5)]
        responses = await asyncio.gather(*tasks)

        assert len(responses) == 5
        for response in responses:
            assert response.content

        stats = qwen_handler.get_stats()
        assert stats["total_requests"] == 5


# Performance Tests
class TestPerformance:
    """Performance-related tests."""

    @pytest.mark.asyncio
    async def test_latency_tracking(self, qwen_handler, inference_request):
        """Test that latency is properly tracked."""
        qwen_handler.load()

        response = await qwen_handler.generate(inference_request)

        assert response.latency_ms > 0
        assert response.latency_ms < 10000  # Should be under 10 seconds for mock

    @pytest.mark.asyncio
    async def test_token_throughput(self, qwen_handler, inference_request):
        """Test token generation throughput calculation."""
        qwen_handler.load()

        response = await qwen_handler.generate(inference_request)

        tokens_per_second = (
            response.usage["completion_tokens"] / (response.latency_ms / 1000)
        )

        assert tokens_per_second > 0  # Should have positive throughput


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
