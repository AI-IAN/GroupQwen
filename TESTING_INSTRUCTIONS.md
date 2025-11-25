# vLLM Handler Testing Instructions

This document provides instructions for testing the newly implemented vLLM inference handler.

## What Was Implemented

### 1. Complete vLLM Handler (`backend/inference/vllm_handler.py`)
- Full vLLM integration with proper model loading
- Support for both Qwen3 and OLMo3 model families
- Streaming and non-streaming generation
- Retry logic with exponential backoff
- GPU memory management
- Comprehensive logging and metrics

### 2. API Integration (`backend/api/routes.py`)
- Wired vLLM handler to `/v1/chat/completions` endpoint
- Handler caching for efficient model reuse
- Fallback to mock responses for non-vLLM models
- Proper error handling and metrics logging

### 3. Test Suite (`tests/test_vllm.py`)
- 31 comprehensive unit tests (all passing)
- Coverage: loading, generation, streaming, error handling, integration

## Quick Start Testing

### Option 1: Unit Tests (No GPU Required)

The test suite uses a mock vLLM engine and works without GPU:

```bash
# Run all vLLM tests
python -m pytest tests/test_vllm.py -v

# Run specific test class
python -m pytest tests/test_vllm.py::TestGeneration -v

# Run with coverage
python -m pytest tests/test_vllm.py --cov=backend.inference.vllm_handler
```

**Expected Output:**
```
============================== 31 passed in 4.79s ==============================
```

### Option 2: API Testing with Mock Engine (No GPU Required)

Start the API server (uses mock engine when vLLM not installed):

```bash
# Start the server
python -m backend.api.main

# In another terminal, test the API
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "model": "qwen3-8b"
  }'
```

**Expected Response:**
```json
{
  "id": "chatcmpl-abc123",
  "model": "Qwen/Qwen3-8B-AWQ",
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "[Mock Response from Qwen/Qwen3-8B-AWQ]..."
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 25,
    "total_tokens": 35
  },
  "metadata": {
    "cache_hit": false,
    "model_selected": "qwen3-8b",
    "latency_ms": 150.5
  }
}
```

### Option 3: Production Testing with Real vLLM (GPU Required)

**Prerequisites:**
- NVIDIA GPU with CUDA support
- vLLM installed: `pip install vllm`
- At least 6GB VRAM for Qwen3-8B or 5GB for OLMo3-7B

**Installation:**
```bash
# Install vLLM (requires CUDA)
pip install vllm

# Verify CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

**Test with Real Model:**

1. Start the API server:
```bash
python -m backend.api.main
```

2. Send a request to a vLLM-configured model:
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Explain quantum computing in simple terms."}
    ],
    "model": "qwen3-8b",
    "temperature": 0.7,
    "max_tokens": 200
  }'
```

3. Test with OLMo3:
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Write a haiku about coding."}
    ],
    "model": "olmo3-7b-instruct",
    "temperature": 0.9,
    "max_tokens": 100
  }'
```

**What to Verify:**
- ✓ Model loads successfully (check logs for "vLLM model loaded")
- ✓ Response contains actual model-generated text (not mock)
- ✓ Token counts are accurate
- ✓ Latency is reasonable (varies by GPU)
- ✓ GPU memory is properly managed (use `nvidia-smi`)

## Testing Different Features

### 1. Test Routing Logic

The router automatically selects the optimal model based on query complexity:

```bash
# Simple query -> should route to qwen3-8b
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# Complex query -> should route to qwen3-14b or qwen3-32b
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{
      "role": "user",
      "content": "Analyze the geopolitical implications of renewable energy transition on OPEC nations."
    }]
  }'
```

Check the `metadata.reasoning` field in the response to see routing decision.

### 2. Test Caching

Send the same query twice to test semantic caching:

```bash
# First request (cache miss)
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "What is 2+2?"}]}'

# Second request (should hit cache)
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "What is 2+2?"}]}'
```

Check `metadata.cache_hit` field in the response.

### 3. Test Different Models

Test each vLLM-configured model:

```bash
# Qwen3-8B (fast, balanced)
curl -X POST http://localhost:8000/v1/chat/completions \
  -d '{"messages": [{"role": "user", "content": "Test"}], "model": "qwen3-8b"}'

# Qwen3-14B (moderate complexity)
curl -X POST http://localhost:8000/v1/chat/completions \
  -d '{"messages": [{"role": "user", "content": "Test"}], "model": "qwen3-14b"}'

# Qwen3-32B (complex reasoning)
curl -X POST http://localhost:8000/v1/chat/completions \
  -d '{"messages": [{"role": "user", "content": "Test"}], "model": "qwen3-32b"}'

# OLMo3-7B-Instruct
curl -X POST http://localhost:8000/v1/chat/completions \
  -d '{"messages": [{"role": "user", "content": "Test"}], "model": "olmo3-7b-instruct"}'
```

### 4. Test Error Handling

Test graceful error handling:

```bash
# Invalid model
curl -X POST http://localhost:8000/v1/chat/completions \
  -d '{"messages": [{"role": "user", "content": "Test"}], "model": "nonexistent-model"}'

# Empty messages
curl -X POST http://localhost:8000/v1/chat/completions \
  -d '{"messages": []}'

# Invalid parameters
curl -X POST http://localhost:8000/v1/chat/completions \
  -d '{"messages": [{"role": "user", "content": "Test"}], "temperature": 5.0}'
```

## Monitoring and Debugging

### Check Logs

The handler provides comprehensive logging:

```bash
# Start server with debug logging
LOG_LEVEL=DEBUG python -m backend.api.main
```

**Key log messages to watch for:**
- `"Loading vLLM model: ..."` - Model loading started
- `"✓ vLLM model loaded successfully"` - Model ready
- `"Generated X tokens in Yms"` - Generation completed
- `"Generation attempt N/3 failed"` - Retry in progress

### View Metrics

Check system metrics:

```bash
curl http://localhost:8000/v1/metrics
```

**Response:**
```json
{
  "total_queries": 42,
  "cache_hit_rate": 0.35,
  "avg_latency_ms": 250.5,
  "models_used": ["qwen3-8b", "qwen3-14b"]
}
```

### Check Handler Statistics

Access handler stats programmatically:

```python
from backend.inference.vllm_handler import create_vllm_handler_from_config
import yaml

# Load config
with open("backend/config/model_config.yaml") as f:
    config = yaml.safe_load(f)["models"]["qwen3_8b"]

# Create handler
handler = create_vllm_handler_from_config("qwen3-8b", config)
handler.load()

# Get stats
stats = handler.get_stats()
print(f"Total requests: {stats['total_requests']}")
print(f"Avg latency: {stats['avg_latency_ms']:.2f}ms")
print(f"Avg tokens/request: {stats['avg_tokens_per_request']:.1f}")
```

## Performance Benchmarking

### Latency Test

Measure average latency over multiple requests:

```bash
# Install httpie for easier testing
pip install httpie

# Run 10 requests and measure time
time for i in {1..10}; do
  http POST localhost:8000/v1/chat/completions \
    messages:='[{"role":"user","content":"Hello"}]' \
    model=qwen3-8b
done
```

### Throughput Test

Test concurrent requests:

```python
import asyncio
import aiohttp
import time

async def send_request(session, i):
    url = "http://localhost:8000/v1/chat/completions"
    data = {
        "messages": [{"role": "user", "content": f"Request {i}"}],
        "model": "qwen3-8b"
    }
    async with session.post(url, json=data) as resp:
        return await resp.json()

async def benchmark(num_requests=10):
    start = time.time()
    async with aiohttp.ClientSession() as session:
        tasks = [send_request(session, i) for i in range(num_requests)]
        results = await asyncio.gather(*tasks)
    elapsed = time.time() - start

    print(f"Completed {num_requests} requests in {elapsed:.2f}s")
    print(f"Throughput: {num_requests/elapsed:.2f} req/s")

asyncio.run(benchmark(10))
```

## Troubleshooting

### Issue: "Model config not found"
**Solution:** Ensure model key matches config (e.g., `qwen3-8b` not `qwen3_8b`)

### Issue: "Model does not use vLLM framework"
**Solution:** Check `model_config.yaml` - only models with `framework: vllm` are supported

### Issue: "vLLM not installed"
**Solution:** Handler falls back to mock engine. Install vLLM for production use:
```bash
pip install vllm
```

### Issue: CUDA out of memory
**Solution:**
1. Reduce `gpu_memory_utilization` in config (default: 0.90)
2. Use smaller model (8B instead of 32B)
3. Unload other models before loading new one

### Issue: Slow inference
**Possible causes:**
1. CPU inference (no GPU) - expected to be slow
2. Cold start - first request loads model
3. Large context - try reducing `max_tokens`

## Next Steps

After verifying the vLLM handler works:

1. **Benchmark Performance:**
   - Compare Qwen3 vs OLMo3 latency
   - Measure cache hit rate improvement
   - Profile GPU memory usage

2. **Test with Real Workloads:**
   - Multi-turn conversations
   - Long-context queries
   - Concurrent users

3. **Continue Development:**
   - Implement llama.cpp handler (Priority 1.2)
   - Complete vision handler (Priority 1.3)
   - Add streaming support to API

## Summary

✅ **All success criteria met:**
- vLLM handler fully implemented with 673 lines of code
- Supports Qwen3 and OLMo3 model families
- All 31 tests passing
- Integrated with API routes and caching
- Comprehensive error handling and retry logic
- GPU memory management implemented
- Production-ready with mock fallback for testing

**Branch:** `claude/implement-vllm-handler-01Rpbz4kMZktYrBQ1ykNtHdr`
**Commit:** Complete vLLM handler implementation
**Status:** ✅ Ready for testing and integration
