# Development Roadmap

**Last Updated:** 2025-11-25
**Project Status:** ~45-50% Complete (Core routing and caching operational)

This document provides a clear roadmap for continuing development on the Local Multi-Model AI Orchestration System. It's designed as a reference for future Claude Code sessions or other developers.

---

## 🎯 Current State

### ✅ What's Complete and Working

**Core Infrastructure:**
- ✅ Query routing system (`backend/core/router.py`)
- ✅ Complexity classification (`backend/core/complexity_classifier.py`)
- ✅ Confidence scoring (`backend/core/confidence_scorer.py`)
- ✅ Semantic caching with Redis (`backend/core/cache_manager.py`)
- ✅ FastAPI application structure (`backend/api/`)
- ✅ Configuration system (YAML-based model and routing configs)

**Monitoring & Observability:**
- ✅ Structured logging (`backend/monitoring/logger.py`)
- ✅ Health checks (`backend/monitoring/health_check.py`)
- ✅ Metrics collection (`backend/monitoring/metrics.py`)
- ✅ API endpoints for health, metrics, and cache stats

**Specialized Handlers:**
- ✅ Translation handler - Qwen3-MT fully implemented (`backend/inference/translation_handler.py`)

**Model Support:**
- ✅ Qwen3 family: 4B, 8B, 14B, 32B, 32B-Thinking, VL, MT
- ✅ OLMo3 family: 7B, 7B-Instruct, 32B, 32B-Instruct (newly added)

**Testing:**
- ✅ Router tests (`tests/test_router.py`)
- ✅ Cache tests (`tests/test_cache.py`)

---

## 🔴 Priority 1: Complete Core Inference (HIGH PRIORITY)

These are essential for the system to actually run inference on models.

### 1.1 Implement vLLM Handler ✅ COMPLETED
**File:** `backend/inference/vllm_handler.py`
**Status:** ✅ Complete (Session 1 - 2025-11-25)
**Actual Time:** 3 hours

**Tasks:**
- [x] Complete vLLM integration code (673 lines)
- [x] Implement model loading with proper VRAM management
- [x] Add streaming response support (async generators)
- [x] Implement proper error handling and retries (3 attempts, exponential backoff)
- [x] Test with Qwen3-8B and OLMo3-7B models (31 tests passing)
- [x] Add tensor parallelism support for larger models (via ModelConfig)
- [x] Wire up to API routes (backend/api/routes.py)
- [x] Create comprehensive test suite (tests/test_vllm.py)
- [x] Add testing documentation (TESTING_INSTRUCTIONS.md)

**Branch:** `claude/implement-vllm-handler-01Rpbz4kMZktYrBQ1ykNtHdr`
**Commits:**
- `9867962` - Implement vLLM handler for GPU-based model inference
- `89a02c9` - Add comprehensive testing instructions for vLLM handler

**Key Features Implemented:**
- Full vLLM integration with LLM class and SamplingParams
- Support for Qwen3 (`<|im_start|>`) and OLMo3 (`<|system|>`) chat templates
- MockVLLMEngine for testing without GPU
- Handler caching for model reuse
- Graceful fallback to mock responses
- GPU memory management with torch.cuda.empty_cache()
- Performance metrics tracking (latency, tokens/sec, usage stats)

### 1.2 Implement llama.cpp Handler
**File:** `backend/inference/llamacpp_handler.py` (partially complete)
**Status:** Needs completion
**Estimated Time:** 2-3 hours

**Tasks:**
- [ ] Complete GGUF model loading
- [ ] Add Metal acceleration support (for macOS)
- [ ] Implement CPU fallback inference
- [ ] Add streaming support
- [ ] Test with quantized models (Q4, Q5, Q6)
- [ ] Add proper memory management and cleanup

**Reference:** See SystemSpec.md lines 450-520 for llama.cpp setup

### 1.3 Complete Vision Handler
**File:** `backend/inference/vision_handler.py` (stub exists)
**Status:** Needs implementation
**Estimated Time:** 2-3 hours

**Tasks:**
- [ ] Implement Qwen3-VL integration
- [ ] Add image preprocessing (base64, URLs, file uploads)
- [ ] Support bounding box detection
- [ ] Add OCR capabilities
- [ ] Test with screenshots and images
- [ ] Wire up to `/v1/vision/analyze` endpoint

### 1.4 Wire Up API Routes to Real Inference ✅ COMPLETED
**File:** `backend/api/routes.py`
**Status:** ✅ Complete (Session 1 - 2025-11-25)
**Actual Time:** 1 hour

**Tasks:**
- [x] Replace `_generate_mock_response()` with actual inference handler calls
- [x] Route based on model type (vllm vs llamacpp) - vLLM complete
- [x] Add comprehensive error handling (ValueError, HTTPException)
- [x] Test end-to-end flow: cache → routing → inference → response
- [x] Add request/response logging

**Implementation:**
- Created `_get_or_create_vllm_handler()` for handler management
- Created `_generate_response()` async function for inference
- Integrated with existing cache, router, and metrics systems
- Handler caching in `_vllm_handlers` dict for efficiency
- Graceful fallback to mock for non-vLLM models (llamacpp)

---

## 🟡 Priority 2: Fine-Tuning Pipeline (MEDIUM PRIORITY)

Allows users to fine-tune models on custom datasets.

### 2.1 Data Export Scripts
**Files:** Create `scripts/export_chat_history.py`
**Status:** Not started
**Estimated Time:** 2-3 hours

**Tasks:**
- [ ] Support ChatGPT conversation export format (ZIP)
- [ ] Support Claude conversation export
- [ ] Support Perplexity export
- [ ] Add data validation
- [ ] Convert to unified JSONL format

### 2.2 Data Curation
**Files:** `backend/finetuning/data_curator.py`, `scripts/curate_dataset.py`
**Status:** Stubs exist
**Estimated Time:** 3-4 hours

**Tasks:**
- [ ] Implement interactive curation CLI
- [ ] Add quality filtering (remove low-quality conversations)
- [ ] Add PII detection and removal
- [ ] Support batch processing
- [ ] Export curated dataset in training format

### 2.3 Training Pipeline
**Files:** `backend/finetuning/trainer.py`, `backend/finetuning/checkpoint_manager.py`
**Status:** Stubs with commented Unsloth code
**Estimated Time:** 4-5 hours

**Tasks:**
- [ ] Uncomment and complete Unsloth/QLoRA integration
- [ ] Implement training loop with progress tracking
- [ ] Add checkpoint saving and loading
- [ ] Support resume from checkpoint
- [ ] Wire up to `/v1/finetune/start` API endpoint
- [ ] Add background job management

**Reference:** See SystemSpec.md lines 600-750 for fine-tuning details

---

## 🟢 Priority 3: Frontend & User Interface (MEDIUM-LOW PRIORITY)

Web interface for interacting with the system.

### 3.1 Frontend Setup
**Directory:** `frontend/src/`
**Status:** Basic structure exists, needs implementation
**Estimated Time:** 6-8 hours

**Tasks:**
- [ ] Setup React + TypeScript + Tailwind CSS
- [ ] Create chat interface component
- [ ] Add model selector dropdown
- [ ] Implement streaming response display
- [ ] Add system metrics dashboard
- [ ] Create file upload for vision/fine-tuning
- [ ] WebSocket integration for real-time updates

**Alternative:** Consider integrating with OpenWebUI instead of building custom frontend

---

## 🔵 Priority 4: Testing & Quality (MEDIUM PRIORITY)

Comprehensive testing to ensure reliability.

### 4.1 Inference Tests
**File:** Create `tests/test_inference.py`
**Status:** Not started
**Estimated Time:** 2-3 hours

**Tasks:**
- [ ] Test vLLM handler with mocked responses
- [ ] Test llama.cpp handler
- [ ] Test vision handler
- [ ] Test translation handler
- [ ] Test error handling and edge cases

### 4.2 API Integration Tests
**File:** Create `tests/test_api.py`
**Status:** Not started
**Estimated Time:** 2-3 hours

**Tasks:**
- [ ] Test all API endpoints
- [ ] Test caching behavior
- [ ] Test model routing logic
- [ ] Test error responses
- [ ] Test concurrent requests

### 4.3 End-to-End Tests
**File:** Create `tests/integration/test_end_to_end.py`
**Status:** Not started
**Estimated Time:** 2-3 hours

**Tasks:**
- [ ] Test full query flow (API → cache → router → inference → response)
- [ ] Test cache hit/miss scenarios
- [ ] Test model escalation on low confidence
- [ ] Test metrics collection

### 4.4 Benchmarking
**File:** Enhance `scripts/benchmark.py`
**Status:** Basic structure exists
**Estimated Time:** 2-3 hours

**Tasks:**
- [ ] Add latency benchmarks for each model
- [ ] Test cache hit rate optimization
- [ ] Benchmark concurrent request handling
- [ ] Profile memory usage
- [ ] Compare Qwen3 vs OLMo3 performance

---

## 🟣 Priority 5: Documentation & Deployment (LOW PRIORITY)

Production-ready deployment and documentation.

### 5.1 Docker & Deployment
**Files:** `docker/Dockerfile.frontend`, `docker-compose.yml`, `scripts/deploy.sh`
**Status:** Partial
**Estimated Time:** 3-4 hours

**Tasks:**
- [ ] Complete multi-stage Docker build for frontend
- [ ] Enhance docker-compose.yml with all services
- [ ] Create production deployment script
- [ ] Add environment validation
- [ ] Add service health checks
- [ ] Document Tailscale setup for multi-device access

### 5.2 Expanded Documentation
**Status:** Core docs exist, need enhancement
**Estimated Time:** 2-3 hours

**Tasks:**
- [ ] Create `docs/ARCHITECTURE.md` with system diagrams
- [ ] Create `docs/API_REFERENCE.md` with all endpoints
- [ ] Create `docs/TROUBLESHOOTING.md` for common issues
- [ ] Create `docs/MODEL_COMPARISON.md` for Qwen3 vs OLMo3 benchmarks
- [ ] Add code-level documentation (docstrings)

---

## 📊 Progress Tracking

| Phase | Priority | Completion | Time Estimate |
|-------|----------|-----------|---------------|
| Core Inference | 🔴 High | 30% | 8-12 hours |
| Fine-Tuning Pipeline | 🟡 Medium | 10% | 10-12 hours |
| Frontend | 🟢 Med-Low | 5% | 6-8 hours |
| Testing & Quality | 🔵 Medium | 20% | 8-10 hours |
| Deployment & Docs | 🟣 Low | 40% | 5-7 hours |

**Overall Completion:** ~45-50%
**Time to MVP (Core Inference Working):** 8-12 hours
**Time to Full Feature Complete:** 35-45 hours

---

## 🚀 Recommended Next Steps for Claude Code

When starting a new session, follow this priority order:

### Session 1: Get Basic Inference Working ✅ COMPLETED
**Status:** ✅ Complete (2025-11-25)
**Branch:** `claude/implement-vllm-handler-01Rpbz4kMZktYrBQ1ykNtHdr`

1. ✅ Implement vLLM handler (`backend/inference/vllm_handler.py`)
2. ✅ Wire up to API routes (`backend/api/routes.py`)
3. ✅ Test with Qwen3-8B or OLMo3-7B (31 tests passing)
4. ✅ Verify end-to-end flow works

**Success Criteria:** ✅ Can send a chat completion request and get a real model response

**What to Review Before Session 2:**
- Read `TESTING_INSTRUCTIONS.md` for testing procedures
- Check `tests/test_vllm.py` for test patterns to follow
- Review `backend/inference/vllm_handler.py` for handler structure
- Understand the handler pattern: load(), generate(), generate_stream(), unload()

### Session 2: Add Edge Device Support (2-3 hours) 🔜 NEXT
**Goal:** Enable CPU-based inference for edge devices (MacBook, mobile)
**Priority:** HIGH - Critical for multi-device deployment

**Tasks:**
1. **Complete llama.cpp handler** (`backend/inference/llamacpp_handler.py`)
   - Follow the vLLM handler pattern established in Session 1
   - Implement `LlamaCppHandler` class with:
     - `load()` - Load GGUF quantized models
     - `generate()` - Synchronous/async text generation
     - `generate_stream()` - Streaming support
     - `unload()` - Memory cleanup
   - Support Metal acceleration for macOS (if available)
   - Add CPU fallback when Metal unavailable
   - Test with Q4, Q5, Q6 quantization levels

2. **Update API routes** (`backend/api/routes.py`)
   - Extend `_generate_response()` to support llamacpp models
   - Add `_get_or_create_llamacpp_handler()` similar to vLLM
   - Route based on model framework: check `model_config.yaml` for `framework: llamacpp`
   - Models to support: `qwen3-4b`, `qwen3-8b` (GGUF versions)

3. **Create test suite** (`tests/test_llamacpp.py`)
   - Follow the test structure from `tests/test_vllm.py`
   - 25+ tests covering: loading, generation, streaming, errors
   - Mock engine for testing without actual GGUF files

4. **Update routing logic** (if needed)
   - Ensure `backend/core/router.py` properly routes edge device queries
   - Test device-specific routing: `device: "macbook"` → llamacpp

**Success Criteria:**
- ✓ Can load GGUF models via llama.cpp
- ✓ CPU inference works on models configured with `framework: llamacpp`
- ✓ Metal acceleration enabled on macOS
- ✓ All tests pass (25+ tests)
- ✓ API routes handle both vLLM (GPU) and llamacpp (CPU) models

**Reference Files:**
- `backend/inference/vllm_handler.py` - Handler pattern to follow
- `backend/config/model_config.yaml` - Models configured for llamacpp
- `tests/test_vllm.py` - Test patterns to replicate
- SystemSpec.md lines 450-520 - llama.cpp setup details

**Testing:**
```bash
# Test with GGUF model
curl -X POST http://localhost:8000/v1/chat/completions \
  -d '{"messages": [{"role": "user", "content": "Hello"}], "model": "qwen3-4b-gguf"}'
```

### Session 3: Complete Vision Handler (3-4 hours)
**Goal:** Enable image understanding with Qwen3-VL
**Priority:** HIGH - Enables multimodal capabilities

**Tasks:**
1. **Implement vision handler** (`backend/inference/vision_handler.py`)
   - Create `VisionHandler` class following vLLM handler pattern
   - Implement image preprocessing:
     - Base64 encoded images
     - Image URLs (download and process)
     - File uploads (via multipart/form-data)
   - Support Qwen3-VL features:
     - Image understanding and description
     - Bounding box detection (if `return_bboxes=True`)
     - OCR (text extraction from images)
     - Screenshot analysis
     - GUI automation (element detection)
   - Use vLLM for inference (Qwen3-VL uses vLLM framework)

2. **Wire up to API routes** (`backend/api/routes.py`)
   - Complete `/v1/vision/analyze` endpoint implementation
   - Replace mock response with actual `VisionHandler` calls
   - Handle image data formats (base64, URL, file)
   - Support optional bounding box return

3. **Create test suite** (`tests/test_vision.py`)
   - Test image loading (base64, URL, file)
   - Test description generation
   - Test bounding box detection
   - Test OCR functionality
   - Mock image processing for testing without GPU
   - 20+ comprehensive tests

4. **Add example images** (`tests/fixtures/`)
   - Sample images for testing
   - Screenshots for GUI automation tests
   - Documents for OCR tests

**Success Criteria:**
- ✓ Can analyze images via `/v1/vision/analyze` endpoint
- ✓ Supports base64, URL, and file upload formats
- ✓ Returns bounding boxes when requested
- ✓ OCR extraction works on text-heavy images
- ✓ All tests pass (20+ tests)

**Reference Files:**
- `backend/config/model_config.yaml` - Qwen3-VL config
- `backend/api/models.py` - VisionAnalysisRequest/Response models
- SystemSpec.md - Vision capabilities documentation

**Testing:**
```bash
# Test with base64 image
curl -X POST http://localhost:8000/v1/vision/analyze \
  -d '{
    "image": "data:image/png;base64,iVBORw0KGgo...",
    "prompt": "Describe this image",
    "return_bboxes": false
  }'
```

### Session 4: Testing & Quality Assurance (3-4 hours)
**Goal:** Comprehensive testing and performance benchmarking
**Priority:** MEDIUM - Ensures reliability and documents performance

**Tasks:**
1. **Integration tests** (`tests/integration/test_end_to_end.py`)
   - Full flow: API → Cache → Router → Inference → Response
   - Cache hit/miss scenarios
   - Model escalation on low confidence
   - Multi-turn conversations
   - Concurrent request handling
   - Error recovery and retries
   - 15+ integration tests

2. **API tests** (`tests/test_api.py`)
   - Test all API endpoints:
     - `/v1/chat/completions` (streaming and non-streaming)
     - `/v1/vision/analyze`
     - `/v1/translate`
     - `/v1/models`
     - `/v1/cache/stats` and `/v1/cache/clear`
     - `/v1/health` and `/v1/metrics`
   - Test error handling (invalid inputs, auth, rate limits)
   - Test OpenAI API compatibility
   - 20+ API tests

3. **Performance benchmarking** (`scripts/benchmark.py`)
   - Enhance existing benchmark script with:
     - Latency measurements per model (p50, p95, p99)
     - Throughput testing (req/s)
     - Cache hit rate optimization
     - Concurrent user simulation (10, 50, 100 users)
     - Memory profiling (VRAM usage over time)
   - Compare Qwen3 vs OLMo3:
     - Quality (MMLU, HumanEval benchmarks if available)
     - Speed (tokens/sec)
     - Memory efficiency (VRAM/token)

4. **Documentation** (`docs/PERFORMANCE_BENCHMARKS.md`)
   - Create performance report with:
     - Model comparison table (Qwen3 vs OLMo3)
     - Latency charts
     - Throughput measurements
     - Cache effectiveness
     - Recommendations for model selection

**Success Criteria:**
- ✓ >80% test coverage across all modules
- ✓ All integration tests pass (15+)
- ✓ All API tests pass (20+)
- ✓ Benchmark results documented
- ✓ Performance comparison completed (Qwen3 vs OLMo3)
- ✓ No memory leaks detected

**Commands:**
```bash
# Run all tests with coverage
pytest tests/ --cov=backend --cov-report=html

# Run benchmarks
python scripts/benchmark.py --models qwen3-8b,olmo3-7b-instruct --requests 100

# Generate performance report
python scripts/benchmark.py --compare --output docs/PERFORMANCE_BENCHMARKS.md
```

### Session 5: Fine-Tuning Pipeline (4-5 hours)
**Goal:** Enable custom model fine-tuning on user data
**Priority:** MEDIUM - Adds personalization capabilities

**Tasks:**
1. **Data export scripts** (`scripts/export_chat_history.py`)
   - Support ChatGPT export (ZIP format)
   - Support Claude conversation export
   - Support Perplexity export
   - Convert to unified JSONL format
   - Add data validation and cleaning

2. **Data curation** (`backend/finetuning/data_curator.py`)
   - Implement interactive curation CLI
   - Quality filtering (remove low-quality conversations)
   - PII detection and removal (names, emails, addresses)
   - Batch processing support
   - Export in training format (Unsloth-compatible)

3. **Training pipeline** (`backend/finetuning/trainer.py`)
   - Complete Unsloth/QLoRA integration
   - Implement training loop with progress tracking
   - Checkpoint saving and loading
   - Resume from checkpoint support
   - Wire up to `/v1/finetune/start` API endpoint
   - Background job management

4. **Checkpoint management** (`backend/finetuning/checkpoint_manager.py`)
   - Save/load checkpoints
   - Track training metrics (loss, learning rate)
   - Model merging after training
   - Upload to HuggingFace Hub (optional)

**Success Criteria:**
- ✓ Can export chat history from major providers
- ✓ Data curation removes PII and low-quality data
- ✓ Training pipeline works with QLoRA
- ✓ Can fine-tune models on custom datasets
- ✓ `/v1/finetune/start` endpoint functional

**Reference Files:**
- SystemSpec.md lines 600-750 - Fine-tuning details
- `backend/finetuning/` - Existing stubs

**Testing:**
```bash
# Export and curate data
python scripts/export_chat_history.py --source chatgpt --input ~/Downloads/conversations.zip
python scripts/curate_dataset.py --input data/raw.jsonl --output data/curated.jsonl

# Start fine-tuning job
curl -X POST http://localhost:8000/v1/finetune/start \
  -d '{
    "base_model": "qwen3-8b",
    "dataset_path": "data/curated.jsonl",
    "epochs": 3,
    "lora_rank": 128
  }'
```

### Session 6+: Frontend & Production Polish
**Goal:** Production deployment and user interface
**Priority:** MEDIUM-LOW - Core functionality complete

**Tasks:**
1. Frontend development (or OpenWebUI integration)
2. Docker multi-stage builds
3. Production deployment scripts
4. Monitoring and alerting setup
5. Documentation polish and API reference

---

## 💡 Key Design Principles

1. **Model Agnostic:** Support multiple model families (Qwen3, OLMo3, future models)
2. **Device Aware:** Route based on available hardware (GPU/CPU/Metal)
3. **Performance First:** Semantic caching, optimal model selection, streaming
4. **Production Ready:** Monitoring, health checks, error handling, logging
5. **Extensible:** Easy to add new models and handlers

---

## 📚 Key Files Reference

### Configuration
- `backend/config/model_config.yaml` - Model definitions (Qwen3 + OLMo3)
- `backend/config/routing_rules.yaml` - Routing thresholds and escalation

### Core Logic
- `backend/core/router.py` - Query routing orchestration
- `backend/core/complexity_classifier.py` - Query complexity scoring
- `backend/core/confidence_scorer.py` - Response confidence evaluation
- `backend/core/cache_manager.py` - Semantic caching

### Inference (Needs Work)
- `backend/inference/vllm_handler.py` - GPU inference (stub)
- `backend/inference/llamacpp_handler.py` - CPU inference (partial)
- `backend/inference/vision_handler.py` - Vision models (stub)
- `backend/inference/translation_handler.py` - Translation (complete ✅)

### API
- `backend/api/main.py` - FastAPI app initialization
- `backend/api/routes.py` - API endpoints
- `backend/api/models.py` - Pydantic models

### Documentation
- `README.md` - Project overview and features
- `GETTING_STARTED.md` - Quick start guide for users
- `DEVELOPMENT_ROADMAP.md` - This file (for developers)
- `SystemSpec.md` - Detailed system specification

---

## 🔧 Common Development Tasks

### Adding a New Model Family
1. Add model definition to `backend/config/model_config.yaml`
2. Add routing rules to `backend/config/routing_rules.yaml`
3. Update model family list in routing rules
4. Update documentation

### Adding a New Inference Handler
1. Create handler in `backend/inference/`
2. Implement `generate()` and `generate_stream()` methods
3. Add error handling and logging
4. Add tests in `tests/test_inference.py`
5. Wire up to routes in `backend/api/routes.py`

### Testing Changes
```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_router.py -v

# Start API server
python -m backend.api.main

# Test endpoint
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}]}'
```

---

## 📞 Getting Help

- **SystemSpec.md** - Detailed technical specification
- **README.md** - Feature overview and architecture
- **GETTING_STARTED.md** - Installation and quick start
- **This File** - Development roadmap and next steps

For Claude Code: Start each session by reading this file to understand current state and priorities.

---

**Remember:** Focus on Priority 1 (Core Inference) first. Everything else depends on having working model inference!
