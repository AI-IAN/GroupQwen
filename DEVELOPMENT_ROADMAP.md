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

### 1.1 Implement vLLM Handler
**File:** `backend/inference/vllm_handler.py` (currently a stub)
**Status:** Needs implementation
**Estimated Time:** 2-3 hours

**Tasks:**
- [ ] Uncomment and complete vLLM integration code
- [ ] Implement model loading with proper VRAM management
- [ ] Add streaming response support
- [ ] Implement proper error handling and retries
- [ ] Test with Qwen3-8B and OLMo3-7B models
- [ ] Add tensor parallelism support for larger models

**Reference:** See SystemSpec.md lines 350-450 for vLLM configuration details

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

### 1.4 Wire Up API Routes to Real Inference
**File:** `backend/api/routes.py`
**Status:** Currently using mock responses
**Estimated Time:** 1-2 hours

**Tasks:**
- [ ] Replace `_generate_mock_response()` with actual inference handler calls
- [ ] Route based on model type (vllm vs llamacpp)
- [ ] Add comprehensive error handling
- [ ] Test end-to-end flow: cache → routing → inference → response
- [ ] Add request/response logging

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

### Session 1: Get Basic Inference Working (3-4 hours)
1. Implement vLLM handler (`backend/inference/vllm_handler.py`)
2. Wire up to API routes (`backend/api/routes.py`)
3. Test with Qwen3-8B or OLMo3-7B
4. Verify end-to-end flow works

**Success Criteria:** Can send a chat completion request and get a real model response

### Session 2: Add Edge Device Support (2-3 hours)
1. Complete llama.cpp handler (`backend/inference/llamacpp_handler.py`)
2. Test with quantized models
3. Add device-specific routing (server vs macOS)

**Success Criteria:** Can run inference on both GPU (vLLM) and CPU (llama.cpp)

### Session 3: Complete Specialized Handlers (3-4 hours)
1. Implement vision handler
2. Test vision API endpoints
3. Add comprehensive error handling

**Success Criteria:** Vision analysis works with images

### Session 4: Testing & Quality (3-4 hours)
1. Create comprehensive test suite
2. Run benchmarks
3. Compare Qwen3 vs OLMo3 performance
4. Document findings

**Success Criteria:** >80% test coverage, documented performance metrics

### Session 5+: Fine-Tuning & Polish
1. Implement fine-tuning pipeline
2. Add frontend (or integrate OpenWebUI)
3. Production deployment

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
