# Quick Start Guide for Future Claude Code Sessions

This guide provides a quick overview for future Claude Code sessions continuing development on the Local Multi-Model AI Orchestration System.

## 📋 Current Status (Last Updated: 2025-11-25)

**Overall Progress:** ~55% Complete (Core routing, caching, and vLLM inference operational)

**Latest Branch:** `claude/implement-vllm-handler-01Rpbz4kMZktYrBQ1ykNtHdr`

### ✅ Completed (Session 1)
- **vLLM Handler:** Full GPU-based inference with Qwen3 & OLMo3 support
- **API Integration:** Routes wired to vLLM handler with caching
- **Testing:** 31 comprehensive tests, all passing
- **Documentation:** Testing instructions and updated roadmap

### 🔜 Next Priority (Session 2)
**llama.cpp Handler** for CPU/edge device inference
- File: `backend/inference/llamacpp_handler.py`
- Estimated Time: 2-3 hours
- See Session 2 details in DEVELOPMENT_ROADMAP.md

## 🚀 How to Start Next Session

### 1. Read Key Documents First
**Essential Reading (5 minutes):**
```bash
# Primary reference - read this first!
cat DEVELOPMENT_ROADMAP.md

# Understand vLLM handler pattern (Session 1 reference)
cat backend/inference/vllm_handler.py | head -100

# Test patterns to follow
cat tests/test_vllm.py | head -50

# Testing procedures
cat TESTING_INSTRUCTIONS.md | head -100
```

### 2. Verify Current State
```bash
# Check git status
git status

# Run existing tests to verify setup
python -m pytest tests/test_vllm.py -v

# Check if API server starts
python -m backend.api.main --help
```

### 3. Start Implementation
Follow the detailed instructions in `DEVELOPMENT_ROADMAP.md` for the next session:
- **Session 2:** llama.cpp Handler (Lines 311-360)
- **Session 3:** Vision Handler (Lines 362-421)
- **Session 4:** Testing & QA (Lines 423-487)
- **Session 5:** Fine-Tuning Pipeline (Lines 489-547)

## 📁 Key Files Reference

### Configuration
- `backend/config/model_config.yaml` - Model definitions (Qwen3 + OLMo3)
- `backend/config/routing_rules.yaml` - Routing thresholds

### Core Implementation (Complete ✅)
- `backend/core/router.py` - Query routing
- `backend/core/complexity_classifier.py` - Complexity scoring
- `backend/core/cache_manager.py` - Semantic caching
- `backend/inference/vllm_handler.py` - GPU inference (✅ Complete)
- `backend/api/routes.py` - API endpoints (✅ vLLM integrated)

### Need Implementation (Priority Order)
1. `backend/inference/llamacpp_handler.py` - CPU inference (Session 2)
2. `backend/inference/vision_handler.py` - Vision models (Session 3)
3. `tests/integration/test_end_to_end.py` - Integration tests (Session 4)
4. `backend/finetuning/trainer.py` - Fine-tuning (Session 5)

### Documentation
- `DEVELOPMENT_ROADMAP.md` - Complete development plan ⭐ READ THIS
- `TESTING_INSTRUCTIONS.md` - Testing procedures
- `README.md` - Project overview
- `SystemSpec.md` - Technical specifications

## 🎯 Implementation Patterns

### Handler Pattern (Established in Session 1)
All inference handlers follow this pattern:

```python
class HandlerName:
    def __init__(self, config):
        # Initialize with configuration
        self._engine = None
        self._is_loaded = False

    def load(self):
        # Load model with retry logic
        # 3 attempts with exponential backoff
        pass

    async def generate(self, request):
        # Single response generation
        # Return InferenceResponse with content, usage, latency
        pass

    async def generate_stream(self, request):
        # Streaming generation
        # Yield chunks as async iterator
        pass

    def get_stats(self):
        # Return handler statistics
        pass

    def unload(self):
        # Clean up resources
        pass
```

### Test Pattern (Follow from test_vllm.py)
```python
class TestHandlerName:
    """Test class for HandlerName."""

    # Fixtures for config and handler
    @pytest.fixture
    def handler_config():
        return Config(...)

    # Test categories:
    # - Model loading/unloading
    # - Generation (basic, edge cases)
    # - Streaming
    # - Error handling
    # - Integration

    @pytest.mark.asyncio
    async def test_feature(self, handler):
        # Test implementation
        pass
```

### API Integration Pattern
```python
# 1. Add handler getter function
def _get_or_create_handler(model_key: str) -> Handler:
    if model_key in _handlers:
        return _handlers[model_key]
    # Create and cache handler
    handler = create_handler_from_config(model_key, config)
    handler.load()
    _handlers[model_key] = handler
    return handler

# 2. Extend _generate_response()
async def _generate_response(request, route_decision):
    model_key = route_decision.model

    # Check model framework and route accordingly
    if framework == "vllm":
        handler = _get_or_create_vllm_handler(model_key)
    elif framework == "llamacpp":
        handler = _get_or_create_llamacpp_handler(model_key)

    # Generate and return
    response = await handler.generate(...)
    return (response.content, response.model, ...)
```

## 🐛 Common Issues & Solutions

### Issue: Import errors when running tests
**Solution:**
```bash
# Ensure you're in the project root
cd /home/user/GroupQwen

# Install missing dependencies
pip install pytest pytest-asyncio pyyaml
```

### Issue: "Model config not found"
**Solution:**
- Check model key format: use `qwen3-8b` not `qwen3_8b`
- Verify model exists in `backend/config/model_config.yaml`

### Issue: "vLLM not installed" in production
**Solution:**
- Handler falls back to mock engine for development
- For production: `pip install vllm`
- Mock is intentional for testing without GPU

### Issue: Tests fail due to missing async
**Solution:**
- Use `@pytest.mark.asyncio` decorator for async tests
- Use `async def` for test functions that await

## 📊 Testing Guidelines

### Run Tests Before Starting
```bash
# Verify existing tests pass
pytest tests/ -v

# Run specific test file
pytest tests/test_vllm.py -v

# Run with coverage
pytest tests/ --cov=backend --cov-report=html
```

### Write Tests During Implementation
- Minimum 20-25 tests per handler
- Follow test structure from `tests/test_vllm.py`
- Include mock engines for testing without dependencies

### Test Coverage Targets
- **Unit Tests:** >80% coverage per module
- **Integration Tests:** All API endpoints
- **Performance:** Latency, throughput benchmarks

## 🔄 Git Workflow

### Branch Naming
```bash
# Pattern: claude/<task-name>-<session-id>
# Example: claude/implement-llamacpp-handler-<session-id>
```

### Commit Messages
```bash
# Format: Clear, descriptive, with details
git commit -m "$(cat <<'EOF'
Implement llama.cpp handler for CPU inference

Changes:
1. backend/inference/llamacpp_handler.py - Complete implementation
   - GGUF model loading
   - Metal acceleration for macOS
   - CPU fallback

2. backend/api/routes.py - Extend for llamacpp
   - Add handler routing

3. tests/test_llamacpp.py - Comprehensive tests
   - 25 tests covering all features

Success Criteria Met:
✓ CPU inference works
✓ All tests pass
EOF
)"
```

### Push to Remote
```bash
# Always use -u flag for new branches
git push -u origin <branch-name>

# Session ID must match for push to succeed
```

## 💡 Best Practices

### Code Style
- Follow patterns from Session 1 (`vllm_handler.py`)
- Use type hints: `def func(param: Type) -> ReturnType`
- Add docstrings for all public methods
- Keep functions focused and single-purpose

### Error Handling
- Use retry logic with exponential backoff (3 attempts)
- Log errors with context: `logger.error(f"Context: {e}", exc_info=True)`
- Raise appropriate exceptions (ValueError, RuntimeError, HTTPException)
- Provide fallback behavior when possible

### Performance
- Cache loaded models (don't reload on every request)
- Use async/await for I/O operations
- Clean up resources in unload() methods
- Track metrics (latency, throughput, memory)

### Documentation
- Update DEVELOPMENT_ROADMAP.md after completing tasks
- Mark tasks as complete with ✅
- Add testing instructions for new features
- Update this file if patterns change

## 📞 Getting Help

**Before Starting:**
1. Read `DEVELOPMENT_ROADMAP.md` (most important!)
2. Review Session 1 implementation (`vllm_handler.py`)
3. Check `TESTING_INSTRUCTIONS.md` for testing examples

**During Development:**
1. Reference SystemSpec.md for technical details
2. Follow test patterns from `tests/test_vllm.py`
3. Check existing handlers for examples

**Key Insight:** Session 1 established all the patterns. Follow them!

## 🎓 Session 1 Learnings

**What Worked Well:**
- Handler pattern with load/generate/stream/unload
- Factory functions for creating from YAML config
- Mock engines for testing without dependencies
- Comprehensive test coverage (31 tests)
- Clear documentation with examples

**Apply These Patterns to Future Sessions:**
- Use same class structure for all handlers
- Create test suite following test_vllm.py structure
- Add mock engines for development
- Document with testing instructions
- Update roadmap immediately after completion

---

**Ready to Continue?** Start with Session 2 in `DEVELOPMENT_ROADMAP.md` lines 311-360!
