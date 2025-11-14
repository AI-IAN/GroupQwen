# Quick Start: Parallel Development

## 🚀 Launch 4 Agents in Parallel (Fastest Approach)

### Phase A: Critical Path Tasks (~3 hours with 4 agents)

Open 4 separate Claude Code sessions and give each agent these instructions:

---

### **Agent 1: vLLM Inference Handler**

**Branch:** `feature/vllm-inference`

**Prompt:**
```
I'm working on implementing vLLM inference for the GroupQwen project.

Please complete the implementation in backend/inference/vllm_handler.py:

1. Uncomment all the actual vLLM code (lines 72-140)
2. Add proper error handling and retries
3. Implement streaming support for real-time responses
4. Test the implementation with Qwen3-8B model
5. Add comprehensive logging

The handler should:
- Load models using vLLM's LLM class
- Support tensor parallelism
- Handle GPU memory management
- Stream responses via AsyncIterator
- Include proper cleanup on errors

Reference the existing structure and make it production-ready.
Only modify backend/inference/vllm_handler.py - don't touch other files.
```

---

### **Agent 2: llama.cpp Inference Handler**

**Branch:** `feature/llamacpp-inference`

**Prompt:**
```
I'm working on implementing llama.cpp inference for edge devices in the GroupQwen project.

Please complete the implementation in backend/inference/llamacpp_handler.py:

1. Implement the LlamaCppHandler class
2. Add model loading with GGUF format support
3. Support Metal backend for macOS
4. Implement CPU inference for non-GPU systems
5. Add streaming support
6. Include proper memory management

The handler should:
- Load GGUF quantized models
- Support both CPU and Metal acceleration
- Implement generate() and generate_stream() methods
- Handle model unloading
- Format messages properly for inference
- Include error handling

Reference the vLLM handler structure but adapt for llama.cpp.
Only modify backend/inference/llamacpp_handler.py - don't touch other files.
```

---

### **Agent 3: Data Processing Pipeline**

**Branch:** `feature/data-pipeline`

**Prompt:**
```
I'm working on the fine-tuning data pipeline for the GroupQwen project.

Please implement:

1. Complete backend/finetuning/data_processor.py
   - Convert conversations to training format
   - Validate dataset format
   - Support multiple input formats

2. Complete backend/finetuning/data_curator.py
   - Interactive curation UI
   - Quality filtering
   - PII detection/removal

3. Create scripts/export_chat_history.py
   - Export from ChatGPT (ZIP format)
   - Export from Claude
   - Export from Perplexity

4. Create scripts/curate_dataset.py
   - CLI for dataset curation
   - Batch processing
   - Progress tracking

Follow the patterns described in SystemSpec.md (lines 614-713).
Only modify files in backend/finetuning/ and scripts/ - don't touch API files.
```

---

### **Agent 4: Comprehensive Testing**

**Branch:** `feature/test-suite`

**Prompt:**
```
I'm working on creating a comprehensive test suite for the GroupQwen project.

Please implement:

1. Create tests/test_inference.py
   - Test vLLM handler
   - Test llama.cpp handler
   - Test vision handler
   - Test translation handler
   - Mock model responses

2. Create tests/test_api.py
   - Test all API endpoints
   - Test error handling
   - Test caching behavior
   - Test metrics logging

3. Expand tests/test_router.py
   - More routing scenarios
   - Edge cases
   - Device-specific routing
   - Escalation logic

4. Create tests/integration/test_end_to_end.py
   - Full query flow
   - Cache integration
   - Metrics tracking

Use pytest and follow existing test patterns in tests/test_router.py.
Only modify files in tests/ directory - don't touch implementation files.
```

---

## 📋 After All 4 Agents Complete

### Integration Steps:

1. **Merge all branches:**
   ```bash
   git checkout main
   git merge feature/vllm-inference
   git merge feature/llamacpp-inference
   git merge feature/data-pipeline
   git merge feature/test-suite
   ```

2. **Wire up inference to routes.py:**
   - Replace `_generate_mock_response()` calls with actual inference
   - Route based on model type (vllm vs llamacpp)
   - Add error handling

3. **Test end-to-end:**
   ```bash
   # Start Redis
   redis-server

   # Run tests
   pytest tests/

   # Start API
   python -m backend.api.main

   # Test chat completion
   curl -X POST http://localhost:8000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"messages": [{"role": "user", "content": "Hello!"}]}'
   ```

---

## 🎯 Expected Timeline

- **Hour 0:** Start all 4 agents in parallel
- **Hour 2-3:** Agents complete their tasks
- **Hour 3-4:** Integration and testing
- **Hour 4:** Full system operational with real inference!

---

## 💡 Tips

1. Each agent works on completely separate files - no conflicts!
2. Agents can work simultaneously - no dependencies between them
3. If one agent finishes early, assign them Phase B tasks
4. Keep branches small and focused for easy merging
5. Test each component individually before integration

---

## Alternative: Sequential Approach

If you prefer one agent at a time:

1. Start with Agent 1 (vLLM) - most critical
2. Then Agent 2 (llama.cpp) - second most critical
3. Then Agent 3 (Data Pipeline)
4. Then Agent 4 (Testing)

This takes ~12 hours vs ~4 hours with parallelization.

---

## Need Help?

- See `PARALLEL_WORK_PLAN.md` for detailed strategy
- See `TASKS.md` for complete task list
- See `SystemSpec.md` for system architecture
