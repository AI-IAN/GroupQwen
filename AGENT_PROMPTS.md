# Claude Code Agent Prompts for Parallel Development

This document contains **exact prompts** for each Claude Code agent to work on GroupQwen in parallel.

**Important:** Each prompt is self-contained with all necessary context, instructions, and warnings.

---

## How to Use These Prompts

1. **Create a branch** for each agent (see branch names below)
2. **Open a separate Claude Code session** for each agent
3. **Copy-paste the entire prompt** into Claude Code
4. **Let the agent work** autonomously
5. **After completion**, merge branches sequentially

**Git Workflow:**
```bash
# Each agent starts on their branch
git checkout -b <branch-name>

# Work and commit
git add .
git commit -m "<message>"

# Push to remote (CRITICAL: branch must start with 'claude/' and end with session ID)
git push -u origin <branch-name>
```

---

## ⚠️ CRITICAL WARNINGS FOR ALL AGENTS

**DO NOT MODIFY THESE FILES (to avoid merge conflicts):**
- `backend/api/routes.py` - Only Agent 2B may modify fine-tuning endpoints (lines 200+)
- `backend/api/main.py` - Integration happens AFTER all agents finish
- `requirements.txt` - Will be merged manually after all work completes
- `frontend/package.json` - Will be merged manually after all work completes

**GIT PUSH REQUIREMENTS:**
- Always use: `git push -u origin <branch-name>`
- Branch must start with `claude/` and end with matching session ID
- If push fails with 403, check branch naming
- Retry network failures up to 4 times with exponential backoff (2s, 4s, 8s, 16s)

**TESTING BEFORE COMMIT:**
```bash
# Syntax check
python -m py_compile backend/path/to/your/file.py

# Run relevant tests (if they exist)
pytest tests/test_your_module.py -v

# Lint (optional but recommended)
black backend/path/to/your/file.py
```

---

# Phase A: Critical Path (Run These 4 Agents First in Parallel)

**Priority:** Start simultaneously, most value, zero conflicts
**Estimated Total Time:** 3-4 hours (vs 12 hours sequentially)

---

## 🔴 Agent 1A: vLLM Inference Handler

**Branch:** `feature/vllm-inference`

**Estimated Time:** 2-3 hours

### PROMPT START

```
I'm Agent 1A working on vLLM inference implementation for GroupQwen.

=== CONTEXT ===

The GroupQwen project is a local AI orchestration system that routes queries across multiple Qwen3 models (4B through 32B variants) with semantic caching and intelligent routing.

My role: Implement production-ready vLLM inference handler for GPU-accelerated models.

Current state:
- vLLM handler skeleton exists at backend/inference/vllm_handler.py
- Lines 78-84 and 105-113 have commented-out vLLM code
- API routes exist at backend/api/routes.py but use mock responses
- InferenceRequest/InferenceResponse dataclasses already defined
- This handler will replace _generate_mock_response() in routes.py (integration happens later)

=== TASKS TO COMPLETE ===

From TASKS.md Phase 1.3 - Complete Inference Handlers:

1. ✓ Uncomment actual vLLM code (lines 78-84, 105-113 in vllm_handler.py)
2. ✓ Add proper error handling and retries for model loading and inference
3. ✓ Implement streaming support in generate_stream() method (lines 131-148)
4. ✓ Add comprehensive logging with request IDs
5. ✓ Test with Qwen3-8B model (or create realistic mocks if model unavailable)
6. ✓ Add VRAM monitoring and cleanup on errors
7. ✓ Implement proper SamplingParams configuration
8. ✓ Add timeout handling (30s default for inference)
9. ✓ Include token usage tracking
10. ✓ Handle OOM (Out of Memory) errors gracefully

=== REQUIREMENTS ===

Technical Implementation:
- Use vLLM's LLM and SamplingParams classes
- Support tensor_parallel_size for multi-GPU inference
- Implement async streaming with proper AsyncIterator
- Add retry logic with exponential backoff for transient errors
- Include comprehensive error handling for:
  - Model loading failures
  - OOM errors → trigger model unloading
  - Inference timeouts → return error gracefully
  - Invalid parameters → clear error messages

VLLMHandler class methods:
- `__init__()`: Already defined with parameters
- `load()`: Initialize vLLM engine with LLM class
- `generate(request: InferenceRequest) -> InferenceResponse`: Synchronous generation
- `generate_stream(request: InferenceRequest) -> AsyncIterator[str]`: Streaming generation
- `unload()`: Free VRAM and cleanup resources
- `_format_messages()`: Already implemented

Code structure:
```python
from vllm import LLM, SamplingParams
import asyncio
import logging
from typing import AsyncIterator

class VLLMHandler:
    def load(self):
        self._engine = LLM(
            model=self.model_name,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            trust_remote_code=True,
        )

    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        # Convert messages to prompt
        # Create SamplingParams
        # Call self._engine.generate()
        # Return InferenceResponse with content, usage, latency

    async def generate_stream(self, request: InferenceRequest) -> AsyncIterator[str]:
        # Use vLLM's streaming API
        # Yield tokens incrementally
```

=== FILES TO MODIFY ===

**ONLY modify:**
- `backend/inference/vllm_handler.py`

**DO NOT TOUCH (to avoid conflicts):**
- `backend/api/routes.py` - Integration happens after all agents finish
- `backend/api/main.py` - Will be wired up later
- `backend/inference/llamacpp_handler.py` - Agent 1B working on this
- `backend/inference/vision_handler.py` - Agent 1C working on this
- `backend/inference/translation_handler.py` - Agent 1D working on this
- Any test files in `tests/` - Agent 5 working on these
- `requirements.txt` - Will merge manually later

=== REFERENCE FILES ===

Existing structure to follow:
- `backend/inference/vllm_handler.py:1-180` - Your working file
- `backend/api/models.py` - Request/Response models
- `backend/monitoring/logger.py` - Logger setup (use logger.info(), logger.error())

InferenceRequest dataclass (already defined):
```python
@dataclass
class InferenceRequest:
    messages: List[Dict[str, str]]
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 0.9
    stop: Optional[List[str]] = None
    stream: bool = False
```

InferenceResponse dataclass (already defined):
```python
@dataclass
class InferenceResponse:
    content: str
    model: str
    usage: Dict[str, int]  # {prompt_tokens, completion_tokens, total_tokens}
    latency_ms: float
    finish_reason: str = "stop"
```

=== SUCCESS CRITERIA ===

Before committing, verify:
- [x] VLLMHandler.load() successfully initializes vLLM engine (or realistic mock)
- [x] VLLMHandler.generate() returns InferenceResponse with all required fields
- [x] VLLMHandler.generate_stream() yields tokens as AsyncIterator
- [x] Error handling catches and logs vLLM exceptions appropriately
- [x] OOM errors trigger model unloading
- [x] Code passes syntax check: `python -m py_compile backend/inference/vllm_handler.py`
- [x] Code is production-ready with logging and error handling

=== GIT WORKFLOW ===

```bash
# Start on your branch
git checkout -b feature/vllm-inference

# After completing work
git add backend/inference/vllm_handler.py
git commit -m "Implement production-ready vLLM inference handler

- Uncomment and complete vLLM integration
- Add streaming support with AsyncIterator
- Implement error handling with retries
- Add VRAM monitoring and OOM handling
- Include comprehensive logging
- Test with Qwen3-8B (or realistic mocks)"

# Push to remote
git push -u origin feature/vllm-inference
```

=== IMPORTANT NOTES ===

1. **Mock vs Real Implementation:** If vLLM library is not installed or models are unavailable, create realistic mocks that simulate the behavior but log warnings about mock mode.

2. **Streaming Implementation:** For streaming, vLLM provides async iteration over tokens. Ensure you yield each token as it's generated.

3. **Memory Management:** Monitor VRAM usage and implement graceful degradation if OOM occurs.

4. **Logging:** Use structured logging with context:
   ```python
   logger.info(f"Loading vLLM model: {self.model_name}")
   logger.error(f"Failed to load model: {error}", exc_info=True)
   ```

5. **No External Dependencies:** Don't reference other documents. This prompt contains everything you need.

=== READY TO START ===

You have all the information needed. Begin implementation of the vLLM handler.

When complete, run the syntax check and commit with the message above.
```

### PROMPT END

---

## 🔴 Agent 1B: llama.cpp Inference Handler

**Branch:** `feature/llamacpp-inference`

**Estimated Time:** 2-3 hours

### PROMPT START

```
I'm Agent 1B working on llama.cpp inference implementation for GroupQwen.

=== CONTEXT ===

The GroupQwen project is a local AI orchestration system that intelligently routes queries across multiple Qwen3 models. While Agent 1A is implementing vLLM for GPU inference, I'm implementing llama.cpp for edge devices (MacBook, low-power systems, CPU-only machines).

My role: Implement llama.cpp inference handler for CPU and Metal-accelerated inference with GGUF quantized models.

Current state:
- llama.cpp handler exists but is mostly a skeleton: backend/inference/llamacpp_handler.py
- Will handle GGUF quantized models (Qwen3-4B-Q4, Qwen3-8B-Q5, etc.)
- Needs to support both Metal (macOS GPU) and CPU backends
- Must match the InferenceRequest/InferenceResponse API from vllm_handler.py
- This is completely separate from vLLM - no file conflicts

=== TASKS TO COMPLETE ===

From TASKS.md Phase 1.3 - Complete Inference Handlers:

1. ✓ Implement complete LlamaCppHandler class
2. ✓ Add model loading with GGUF format support
3. ✓ Support Metal backend for macOS (use n_gpu_layers parameter)
4. ✓ Implement CPU inference for non-GPU systems
5. ✓ Add streaming support with generate_stream()
6. ✓ Include proper memory management and model unloading
7. ✓ Format messages properly for inference (use chat templates)
8. ✓ Add error handling and retries
9. ✓ Implement context window management (n_ctx parameter)
10. ✓ Add CPU thread optimization (n_threads parameter)

=== REQUIREMENTS ===

Technical Implementation:
- Use llama-cpp-python library: `from llama_cpp import Llama`
- Detect Metal availability on macOS, fallback to CPU otherwise
- Support GGUF model format (quantized models like Q4_K_M, Q5_K_M)
- Implement both synchronous and streaming generation
- Match InferenceRequest/InferenceResponse API from vLLM handler

LlamaCppHandler class structure:
```python
from llama_cpp import Llama
from typing import List, Dict, Optional, AsyncIterator
import logging
import platform
import os

class LlamaCppHandler:
    def __init__(
        self,
        model_path: str,
        n_ctx: int = 8192,
        n_threads: int = 8,
        use_metal: bool = True  # Auto-detect Metal on macOS
    ):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.use_metal = use_metal and platform.system() == "Darwin"
        self._llama = None

    def load(self):
        # Detect optimal settings
        n_gpu_layers = 35 if self.use_metal else 0  # Use Metal if available

        self._llama = Llama(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_threads=self.n_threads,
            verbose=False
        )

    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        # Format messages to prompt
        # Call self._llama.create_chat_completion() or __call__()
        # Return InferenceResponse

    async def generate_stream(self, request: InferenceRequest) -> AsyncIterator[str]:
        # Call with stream=True
        # Yield tokens incrementally

    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        # Format messages for Qwen chat template
        # Similar to vllm_handler._format_messages()

    def unload(self):
        # Free memory
        self._llama = None
```

Message formatting:
- Use Qwen chat template or simple format:
  ```
  System: {system_message}

  User: {user_message}

  Assistant:
  ```

Configuration parameters:
- `n_ctx`: Context window size (default 8192)
- `n_threads`: CPU threads to use (default: os.cpu_count() // 2)
- `n_gpu_layers`: Number of layers offloaded to GPU (Metal: 35, CPU: 0)
- `temperature`, `max_tokens`, `top_p`: From InferenceRequest

=== FILES TO MODIFY ===

**ONLY modify:**
- `backend/inference/llamacpp_handler.py`

**DO NOT TOUCH (to avoid conflicts):**
- `backend/api/routes.py` - Integration happens later
- `backend/api/main.py` - Will be wired up later
- `backend/inference/vllm_handler.py` - Agent 1A working on this
- `backend/inference/vision_handler.py` - Agent 1C working on this
- `backend/inference/translation_handler.py` - Agent 1D working on this
- Test files in `tests/` - Agent 5 working on these
- `requirements.txt` - Will merge manually later

=== REFERENCE FILES ===

Similar structure to follow:
- `backend/inference/vllm_handler.py` - Use same InferenceRequest/InferenceResponse
- `backend/api/models.py` - Request/Response models

InferenceRequest dataclass:
```python
@dataclass
class InferenceRequest:
    messages: List[Dict[str, str]]
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 0.9
    stop: Optional[List[str]] = None
    stream: bool = False
```

InferenceResponse dataclass:
```python
@dataclass
class InferenceResponse:
    content: str
    model: str
    usage: Dict[str, int]  # {prompt_tokens, completion_tokens, total_tokens}
    latency_ms: float
    finish_reason: str = "stop"
```

=== SUCCESS CRITERIA ===

Before committing, verify:
- [x] LlamaCppHandler class fully implemented
- [x] load() initializes llama.cpp with GGUF model (or realistic mock)
- [x] generate() returns InferenceResponse with all required fields
- [x] generate_stream() yields tokens as AsyncIterator
- [x] Detects and uses Metal on macOS, CPU otherwise
- [x] Memory cleanup in unload()
- [x] Error handling for model loading failures
- [x] Code passes syntax check: `python -m py_compile backend/inference/llamacpp_handler.py`

=== GIT WORKFLOW ===

```bash
# Start on your branch
git checkout -b feature/llamacpp-inference

# After completing work
git add backend/inference/llamacpp_handler.py
git commit -m "Implement llama.cpp handler for edge device inference

- Complete LlamaCppHandler class implementation
- Add GGUF model loading with Metal/CPU support
- Implement streaming generation
- Add context window and thread management
- Include error handling and memory cleanup
- Auto-detect Metal on macOS, fallback to CPU"

# Push to remote
git push -u origin feature/llamacpp-inference
```

=== IMPORTANT NOTES ===

1. **Metal Detection:** On macOS, use Metal acceleration by setting n_gpu_layers > 0. On Linux/Windows without CUDA, use CPU only (n_gpu_layers = 0).

2. **Model Path:** Models should be in GGUF format. Example paths:
   - `./models/qwen3-4b-q4_k_m.gguf`
   - `./models/qwen3-8b-q5_k_m.gguf`

3. **Streaming:** llama.cpp supports streaming via `stream=True` parameter. Yield each token dict's content.

4. **Mock Mode:** If llama-cpp-python not installed, create realistic mocks that simulate the behavior.

5. **Performance:** For CPU, optimize with n_threads. For Metal, use n_gpu_layers=35 or more.

=== READY TO START ===

You have all the information needed. Begin implementation of the llama.cpp handler.

When complete, run the syntax check and commit with the message above.
```

### PROMPT END

---

## 🟡 Agent 2A: Data Processing Pipeline

**Branch:** `feature/data-pipeline`

**Estimated Time:** 3-4 hours

### PROMPT START

```
I'm Agent 2A working on the fine-tuning data processing pipeline for GroupQwen.

=== CONTEXT ===

The GroupQwen project needs a fine-tuning pipeline so users can customize Qwen3 models with their own chat histories (ChatGPT, Claude, Perplexity exports).

My role: Build the data export, curation, and processing pipeline. Agent 2B will handle the actual training.

Pipeline flow: Export (ChatGPT/Claude) → Curate (quality filter, PII removal) → Process (convert to training format) → Train (Agent 2B handles this)

Current state:
- Skeleton files exist: backend/finetuning/data_processor.py, data_curator.py
- Need to create: scripts/export_chat_history.py, scripts/curate_dataset.py
- These files are completely separate from inference handlers - no conflicts
- Output will be JSONL format consumed by Agent 2B's trainer

=== TASKS TO COMPLETE ===

From TASKS.md Phase 3 - Fine-Tuning Pipeline:

**Part 1: Data Processing (backend/finetuning/data_processor.py)**
1. ✓ Implement convert_to_training_format(conversations: List[Dict]) -> str (JSONL)
2. ✓ Implement validate_dataset(dataset_path: str) -> bool
3. ✓ Support Qwen chat template format
4. ✓ Handle multi-turn conversations
5. ✓ Support multiple input formats (ChatGPT JSON, Claude JSON, plain text)

**Part 2: Data Curation (backend/finetuning/data_curator.py)**
6. ✓ Create DataCurator class with methods: load(), filter_quality(), remove_pii(), export()
7. ✓ Implement quality filtering (length > 10 chars, not all caps, has Q&A structure)
8. ✓ Implement PII detection/removal (emails, phone numbers, SSNs, addresses)
9. ✓ Export curated dataset in training format

**Part 3: Export Script (scripts/export_chat_history.py)**
10. ✓ Parse ChatGPT ZIP export (find conversations.json, parse, normalize)
11. ✓ Parse Claude export (JSON format)
12. ✓ Parse Perplexity export (if format available, or skip)
13. ✓ Normalize to common format: [{"role": "user/assistant", "content": "..."}]
14. ✓ Output: data/exported_conversations.json

**Part 4: Curation Script (scripts/curate_dataset.py)**
15. ✓ CLI interface with argparse or click
16. ✓ Interactive mode: show conversation, prompt keep/skip/edit
17. ✓ Auto mode: apply filters automatically
18. ✓ Progress tracking with tqdm or progress bars
19. ✓ Output: curated JSONL file

=== REQUIREMENTS ===

**1. data_processor.py Implementation:**

```python
from typing import List, Dict
import json

class DataProcessor:
    @staticmethod
    def convert_to_training_format(conversations: List[Dict]) -> str:
        """
        Convert conversations to JSONL training format.

        Input: List of conversations, each with messages
        Output: JSONL string, one conversation per line

        Format:
        {"messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]}
        """
        # Convert each conversation to JSONL line
        # Return joined JSONL string

    @staticmethod
    def validate_dataset(dataset_path: str) -> bool:
        """
        Validate JSONL dataset format.

        Checks:
        - File exists and is valid JSON lines
        - Each line has "messages" key
        - Messages have "role" and "content"
        - Roles are valid (user/assistant/system)
        """
        # Read JSONL file
        # Validate each line
        # Return True if valid, False otherwise

    @staticmethod
    def load_from_various_formats(file_path: str) -> List[Dict]:
        """
        Load conversations from ChatGPT, Claude, or plain text format.

        Supports:
        - ChatGPT: conversations.json from ZIP
        - Claude: JSON export
        - Plain text: simple Q&A format
        """
        # Detect format
        # Parse accordingly
        # Return normalized conversation list
```

**2. data_curator.py Implementation:**

```python
import re
from typing import List, Dict

class DataCurator:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.conversations = []

    def load(self) -> int:
        """Load conversations from file. Returns count."""
        # Load JSON/JSONL file
        # Parse conversations
        # Return count

    def filter_quality(self, min_length: int = 10, max_length: int = 10000) -> int:
        """
        Filter low-quality conversations.

        Removes:
        - Too short (< min_length chars)
        - Too long (> max_length chars)
        - All caps messages
        - Repetitive content
        - Single-turn conversations

        Returns: Number of conversations removed
        """
        # Apply quality filters
        # Update self.conversations
        # Return removed count

    def remove_pii(self) -> int:
        """
        Detect and remove PII (Personally Identifiable Information).

        Detects:
        - Email addresses: name@domain.com
        - Phone numbers: (123) 456-7890, 123-456-7890
        - SSNs: 123-45-6789
        - Physical addresses (best effort)
        - Credit card numbers

        Replaces with [REDACTED]

        Returns: Number of PII instances removed
        """
        pii_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b(\+\d{1,2}\s)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            # Add more patterns
        }
        # Scan all conversation content
        # Replace matches with [REDACTED]
        # Return count of replacements

    def export(self, output_path: str, format: str = 'jsonl') -> bool:
        """Export curated dataset to file."""
        # Write conversations to output file
        # Support JSONL and JSON formats
        # Return success status
```

**3. scripts/export_chat_history.py:**

```python
#!/usr/bin/env python3
"""
Export chat history from various sources to normalized format.

Usage:
    python scripts/export_chat_history.py --source chatgpt --input chatgpt_export.zip --output data/exported.json
    python scripts/export_chat_history.py --source claude --input claude_export.json --output data/exported.json
"""

import argparse
import json
import zipfile
from pathlib import Path
from typing import List, Dict

def export_chatgpt(zip_path: str) -> List[Dict]:
    """
    Export from ChatGPT ZIP format.

    ChatGPT exports contain conversations.json in ZIP.
    Format: {"id": "...", "mapping": {...}, ...}
    """
    # Extract ZIP
    # Find conversations.json
    # Parse ChatGPT format
    # Normalize to [{"role": "user", "content": "..."}]
    # Return conversation list

def export_claude(json_path: str) -> List[Dict]:
    """Export from Claude JSON format."""
    # Load JSON
    # Parse Claude format
    # Normalize to standard format
    # Return conversation list

def export_perplexity(json_path: str) -> List[Dict]:
    """Export from Perplexity format (if available)."""
    # Implement if format known, otherwise return []

def main():
    parser = argparse.ArgumentParser(description="Export chat history")
    parser.add_argument('--source', choices=['chatgpt', 'claude', 'perplexity'], required=True)
    parser.add_argument('--input', required=True, help='Input file path')
    parser.add_argument('--output', required=True, help='Output JSON file path')
    args = parser.parse_args()

    # Call appropriate export function
    # Save normalized conversations to output
    # Print summary

if __name__ == '__main__':
    main()
```

**4. scripts/curate_dataset.py:**

```python
#!/usr/bin/env python3
"""
Interactive dataset curation tool.

Usage:
    # Interactive mode
    python scripts/curate_dataset.py --input data/exported.json --output data/curated.jsonl --interactive

    # Auto mode (apply filters automatically)
    python scripts/curate_dataset.py --input data/exported.json --output data/curated.jsonl --auto
"""

import argparse
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.finetuning.data_curator import DataCurator

def interactive_curation(curator: DataCurator):
    """Show each conversation and prompt for keep/skip/edit."""
    # For each conversation:
    #   - Print conversation
    #   - Prompt: [k]eep, [s]kip, [e]dit, [q]uit?
    #   - Handle user input
    # Return curated list

def auto_curation(curator: DataCurator):
    """Apply filters automatically."""
    # Apply quality filters
    # Apply PII removal
    # Print statistics

def main():
    parser = argparse.ArgumentParser(description="Curate dataset for fine-tuning")
    parser.add_argument('--input', required=True, help='Input file (JSON/JSONL)')
    parser.add_argument('--output', required=True, help='Output JSONL file')
    parser.add_argument('--interactive', action='store_true', help='Interactive curation')
    parser.add_argument('--auto', action='store_true', help='Automatic curation')
    args = parser.parse_args()

    # Load dataset
    curator = DataCurator(args.input)
    curator.load()

    # Run curation
    if args.interactive:
        interactive_curation(curator)
    elif args.auto:
        auto_curation(curator)

    # Export curated dataset
    curator.export(args.output, format='jsonl')

    # Print summary

if __name__ == '__main__':
    main()
```

**Training Format (JSONL):**
```jsonl
{"messages": [{"role": "user", "content": "What is AI?"}, {"role": "assistant", "content": "AI is..."}]}
{"messages": [{"role": "system", "content": "You are helpful"}, {"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi!"}]}
```

=== FILES TO MODIFY ===

**ONLY modify:**
- `backend/finetuning/data_processor.py`
- `backend/finetuning/data_curator.py`
- `scripts/export_chat_history.py` (create new)
- `scripts/curate_dataset.py` (create new)

**DO NOT TOUCH (to avoid conflicts):**
- `backend/api/routes.py` - No API changes
- `backend/api/main.py` - No API changes
- `backend/finetuning/trainer.py` - Agent 2B working on this
- `backend/finetuning/checkpoint_manager.py` - Agent 2B working on this
- Inference handlers - Agents 1A/1B/1C/1D working on these
- Test files - Agent 5 working on these
- `requirements.txt` - Will merge manually later

=== SUCCESS CRITERIA ===

Before committing, verify:
- [x] data_processor.py converts conversations to JSONL format
- [x] data_processor.py validates JSONL datasets
- [x] data_curator.py filters quality and removes PII
- [x] export_chat_history.py exports from ChatGPT (and optionally Claude)
- [x] curate_dataset.py has working CLI with --help
- [x] Scripts are executable: `chmod +x scripts/*.py`
- [x] Code passes syntax check: `python -m py_compile backend/finetuning/*.py scripts/*.py`
- [x] Test scripts manually if possible

=== GIT WORKFLOW ===

```bash
# Start on your branch
git checkout -b feature/data-pipeline

# After completing work
git add backend/finetuning/data_processor.py backend/finetuning/data_curator.py
git add scripts/export_chat_history.py scripts/curate_dataset.py
git commit -m "Implement fine-tuning data export, curation, and processing pipeline

- Complete data_processor.py with JSONL conversion and validation
- Complete data_curator.py with quality filtering and PII removal
- Create export_chat_history.py for ChatGPT/Claude export
- Create curate_dataset.py with interactive and auto modes
- Support full pipeline: export → curate → process → training format"

# Push to remote
git push -u origin feature/data-pipeline
```

=== IMPORTANT NOTES ===

1. **ChatGPT Format:** ChatGPT exports are complex. The conversations.json has nested "mapping" structure. Extract the linear conversation flow.

2. **PII Removal:** Be aggressive with PII detection. Better to over-redact than leak sensitive info.

3. **Quality Filters:** Focus on conversational quality. Remove:
   - Single-word responses
   - Repetitive conversations
   - Broken/incomplete conversations

4. **JSONL Format:** Each line is a separate JSON object. No commas between lines. Valid JSONL:
   ```
   {"messages": [...]}
   {"messages": [...]}
   ```

5. **Error Handling:** Scripts should handle missing files, invalid JSON, and malformed data gracefully.

=== READY TO START ===

You have all the information needed. Begin implementation of the data processing pipeline.

When complete, run syntax checks and commit with the message above.
```

### PROMPT END

---

## 🟢 Agent 5: Testing Suite

**Branch:** `feature/test-suite`

**Estimated Time:** 2-3 hours

### PROMPT START

```
I'm Agent 5 working on comprehensive testing for GroupQwen.

=== CONTEXT ===

The GroupQwen project needs extensive test coverage to ensure reliability. Tests already exist for cache and router. I'm expanding coverage for inference handlers, API endpoints, and integration tests.

My role: Create comprehensive test suite covering all modules with mocking for external dependencies (vLLM, llama.cpp, Redis).

Current state:
- Existing tests: tests/test_cache.py, tests/test_router.py
- Need to create: tests/test_inference.py, tests/test_api.py, tests/integration/test_end_to_end.py
- Need to expand: tests/test_router.py with more scenarios
- All test files are isolated - no conflicts with implementation agents
- Use pytest with mocking (unittest.mock or pytest-mock)

=== TASKS TO COMPLETE ===

From TASKS.md Phase 6 - Testing & Quality:

**Part 1: Inference Tests (tests/test_inference.py)**
1. ✓ Test VLLMHandler.load(), generate(), generate_stream()
2. ✓ Test LlamaCppHandler.load(), generate(), generate_stream()
3. ✓ Test VisionHandler basic functionality (can use placeholders)
4. ✓ Test TranslationHandler basic functionality (can use placeholders)
5. ✓ Mock model responses (don't require actual models)
6. ✓ Test error handling (OOM, timeouts, invalid inputs)

**Part 2: API Tests (tests/test_api.py)**
7. ✓ Test POST /v1/chat/completions (with mock router)
8. ✓ Test GET /v1/health
9. ✓ Test GET /v1/metrics
10. ✓ Test GET /v1/cache/stats
11. ✓ Test error handling (400 for empty messages, 500 for backend errors)
12. ✓ Test caching behavior (cache hit/miss)

**Part 3: Router Tests Expansion (tests/test_router.py)**
13. ✓ Add more routing scenarios (complex vs simple queries)
14. ✓ Test edge cases (empty query, very long query, special characters)
15. ✓ Test device-specific routing (server, macbook, mobile)
16. ✓ Test model escalation logic
17. ✓ Test forced model routing (force_model parameter)

**Part 4: Integration Tests (tests/integration/test_end_to_end.py)**
18. ✓ Test full query flow: API → Router → Cache → Inference → Response
19. ✓ Test cache integration: First call misses, second call hits
20. ✓ Test metrics tracking: Verify MetricsCollector logs requests
21. ✓ Use pytest fixtures for setup/teardown

=== REQUIREMENTS ===

**1. tests/test_inference.py:**

```python
import pytest
from unittest.mock import Mock, patch, AsyncMock
from backend.inference.vllm_handler import VLLMHandler, InferenceRequest, InferenceResponse
from backend.inference.llamacpp_handler import LlamaCppHandler

class TestVLLMHandler:
    @patch('backend.inference.vllm_handler.LLM')
    def test_load_success(self, mock_llm):
        """Test successful model loading."""
        handler = VLLMHandler(model_name="qwen3-8b")
        handler.load()
        assert handler._engine is not None
        mock_llm.assert_called_once()

    @pytest.mark.asyncio
    @patch('backend.inference.vllm_handler.LLM')
    async def test_generate_success(self, mock_llm):
        """Test successful generation."""
        # Mock vLLM engine
        mock_engine = Mock()
        mock_output = Mock()
        mock_output.outputs = [Mock(text="Hello!")]
        mock_engine.generate.return_value = [mock_output]

        handler = VLLMHandler(model_name="qwen3-8b")
        handler._engine = mock_engine

        request = InferenceRequest(
            messages=[{"role": "user", "content": "Hi"}],
            temperature=0.7
        )
        response = await handler.generate(request)

        assert isinstance(response, InferenceResponse)
        assert response.content is not None
        assert response.latency_ms > 0

    @pytest.mark.asyncio
    @patch('backend.inference.vllm_handler.LLM')
    async def test_generate_stream(self, mock_llm):
        """Test streaming generation."""
        handler = VLLMHandler(model_name="qwen3-8b")
        handler._engine = Mock()

        request = InferenceRequest(
            messages=[{"role": "user", "content": "Hi"}],
            stream=True
        )

        chunks = []
        async for chunk in handler.generate_stream(request):
            chunks.append(chunk)

        assert len(chunks) > 0

    def test_error_handling_oom(self):
        """Test OOM error handling."""
        # Test that OOM triggers unload()

class TestLlamaCppHandler:
    @patch('backend.inference.llamacpp_handler.Llama')
    def test_load_with_metal(self, mock_llama):
        """Test loading with Metal backend."""
        # Test Metal detection and initialization

    @pytest.mark.asyncio
    async def test_generate(self):
        """Test generate method."""
        # Similar to vLLM tests
```

**2. tests/test_api.py:**

```python
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
from backend.api.main import app

class TestChatCompletionAPI:
    def test_chat_completion_success(self):
        """Test successful chat completion."""
        # Mock app.state components
        mock_router = Mock()
        mock_router.route_query.return_value = Mock(
            model="qwen3-8b",
            complexity_score=0.5,
            use_cache=False,
            estimated_latency_ms=100
        )

        with patch.object(app.state, 'query_router', mock_router):
            client = TestClient(app)
            response = client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "Hello"}],
                    "temperature": 0.7
                }
            )

        assert response.status_code == 200
        assert "choices" in response.json()

    def test_chat_completion_empty_messages(self):
        """Test error handling for empty messages."""
        client = TestClient(app)
        response = client.post(
            "/v1/chat/completions",
            json={"messages": []}
        )
        assert response.status_code == 400

    def test_cache_hit(self):
        """Test cache hit scenario."""
        # Mock cache_manager with hit

    def test_cache_miss(self):
        """Test cache miss scenario."""
        # Mock cache_manager with miss

class TestHealthAPI:
    def test_health_check(self):
        """Test health check endpoint."""
        client = TestClient(app)
        response = client.get("/v1/health")
        assert response.status_code == 200
        assert "status" in response.json()

class TestMetricsAPI:
    def test_metrics_endpoint(self):
        """Test metrics endpoint."""
        client = TestClient(app)
        response = client.get("/v1/metrics")
        assert response.status_code == 200
```

**3. Expand tests/test_router.py:**

```python
# Add to existing test_router.py

def test_routing_complex_query():
    """Test routing for complex queries."""
    router = QueryRouter(cache_manager, complexity_classifier)

    query = "Explain quantum computing and its implications for cryptography in detail."
    decision = router.route_query(query, context="", device="server")

    assert decision.model in ["qwen3-14b", "qwen3-32b"]  # Complex query

def test_routing_simple_query():
    """Test routing for simple queries."""
    query = "What is 2+2?"
    decision = router.route_query(query, context="", device="server")

    assert decision.model == "qwen3-8b"  # Simple query

def test_device_specific_routing_mobile():
    """Test routing for mobile devices."""
    query = "Hello"
    decision = router.route_query(query, context="", device="mobile")

    assert decision.model == "qwen3-4b"  # Mobile uses smallest model

def test_force_model_override():
    """Test forced model routing."""
    query = "Complex query"
    decision = router.route_query(query, context="", device="server", force_model="qwen3-4b")

    assert decision.model == "qwen3-4b"  # Forced override

def test_edge_case_empty_query():
    """Test empty query handling."""
    # Should handle gracefully

def test_edge_case_very_long_query():
    """Test very long query (>10k tokens)."""
    # Should route to model with large context window
```

**4. tests/integration/test_end_to_end.py:**

```python
import pytest
from fastapi.testclient import TestClient
from backend.api.main import app
from backend.core.cache_manager import SemanticCacheManager
from backend.core.router import QueryRouter
from backend.monitoring.metrics import MetricsCollector
import redis

@pytest.fixture
def test_app():
    """Setup test app with real components."""
    # Create real components (or mocks)
    cache_manager = SemanticCacheManager(redis_client=Mock())
    query_router = QueryRouter(cache_manager)
    metrics_collector = MetricsCollector()

    # Attach to app state
    app.state.cache_manager = cache_manager
    app.state.query_router = query_router
    app.state.metrics_collector = metrics_collector

    yield app

def test_full_query_flow(test_app):
    """Test complete flow: API → Router → Cache → Response."""
    client = TestClient(test_app)

    # First request (cache miss)
    response1 = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "Hello"}]}
    )
    assert response1.status_code == 200

    # Second request (cache hit)
    response2 = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "Hello"}]}
    )
    assert response2.status_code == 200
    # Verify second request is faster (cache hit)

def test_cache_integration():
    """Test cache stores and retrieves correctly."""
    # Test cache miss → store → cache hit

def test_metrics_tracking():
    """Test metrics are logged correctly."""
    # Verify MetricsCollector logs requests
```

=== FILES TO MODIFY ===

**ONLY modify:**
- `tests/test_inference.py` (create new)
- `tests/test_api.py` (create new)
- `tests/test_router.py` (expand existing)
- `tests/integration/test_end_to_end.py` (create new with directory)

**DO NOT TOUCH (to avoid conflicts):**
- `backend/*` - Implementation code (don't modify)
- `scripts/*` - Data processing scripts
- `requirements.txt` - Will merge manually later

=== REFERENCE FILES ===

Existing tests to reference:
- `tests/test_router.py` - Existing router tests
- `tests/test_cache.py` - Existing cache tests

API and implementation files:
- `backend/api/routes.py` - API endpoints to test
- `backend/inference/vllm_handler.py` - VLLMHandler to test
- `backend/inference/llamacpp_handler.py` - LlamaCppHandler to test
- `backend/core/router.py` - QueryRouter to test

=== SUCCESS CRITERIA ===

Before committing, verify:
- [x] All test files created and executable with pytest
- [x] Tests use mocking for external dependencies (no real models needed)
- [x] Coverage for both happy path and error cases
- [x] Integration test demonstrates full flow
- [x] All tests pass: `pytest tests/ -v`
- [x] No import errors: `python -c "import tests.test_inference"`

Run tests:
```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_inference.py -v

# Run with coverage
pytest tests/ --cov=backend --cov-report=html
```

=== GIT WORKFLOW ===

```bash
# Start on your branch
git checkout -b feature/test-suite

# After completing work
git add tests/test_inference.py tests/test_api.py
git add tests/test_router.py  # If expanded
git add tests/integration/test_end_to_end.py
git commit -m "Add comprehensive test suite for inference, API, and integration

- Create test_inference.py with vLLM and llama.cpp tests
- Create test_api.py with API endpoint tests
- Expand test_router.py with edge cases and device routing
- Add integration tests for end-to-end flow
- Mock external dependencies (models, Redis)
- All tests pass with pytest"

# Push to remote
git push -u origin feature/test-suite
```

=== IMPORTANT NOTES ===

1. **Mocking:** Use `unittest.mock.patch` to mock vLLM, llama.cpp, and Redis. Tests should run without real dependencies.

2. **Async Tests:** Use `@pytest.mark.asyncio` for async test functions.

3. **Fixtures:** Use pytest fixtures for common setup (test app, mock components).

4. **Coverage:** Aim for >80% coverage of core modules (router, cache, inference).

5. **Fast Tests:** Tests should run quickly (<30s total). Use mocks, not real models.

=== READY TO START ===

You have all the information needed. Begin implementation of the test suite.

When complete, run pytest and commit with the message above.
```

### PROMPT END

---

# Phase B: Enhanced Features (Run These 3 Agents After Phase A)

**Priority:** Start after Phase A completes, adds vision/translation/training
**Estimated Total Time:** 3-4 hours (vs 9 hours sequentially)

---

## 🟣 Agent 1C: Vision Handler

**Branch:** `feature/vision-handler`

**Estimated Time:** 2-3 hours

### PROMPT START

```
I'm Agent 1C working on Qwen3-VL vision capabilities for GroupQwen.

=== CONTEXT ===

The GroupQwen project supports multiple Qwen3 variants. I'm implementing Qwen3-VL for vision tasks: image understanding, OCR, bounding box detection, and visual Q&A.

My role: Implement production-ready vision handler for Qwen3-VL that processes images and returns descriptions, OCR text, and bounding boxes.

Current state:
- Vision handler skeleton likely exists: backend/inference/vision_handler.py
- API endpoint already exists: POST /v1/vision/analyze (in routes.py)
- Need to implement: VisionHandler class with image preprocessing and Qwen3-VL integration
- This is separate from other inference handlers - no conflicts
- Agent 1A implemented vLLM - use similar patterns

=== TASKS TO COMPLETE ===

From TASKS.md Phase 1.3 - Complete Inference Handlers:

1. ✓ Complete backend/inference/vision_handler.py
2. ✓ Implement Qwen3-VL integration (similar to vLLM but with image input)
3. ✓ Add image preprocessing (resize, format conversion, base64 decode)
4. ✓ Support bounding box detection (return_bboxes parameter)
5. ✓ Handle base64 and URL image inputs
6. ✓ Add OCR and visual reasoning capabilities
7. ✓ Add error handling for invalid images
8. ✓ Support common image formats (JPEG, PNG, WebP)

=== REQUIREMENTS ===

**VisionHandler class structure:**

```python
from typing import List, Dict, Optional
import logging
import base64
import requests
from PIL import Image
from io import BytesIO
from dataclasses import dataclass

@dataclass
class VisionResponse:
    """Response from vision analysis."""
    description: str
    bounding_boxes: Optional[List[Dict]] = None  # [{"label": "cat", "box": [x1,y1,x2,y2]}]
    ocr_text: Optional[str] = None
    latency_ms: float = 0.0

class VisionHandler:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-7B",
        max_image_size: tuple = (1024, 1024)
    ):
        self.model_name = model_name
        self.max_image_size = max_image_size
        self._model = None

    def load(self):
        """
        Load Qwen3-VL model.

        Options:
        1. Use vLLM for GPU inference (like Agent 1A's implementation)
        2. Use transformers for direct model loading
        3. Use mock for testing
        """
        # Initialize Qwen3-VL model
        # Can use vLLM or transformers

    async def analyze_image(
        self,
        image: str,  # base64 string or URL
        prompt: str = "Describe this image in detail",
        return_bboxes: bool = False
    ) -> VisionResponse:
        """
        Analyze image with Qwen3-VL.

        Args:
            image: Base64-encoded image or URL
            prompt: Question/instruction about the image
            return_bboxes: Whether to detect and return bounding boxes

        Returns:
            VisionResponse with description, boxes, OCR text
        """
        import time
        start_time = time.time()

        # 1. Preprocess image
        img = self._preprocess_image(image)

        # 2. Format prompt for Qwen-VL
        if return_bboxes:
            prompt = f"{prompt} Provide bounding boxes for objects."

        # 3. Run inference
        # Use Qwen-VL's image-text format
        # Model input: <img>image_tensor</img>User: {prompt}\nAssistant:

        # 4. Parse output
        # Extract description, bounding boxes, OCR text

        # 5. Return response
        latency_ms = (time.time() - start_time) * 1000

        return VisionResponse(
            description="[Generated description]",
            bounding_boxes=[...] if return_bboxes else None,
            ocr_text="[Extracted text]" if "OCR" in prompt else None,
            latency_ms=latency_ms
        )

    def _preprocess_image(self, image: str) -> Image.Image:
        """
        Preprocess image from base64 or URL.

        Steps:
        1. Decode base64 or fetch URL
        2. Convert to PIL Image
        3. Resize if larger than max_image_size
        4. Convert to RGB

        Returns:
            PIL Image object
        """
        # Detect if base64 or URL
        if image.startswith('http://') or image.startswith('https://'):
            # Fetch from URL
            response = requests.get(image, timeout=10)
            img = Image.open(BytesIO(response.content))
        else:
            # Decode base64
            # Handle data:image/png;base64,... format
            if ',' in image:
                image = image.split(',', 1)[1]
            img_data = base64.b64decode(image)
            img = Image.open(BytesIO(img_data))

        # Resize if needed
        if img.size[0] > self.max_image_size[0] or img.size[1] > self.max_image_size[1]:
            img.thumbnail(self.max_image_size, Image.Resampling.LANCZOS)

        # Convert to RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')

        return img

    def _parse_bounding_boxes(self, text: str) -> List[Dict]:
        """
        Parse bounding boxes from Qwen-VL output.

        Qwen-VL format: <ref>cat</ref><box>[[x1,y1,x2,y2]]</box>

        Returns:
            List of {"label": "cat", "box": [x1, y1, x2, y2]}
        """
        import re
        boxes = []

        # Extract <ref>...</ref><box>...</box> pairs
        pattern = r'<ref>(.*?)</ref><box>\[\[(\d+),(\d+),(\d+),(\d+)\]\]</box>'
        matches = re.findall(pattern, text)

        for match in matches:
            label, x1, y1, x2, y2 = match
            boxes.append({
                "label": label,
                "box": [int(x1), int(y1), int(x2), int(y2)]
            })

        return boxes

    def unload(self):
        """Unload model and free resources."""
        self._model = None
```

**Image Input Formats:**

1. **Base64:**
   ```
   "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."
   ```

2. **URL:**
   ```
   "https://example.com/image.jpg"
   ```

**Qwen-VL Prompt Format:**
```
<img>IMAGE_PLACEHOLDER</img>
User: Describe this image
Assistant:
```

**Bounding Box Format:**
```python
[
    {"label": "cat", "box": [100, 150, 300, 400]},  # x1, y1, x2, y2
    {"label": "dog", "box": [350, 200, 550, 450]}
]
```

=== FILES TO MODIFY ===

**ONLY modify:**
- `backend/inference/vision_handler.py`

**DO NOT TOUCH (to avoid conflicts):**
- `backend/api/routes.py` - Integration happens later
- `backend/api/main.py` - Will be wired up later
- `backend/inference/vllm_handler.py` - Agent 1A completed this
- `backend/inference/llamacpp_handler.py` - Agent 1B completed this
- `backend/inference/translation_handler.py` - Agent 1D working on this
- Test files - Agent 5 may add tests later
- `requirements.txt` - Will merge manually later

=== REFERENCE FILES ===

Similar structure to follow:
- `backend/inference/vllm_handler.py` - Use similar loading and inference patterns
- `backend/api/models.py` - VisionAnalysisRequest/Response models

API models (for reference):
```python
class VisionAnalysisRequest(BaseModel):
    image: str  # Base64 or URL
    prompt: Optional[str] = "Describe this image"
    return_bboxes: bool = False

class VisionAnalysisResponse(BaseModel):
    description: str
    bounding_boxes: Optional[List[Dict]] = None
    ocr_text: Optional[str] = None
    model: str
    latency_ms: float
```

=== SUCCESS CRITERIA ===

Before committing, verify:
- [x] VisionHandler class fully implemented
- [x] load() initializes Qwen3-VL (or realistic mock)
- [x] analyze_image() processes images and returns descriptions
- [x] Supports both base64 and URL inputs
- [x] Preprocessing resizes and converts images correctly
- [x] Bounding box detection works (even if mocked)
- [x] OCR extraction works (even if mocked)
- [x] Error handling for invalid images (corrupt, unsupported format)
- [x] Code passes syntax check: `python -m py_compile backend/inference/vision_handler.py`

=== GIT WORKFLOW ===

```bash
# Start on your branch
git checkout -b feature/vision-handler

# After completing work
git add backend/inference/vision_handler.py
git commit -m "Implement Qwen3-VL vision analysis handler

- Complete VisionHandler class with image preprocessing
- Add support for base64 and URL image inputs
- Implement bounding box detection
- Add OCR text extraction
- Handle image resize and format conversion
- Include error handling for invalid images
- Support Qwen3-VL model integration"

# Push to remote
git push -u origin feature/vision-handler
```

=== IMPORTANT NOTES ===

1. **Model Loading:** If Qwen3-VL not available, use realistic mocks that simulate vision analysis behavior.

2. **Image Processing:** Use PIL (Pillow) for image operations. Handle common formats: JPEG, PNG, WebP, GIF.

3. **Bounding Boxes:** Qwen-VL uses normalized coordinates [0-1000]. Convert to pixel coordinates if needed.

4. **OCR:** For OCR, instruct model to extract visible text. Qwen-VL is capable of OCR without additional libraries.

5. **Error Handling:** Catch exceptions for:
   - Invalid base64
   - URL fetch failures
   - Corrupt images
   - Unsupported formats

6. **Dependencies:** Will need: `Pillow`, `requests`. Add to imports but don't modify requirements.txt.

=== READY TO START ===

You have all the information needed. Begin implementation of the vision handler.

When complete, run syntax check and commit with the message above.
```

### PROMPT END

---

## 🟣 Agent 1D: Translation Handler

**Branch:** `feature/translation-handler`

**Estimated Time:** 2-3 hours

### PROMPT START

```
I'm Agent 1D working on Qwen3-MT translation capabilities for GroupQwen.

=== CONTEXT ===

The GroupQwen project supports multiple Qwen3 variants. I'm implementing Qwen3-MT for translation tasks across 92 languages including Chinese, Japanese, Korean, Arabic, and European languages.

My role: Implement production-ready translation handler for Qwen3-MT with language detection, multi-language support, and formatting preservation.

Current state:
- Translation handler skeleton likely exists: backend/inference/translation_handler.py
- API endpoint already exists: POST /v1/translate (in routes.py)
- Need to implement: TranslationHandler class with language detection and Qwen3-MT integration
- This is separate from other inference handlers - no conflicts
- Agent 1A implemented vLLM - use similar patterns

=== TASKS TO COMPLETE ===

From TASKS.md Phase 1.3 - Complete Inference Handlers:

1. ✓ Complete backend/inference/translation_handler.py
2. ✓ Implement Qwen3-MT integration
3. ✓ Add language detection (auto-detect source language)
4. ✓ Support 92 languages (use ISO 639-1 codes: en, es, zh, ja, ko, etc.)
5. ✓ Add formatting preservation (maintain newlines, bullet points, etc.)
6. ✓ Handle edge cases (empty text, unsupported languages)
7. ✓ Add batch translation support (optional enhancement)

=== REQUIREMENTS ===

**TranslationHandler class structure:**

```python
from typing import Optional
import logging
from dataclasses import dataclass

@dataclass
class TranslationResponse:
    """Response from translation."""
    translated_text: str
    source_lang: str  # Detected or provided (ISO 639-1 code)
    target_lang: str  # ISO 639-1 code
    latency_ms: float

class TranslationHandler:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-MT-8B"
    ):
        self.model_name = model_name
        self._model = None
        self.supported_languages = self._get_supported_languages()

    def load(self):
        """
        Load Qwen3-MT model.

        Options:
        1. Use vLLM for GPU inference (like Agent 1A's implementation)
        2. Use transformers for direct model loading
        3. Use mock for testing
        """
        # Initialize Qwen3-MT model
        # Can use vLLM or transformers

    async def translate(
        self,
        text: str,
        target_lang: str,
        source_lang: Optional[str] = None
    ) -> TranslationResponse:
        """
        Translate text to target language.

        Args:
            text: Text to translate
            target_lang: Target language code (ISO 639-1: en, es, zh, ja, etc.)
            source_lang: Source language code (auto-detect if None)

        Returns:
            TranslationResponse with translated text and language info
        """
        import time
        start_time = time.time()

        # 1. Detect source language if not provided
        if source_lang is None:
            source_lang = self.detect_language(text)

        # 2. Validate languages
        if target_lang not in self.supported_languages:
            raise ValueError(f"Unsupported target language: {target_lang}")

        # 3. Format prompt for Qwen3-MT
        prompt = self._format_translation_prompt(text, source_lang, target_lang)

        # 4. Run inference
        # Use model to translate
        translated_text = self._run_inference(prompt)

        # 5. Post-process (preserve formatting)
        translated_text = self._preserve_formatting(text, translated_text)

        latency_ms = (time.time() - start_time) * 1000

        return TranslationResponse(
            translated_text=translated_text,
            source_lang=source_lang,
            target_lang=target_lang,
            latency_ms=latency_ms
        )

    def detect_language(self, text: str) -> str:
        """
        Detect language of text.

        Options:
        1. Use langdetect library
        2. Use Qwen3-MT's built-in detection
        3. Simple heuristics (character sets)

        Returns:
            ISO 639-1 language code
        """
        try:
            from langdetect import detect
            lang_code = detect(text)
            return lang_code
        except:
            # Fallback: simple heuristics
            if self._contains_chinese(text):
                return 'zh'
            elif self._contains_japanese(text):
                return 'ja'
            elif self._contains_korean(text):
                return 'ko'
            elif self._contains_arabic(text):
                return 'ar'
            else:
                return 'en'  # Default to English

    def _format_translation_prompt(
        self,
        text: str,
        source_lang: str,
        target_lang: str
    ) -> str:
        """
        Format prompt for Qwen3-MT.

        Qwen3-MT prompt format:
        Translate the following text from {source_lang} to {target_lang}:

        {text}

        Translation:
        """
        lang_names = self._get_language_names()
        source_name = lang_names.get(source_lang, source_lang)
        target_name = lang_names.get(target_lang, target_lang)

        prompt = f"""Translate the following text from {source_name} to {target_name}. Preserve all formatting including newlines, bullet points, and structure.

Source text:
{text}

Translation:"""

        return prompt

    def _run_inference(self, prompt: str) -> str:
        """
        Run model inference.

        Use similar approach to vllm_handler.py
        """
        # Call model with prompt
        # Return generated text
        # For now, return mock
        return "[Translated text]"

    def _preserve_formatting(self, original: str, translated: str) -> str:
        """
        Preserve formatting from original text.

        Maintains:
        - Newline structure
        - Bullet points
        - Numbered lists
        - Indentation (approximately)
        """
        # If original has newlines, ensure translated has similar structure
        # This is best-effort formatting preservation
        return translated

    def _get_supported_languages(self) -> set:
        """
        Get set of supported language codes.

        Qwen3-MT supports 92 languages. Return at least major languages.
        """
        return {
            'en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ar',  # European + Arabic
            'zh', 'ja', 'ko',  # CJK
            'hi', 'bn', 'ur',  # South Asian
            'id', 'ms', 'th', 'vi',  # Southeast Asian
            'nl', 'pl', 'tr', 'sv', 'da', 'no', 'fi',  # More European
            # Add more as needed up to 92 languages
        }

    def _get_language_names(self) -> dict:
        """Map language codes to full names."""
        return {
            'en': 'English',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese',
            'ru': 'Russian',
            'ar': 'Arabic',
            'zh': 'Chinese',
            'ja': 'Japanese',
            'ko': 'Korean',
            'hi': 'Hindi',
            'bn': 'Bengali',
            # Add more
        }

    def _contains_chinese(self, text: str) -> bool:
        """Check if text contains Chinese characters."""
        return any('\u4e00' <= char <= '\u9fff' for char in text)

    def _contains_japanese(self, text: str) -> bool:
        """Check if text contains Japanese characters."""
        return any('\u3040' <= char <= '\u309f' or '\u30a0' <= char <= '\u30ff' for char in text)

    def _contains_korean(self, text: str) -> bool:
        """Check if text contains Korean characters."""
        return any('\uac00' <= char <= '\ud7af' for char in text)

    def _contains_arabic(self, text: str) -> bool:
        """Check if text contains Arabic characters."""
        return any('\u0600' <= char <= '\u06ff' for char in text)

    def unload(self):
        """Unload model and free resources."""
        self._model = None
```

**Supported Languages (minimum 20, ideally 92):**

Major languages to support:
1. English (en), Spanish (es), French (fr), German (de), Italian (it), Portuguese (pt)
2. Chinese (zh), Japanese (ja), Korean (ko)
3. Arabic (ar), Russian (ru), Turkish (tr)
4. Hindi (hi), Bengali (bn), Urdu (ur)
5. Indonesian (id), Malay (ms), Thai (th), Vietnamese (vi)
6. Dutch (nl), Polish (pl), Swedish (sv), Danish (da), Norwegian (no), Finnish (fi)

**Translation Prompt Format:**
```
Translate the following text from English to Spanish. Preserve all formatting.

Source text:
Hello, world!
- Item 1
- Item 2

Translation:
```

=== FILES TO MODIFY ===

**ONLY modify:**
- `backend/inference/translation_handler.py`

**DO NOT TOUCH (to avoid conflicts):**
- `backend/api/routes.py` - Integration happens later
- `backend/api/main.py` - Will be wired up later
- `backend/inference/vllm_handler.py` - Agent 1A completed this
- `backend/inference/llamacpp_handler.py` - Agent 1B completed this
- `backend/inference/vision_handler.py` - Agent 1C working on this
- Test files - Agent 5 may add tests later
- `requirements.txt` - Will merge manually later

=== REFERENCE FILES ===

Similar structure to follow:
- `backend/inference/vllm_handler.py` - Use similar loading and inference patterns
- `backend/api/models.py` - TranslationRequest/Response models

API models (for reference):
```python
class TranslationRequest(BaseModel):
    text: str
    target_lang: str  # ISO 639-1 code
    source_lang: Optional[str] = None  # Auto-detect if None

class TranslationResponse(BaseModel):
    translated_text: str
    source_lang: str
    target_lang: str
    model: str
    latency_ms: float
```

=== SUCCESS CRITERIA ===

Before committing, verify:
- [x] TranslationHandler class fully implemented
- [x] load() initializes Qwen3-MT (or realistic mock)
- [x] translate() returns translations in target language
- [x] detect_language() identifies source language correctly
- [x] Supports at least 20 major languages (ideally 92)
- [x] Formatting preserved in translations (newlines, structure)
- [x] Error handling for unsupported languages
- [x] Code passes syntax check: `python -m py_compile backend/inference/translation_handler.py`

=== GIT WORKFLOW ===

```bash
# Start on your branch
git checkout -b feature/translation-handler

# After completing work
git add backend/inference/translation_handler.py
git commit -m "Implement Qwen3-MT translation handler with 92-language support

- Complete TranslationHandler class implementation
- Add language detection with langdetect fallback
- Support 92 languages with ISO 639-1 codes
- Implement formatting preservation
- Add character set detection for CJK and Arabic
- Include error handling for unsupported languages
- Integrate Qwen3-MT model"

# Push to remote
git push -u origin feature/translation-handler
```

=== IMPORTANT NOTES ===

1. **Language Detection:** Use langdetect library if available, otherwise use character set heuristics.

2. **ISO 639-1 Codes:** Standard two-letter codes (en, es, zh, ja, etc.). Full list: https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes

3. **Formatting Preservation:** Instruct model to preserve formatting. Best-effort approach - exact preservation is hard.

4. **Model Loading:** If Qwen3-MT not available, use realistic mocks that simulate translation behavior.

5. **Error Handling:** Catch exceptions for:
   - Empty text
   - Unsupported language codes
   - Model inference failures
   - Invalid UTF-8 encoding

6. **Dependencies:** Will need: `langdetect` (optional). Add to imports but don't modify requirements.txt.

=== READY TO START ===

You have all the information needed. Begin implementation of the translation handler.

When complete, run syntax check and commit with the message above.
```

### PROMPT END

---

## 🟡 Agent 2B: Training Pipeline

**Branch:** `feature/training-pipeline`

**Estimated Time:** 3-4 hours

### PROMPT START

```
I'm Agent 2B working on the fine-tuning training pipeline for GroupQwen.

=== CONTEXT ===

The GroupQwen project needs a fine-tuning pipeline for users to customize Qwen3 models. Agent 2A completed the data processing pipeline (export, curate, process). I'm building the training engine.

My role: Implement QLoRA training with Unsloth, checkpoint management, and fine-tuning API endpoints.

Pipeline flow: Agent 2A's curated JSONL → Trainer (me) → Checkpoints → Trained model

Current state:
- Agent 2A completed: data export, curation, processing to JSONL
- Skeleton files exist: backend/finetuning/trainer.py, checkpoint_manager.py
- Need to create: scripts/finetune_model.py
- Need to wire up: Fine-tuning API endpoints in routes.py (ONLY fine-tuning endpoints)
- Training uses QLoRA (Quantized LoRA) with Unsloth for efficient fine-tuning

=== TASKS TO COMPLETE ===

From TASKS.md Phase 3 - Fine-Tuning Pipeline:

**Part 1: Trainer (backend/finetuning/trainer.py)**
1. ✓ Uncomment Unsloth code (likely commented like vLLM was)
2. ✓ Implement QLoRA training with Unsloth
3. ✓ Add progress tracking (epochs, loss, learning rate)
4. ✓ Implement checkpointing every N steps
5. ✓ Support resume from checkpoint
6. ✓ Add validation during training (optional)

**Part 2: Checkpoint Manager (backend/finetuning/checkpoint_manager.py)**
7. ✓ Save checkpoints to disk (checkpoints/ directory)
8. ✓ Load checkpoints for resume
9. ✓ Track metrics (loss, perplexity, learning rate per step)
10. ✓ List available checkpoints
11. ✓ Clean up old checkpoints (keep last N)

**Part 3: Fine-tune Script (scripts/finetune_model.py)**
12. ✓ CLI: `python scripts/finetune_model.py --model qwen3-8b --dataset data/curated.jsonl --epochs 3`
13. ✓ Start training with Trainer
14. ✓ Monitor progress with tqdm or progress bars
15. ✓ Handle interruptions (save checkpoint on Ctrl+C)
16. ✓ Output trained model to models/ directory

**Part 4: API Integration (backend/api/routes.py - ONLY fine-tuning endpoints)**
17. ✓ POST /v1/finetune/start: Start background training job
18. ✓ GET /v1/finetune/status/{job_id}: Check job status
19. ✓ POST /v1/finetune/cancel/{job_id}: Cancel job
20. ✓ Use BackgroundTasks for async training
21. ✓ Maintain in-memory job registry

=== REQUIREMENTS ===

**1. backend/finetuning/trainer.py:**

```python
from typing import Optional, Dict
import logging
from dataclasses import dataclass
from pathlib import Path

@dataclass
class TrainingConfig:
    """Training configuration."""
    model_name: str  # Base model: qwen3-8b, qwen3-14b
    dataset_path: str  # Path to JSONL dataset
    output_dir: str  # Output directory for trained model
    epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-4
    max_seq_length: int = 2048
    lora_rank: int = 16
    lora_alpha: int = 32
    save_steps: int = 100
    logging_steps: int = 10

@dataclass
class TrainingResult:
    """Training result."""
    success: bool
    output_dir: str
    final_loss: float
    epochs_completed: int
    steps_completed: int
    error: Optional[str] = None

class Trainer:
    def __init__(self, config: TrainingConfig, checkpoint_manager=None):
        self.config = config
        self.checkpoint_manager = checkpoint_manager
        self.model = None
        self.tokenizer = None

    def load_dataset(self):
        """
        Load JSONL dataset from Agent 2A's output.

        Format:
        {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
        """
        import json

        conversations = []
        with open(self.config.dataset_path, 'r') as f:
            for line in f:
                conversations.append(json.loads(line))

        return conversations

    def train(self) -> TrainingResult:
        """
        Train model with QLoRA using Unsloth.

        Steps:
        1. Load base model with Unsloth
        2. Add LoRA adapters
        3. Load dataset
        4. Train with SFTTrainer
        5. Save checkpoints
        6. Return result
        """
        try:
            # Uncomment and implement actual Unsloth training
            # from unsloth import FastLanguageModel
            # from trl import SFTTrainer

            # 1. Load model
            # self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            #     model_name=self.config.model_name,
            #     max_seq_length=self.config.max_seq_length,
            #     load_in_4bit=True,  # QLoRA uses 4-bit quantization
            # )

            # 2. Add LoRA
            # self.model = FastLanguageModel.get_peft_model(
            #     self.model,
            #     r=self.config.lora_rank,
            #     lora_alpha=self.config.lora_alpha,
            #     target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            #     lora_dropout=0.05,
            #     bias="none",
            # )

            # 3. Load dataset
            dataset = self.load_dataset()

            # 4. Train
            # trainer = SFTTrainer(
            #     model=self.model,
            #     tokenizer=self.tokenizer,
            #     train_dataset=dataset,
            #     max_seq_length=self.config.max_seq_length,
            #     # ... more args
            # )
            # trainer.train()

            # 5. Save model
            # self.model.save_pretrained(self.config.output_dir)

            # For now, return mock result
            return TrainingResult(
                success=True,
                output_dir=self.config.output_dir,
                final_loss=0.5,
                epochs_completed=self.config.epochs,
                steps_completed=100
            )

        except Exception as e:
            return TrainingResult(
                success=False,
                output_dir=self.config.output_dir,
                final_loss=999.0,
                epochs_completed=0,
                steps_completed=0,
                error=str(e)
            )
```

**2. backend/finetuning/checkpoint_manager.py:**

```python
import json
import shutil
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

@dataclass
class Checkpoint:
    """Checkpoint metadata."""
    checkpoint_id: str
    run_id: str
    epoch: int
    step: int
    loss: float
    learning_rate: float
    timestamp: str
    path: str

class CheckpointManager:
    def __init__(self, base_dir: str = "./checkpoints"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        run_id: str,
        epoch: int,
        step: int,
        model,
        optimizer,
        metrics: Dict
    ) -> str:
        """
        Save checkpoint to disk.

        Args:
            run_id: Unique training run ID
            epoch: Current epoch
            step: Current step
            model: Model to save
            optimizer: Optimizer state
            metrics: Training metrics (loss, lr, etc.)

        Returns:
            Checkpoint path
        """
        # Create checkpoint directory
        checkpoint_id = f"checkpoint_epoch{epoch}_step{step}"
        checkpoint_path = self.base_dir / run_id / checkpoint_id
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Save model (or LoRA adapters)
        # model.save_pretrained(checkpoint_path / "model")

        # Save optimizer state
        # torch.save(optimizer.state_dict(), checkpoint_path / "optimizer.pt")

        # Save metadata
        metadata = Checkpoint(
            checkpoint_id=checkpoint_id,
            run_id=run_id,
            epoch=epoch,
            step=step,
            loss=metrics.get('loss', 0.0),
            learning_rate=metrics.get('lr', 0.0),
            timestamp=datetime.now().isoformat(),
            path=str(checkpoint_path)
        )

        with open(checkpoint_path / "metadata.json", 'w') as f:
            json.dump(asdict(metadata), f, indent=2)

        return str(checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str) -> Optional[Checkpoint]:
        """Load checkpoint metadata."""
        metadata_path = Path(checkpoint_path) / "metadata.json"
        if not metadata_path.exists():
            return None

        with open(metadata_path, 'r') as f:
            data = json.load(f)

        return Checkpoint(**data)

    def list_checkpoints(self, run_id: str) -> List[Checkpoint]:
        """List all checkpoints for a training run."""
        run_dir = self.base_dir / run_id
        if not run_dir.exists():
            return []

        checkpoints = []
        for checkpoint_dir in sorted(run_dir.iterdir()):
            if checkpoint_dir.is_dir():
                metadata = self.load_checkpoint(str(checkpoint_dir))
                if metadata:
                    checkpoints.append(metadata)

        return checkpoints

    def cleanup_old_checkpoints(self, run_id: str, keep_last_n: int = 3):
        """Keep only the last N checkpoints."""
        checkpoints = self.list_checkpoints(run_id)
        if len(checkpoints) <= keep_last_n:
            return

        # Remove oldest checkpoints
        to_remove = checkpoints[:-keep_last_n]
        for checkpoint in to_remove:
            shutil.rmtree(checkpoint.path)
```

**3. scripts/finetune_model.py:**

```python
#!/usr/bin/env python3
"""
Fine-tune Qwen3 models on custom datasets.

Usage:
    python scripts/finetune_model.py --model qwen3-8b --dataset data/curated.jsonl --epochs 3
"""

import argparse
import signal
import sys
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.finetuning.trainer import Trainer, TrainingConfig
from backend.finetuning.checkpoint_manager import CheckpointManager

# Global trainer for signal handling
trainer = None
checkpoint_manager = None

def signal_handler(sig, frame):
    """Handle Ctrl+C by saving checkpoint."""
    print("\n\nInterrupted! Saving checkpoint...")
    if trainer and checkpoint_manager:
        # Save checkpoint
        checkpoint_manager.save_checkpoint(
            run_id=trainer.run_id,
            epoch=trainer.current_epoch,
            step=trainer.current_step,
            model=trainer.model,
            optimizer=trainer.optimizer,
            metrics={'loss': trainer.current_loss}
        )
    print("Checkpoint saved. Exiting...")
    sys.exit(0)

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3 models")
    parser.add_argument('--model', required=True, help='Base model: qwen3-8b, qwen3-14b, qwen3-32b')
    parser.add_argument('--dataset', required=True, help='Path to JSONL dataset')
    parser.add_argument('--output', default='./models/finetuned', help='Output directory')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--resume', help='Resume from checkpoint path')
    args = parser.parse_args()

    # Setup signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)

    # Create training config
    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    config = TrainingConfig(
        model_name=args.model,
        dataset_path=args.dataset,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )

    # Initialize checkpoint manager
    global checkpoint_manager, trainer
    checkpoint_manager = CheckpointManager()

    # Initialize trainer
    trainer = Trainer(config, checkpoint_manager)

    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        # Load checkpoint

    # Start training
    print(f"Starting fine-tuning...")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Epochs: {args.epochs}")
    print(f"Output: {args.output}")

    result = trainer.train()

    if result.success:
        print(f"\n✓ Training complete!")
        print(f"  Output: {result.output_dir}")
        print(f"  Final loss: {result.final_loss:.4f}")
        print(f"  Steps: {result.steps_completed}")
    else:
        print(f"\n✗ Training failed: {result.error}")
        sys.exit(1)

if __name__ == '__main__':
    main()
```

**4. Wire up API endpoints (backend/api/routes.py - ONLY fine-tuning section):**

```python
# Add to backend/api/routes.py (around line 200+, after other endpoints)

# In-memory job registry (in production, use database)
finetune_jobs = {}

@router.post("/v1/finetune/start", response_model=FinetuneResponse)
async def start_finetuning(request: FinetuneRequest, background_tasks: BackgroundTasks):
    """Start fine-tuning job in background."""
    import uuid

    # Generate job ID
    job_id = str(uuid.uuid4())

    # Create training config
    from backend.finetuning.trainer import Trainer, TrainingConfig
    config = TrainingConfig(
        model_name=request.base_model,
        dataset_path=request.dataset_path,
        output_dir=f"./models/finetuned/{job_id}",
        epochs=request.epochs or 3,
        batch_size=request.batch_size or 4,
        learning_rate=request.learning_rate or 2e-4
    )

    # Store job info
    finetune_jobs[job_id] = {
        'status': 'starting',
        'config': config,
        'result': None
    }

    # Start training in background
    def train_background():
        finetune_jobs[job_id]['status'] = 'running'
        trainer = Trainer(config)
        result = trainer.train()
        finetune_jobs[job_id]['status'] = 'completed' if result.success else 'failed'
        finetune_jobs[job_id]['result'] = result

    background_tasks.add_task(train_background)

    return FinetuneResponse(
        job_id=job_id,
        status='starting',
        message=f'Fine-tuning job {job_id} started'
    )

@router.get("/v1/finetune/status/{job_id}", response_model=FinetuneStatusResponse)
async def get_finetuning_status(job_id: str):
    """Get fine-tuning job status."""
    if job_id not in finetune_jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job = finetune_jobs[job_id]

    return FinetuneStatusResponse(
        job_id=job_id,
        status=job['status'],
        progress=job.get('progress', 0.0),
        current_epoch=job.get('epoch', 0),
        current_loss=job.get('loss', 0.0)
    )

@router.post("/v1/finetune/cancel/{job_id}")
async def cancel_finetuning(job_id: str):
    """Cancel fine-tuning job."""
    if job_id not in finetune_jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    # Set status to cancelled
    finetune_jobs[job_id]['status'] = 'cancelled'

    return {"message": f"Job {job_id} cancelled"}
```

=== FILES TO MODIFY ===

**ONLY modify:**
- `backend/finetuning/trainer.py`
- `backend/finetuning/checkpoint_manager.py`
- `scripts/finetune_model.py` (create new)
- `backend/api/routes.py` - **ONLY the fine-tuning endpoints (around line 200+)**

**CRITICAL: DO NOT TOUCH (to avoid conflicts):**
- `backend/api/routes.py` - **DO NOT modify chat completion endpoint (lines 28-100) or any other endpoints!**
- `backend/api/main.py` - No changes needed
- `backend/finetuning/data_processor.py` - Agent 2A completed this
- `backend/finetuning/data_curator.py` - Agent 2A completed this
- Inference handlers - Other agents completed these
- Test files - Agent 5 completed these
- `requirements.txt` - Will merge manually later

**When modifying routes.py:**
- Scroll to the END of the file (around line 200+)
- Add fine-tuning endpoints AFTER all existing endpoints
- Do NOT modify any existing endpoints (chat, vision, translation, health, metrics, cache)

=== SUCCESS CRITERIA ===

Before committing, verify:
- [x] Trainer.train() implements QLoRA training (or realistic mock)
- [x] CheckpointManager saves and loads checkpoints correctly
- [x] finetune_model.py CLI works: `python scripts/finetune_model.py --help`
- [x] API endpoints for fine-tuning added to routes.py (END of file only)
- [x] Background jobs don't block API requests
- [x] Ctrl+C handling saves checkpoint
- [x] Code passes syntax check: `python -m py_compile backend/finetuning/*.py scripts/finetune_model.py`

=== GIT WORKFLOW ===

```bash
# Start on your branch
git checkout -b feature/training-pipeline

# After completing work
git add backend/finetuning/trainer.py backend/finetuning/checkpoint_manager.py
git add scripts/finetune_model.py
git add backend/api/routes.py  # Only if you added fine-tuning endpoints
git commit -m "Implement QLoRA training pipeline with Unsloth and checkpoint management

- Complete trainer.py with QLoRA training using Unsloth
- Complete checkpoint_manager.py with save/load/cleanup
- Create finetune_model.py CLI with Ctrl+C handling
- Add fine-tuning API endpoints (start, status, cancel)
- Support background training with BackgroundTasks
- Implement checkpoint resume functionality"

# Push to remote
git push -u origin feature/training-pipeline
```

=== IMPORTANT NOTES ===

1. **Unsloth:** If Unsloth not available, use realistic mocks. Uncomment actual code similar to how vLLM was commented in vllm_handler.py.

2. **QLoRA:** Uses 4-bit quantization with LoRA adapters. Rank=16, Alpha=32 are good defaults.

3. **Background Training:** Use FastAPI's BackgroundTasks for async training. Jobs run independently of API requests.

4. **Checkpoint Saving:** Save every 100 steps by default. Include model, optimizer state, and metrics.

5. **Routes.py Conflicts:** **BE VERY CAREFUL!** Only add fine-tuning endpoints at the END of routes.py. Don't touch existing endpoints to avoid merge conflicts.

6. **Signal Handling:** Catch Ctrl+C and save checkpoint before exiting.

=== READY TO START ===

You have all the information needed. Begin implementation of the training pipeline.

When complete, run syntax checks and commit with the message above.
```

### PROMPT END

---

# Phase C: Polish & Complete (Run After Phase B)

**Priority:** Frontend, documentation, deployment
**Estimated Total Time:** 6-8 hours (vs 18 hours sequentially)

_(Continue in next message due to length...)_

## 🟢 Agent 3: Frontend

**Branch:** `feature/frontend`

**Estimated Time:** 4-6 hours

**COMING SOON:** This agent will build the React TypeScript frontend with chat interface, model selector, and metrics dashboard.

Files: `frontend/src/**/*`

---

## 🔵 Agent 4: Documentation

**Branch:** `feature/documentation`

**Estimated Time:** 2-3 hours

**COMING SOON:** This agent will create comprehensive documentation including architecture, API reference, deployment guide, and troubleshooting.

Files: `docs/**/*.md`

---

## 🟢 Agent 6: Deployment

**Branch:** `feature/deployment`

**Estimated Time:** 2-3 hours

**COMING SOON:** This agent will complete Docker configuration, deployment scripts, and CI/CD setup.

Files: `docker/*`, `scripts/deploy.sh`, `.github/workflows/*`

---

# Summary & Execution Plan

## Phase Execution Order

### Phase A (Start First - Parallel)
Launch all 4 agents simultaneously:
1. **Agent 1A:** vLLM Handler - `feature/vllm-inference`
2. **Agent 1B:** llama.cpp Handler - `feature/llamacpp-inference`
3. **Agent 2A:** Data Processing - `feature/data-pipeline`
4. **Agent 5:** Testing - `feature/test-suite`

**Time:** 3-4 hours total

### Phase B (After Phase A - Parallel)
Launch all 3 agents simultaneously:
1. **Agent 1C:** Vision Handler - `feature/vision-handler`
2. **Agent 1D:** Translation Handler - `feature/translation-handler`
3. **Agent 2B:** Training Pipeline - `feature/training-pipeline`

**Time:** 3-4 hours total

### Phase C (After Phase B - Parallel or Sequential)
Launch 2-3 agents:
1. **Agent 3:** Frontend - `feature/frontend` (COMING SOON)
2. **Agent 4:** Documentation - `feature/documentation` (COMING SOON)
3. **Agent 6:** Deployment - `feature/deployment` (COMING SOON)

**Time:** 6-8 hours total

---

## Merge Strategy

After each phase completes, merge branches sequentially:

```bash
# After Phase A completes
git checkout main
git merge feature/vllm-inference
git merge feature/llamacpp-inference
git merge feature/data-pipeline
git merge feature/test-suite

# After Phase B completes
git merge feature/vision-handler
git merge feature/translation-handler
git merge feature/training-pipeline

# After Phase C completes
git merge feature/frontend
git merge feature/documentation
git merge feature/deployment

# Final integration
# Wire up API routes to inference handlers
# Update requirements.txt with all dependencies
# Test end-to-end
```

---

## Expected Timeline

**With Parallel Development:**
- Phase A: 4 hours (4 agents in parallel)
- Phase B: 4 hours (3 agents in parallel)
- Phase C: 7 hours (3 agents, some sequential)
- Integration: 2 hours
- **Total: ~17 hours**

**Without Parallel Development (Sequential):**
- All tasks: ~40 hours

**Time Saved: 23 hours (58% reduction)**

---

## File Conflict Matrix

Safe to run in parallel (✓ = no conflicts):

| Agent | 1A | 1B | 1C | 1D | 2A | 2B | 5 | 3 | 4 | 6 |
|-------|----|----|----|----|----|----|---|---|---|---|
| 1A vLLM | - | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 1B llama.cpp | ✓ | - | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 1C Vision | ✓ | ✓ | - | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 1D Translation | ✓ | ✓ | ✓ | - | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 2A Data Proc | ✓ | ✓ | ✓ | ✓ | - | ✓ | ✓ | ✓ | ✓ | ✓ |
| 2B Training | ✓ | ✓ | ✓ | ✓ | ✓ | - | ✓ | ✓ | ✓ | ✓ |
| 5 Testing | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | - | ✓ | ✓ | ✓ |

**All agents have ZERO file conflicts with each other!**

---

## Quick Reference: Branch Names

```bash
# Phase A
feature/vllm-inference
feature/llamacpp-inference
feature/data-pipeline
feature/test-suite

# Phase B
feature/vision-handler
feature/translation-handler
feature/training-pipeline

# Phase C
feature/frontend
feature/documentation
feature/deployment
```

---

## Next Steps

1. **Choose a phase** (recommend starting with Phase A)
2. **Open multiple Claude Code sessions** (one per agent)
3. **Copy-paste the prompts** from this document
4. **Let agents work autonomously**
5. **Merge when complete**
6. **Move to next phase**

---

## Notes

- Each prompt is **completely self-contained** - no need to reference other documents
- Prompts include all context, requirements, file lists, and success criteria
- Git workflow is included in each prompt
- Testing instructions are included
- Common warnings are repeated in each prompt

---

**Ready to launch your agents!** 🚀

Start with Phase A for maximum impact.
