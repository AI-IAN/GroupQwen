# Qwen3 Local AI Orchestration System - Specification

## Project Overview

A production-grade local AI inference system that intelligently routes queries across multiple Qwen3 models (4B through 32B variants) with semantic caching, fine-tuning capabilities, and multi-device remote access via Tailscale. Mirrors Airbnb's 13-model architecture on local hardware (RTX 5090, 32GB VRAM, Ryzen 7 7800X3D).

**Target Performance**:
- 40-60% of queries served from semantic cache (<10ms)
- Remaining queries: Qwen3-8B (200ms), Qwen3-14B (400ms), or Qwen3-32B (800ms)
- Multi-device access (RTX 5090 server, M3 MacBook, iPhone via Tailscale)

---

## System Architecture

### High-Level Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend Layer                           │
│  ┌──────────────┬──────────────┬──────────────┐             │
│  │  OpenWebUI   │  Mobile Web  │  Qwen Code   │             │
│  │  (Browser)   │  (Tailscale) │  (CLI)       │             │
│  └──────────────┴──────────────┴──────────────┘             │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                  API Gateway (FastAPI)                      │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ • Request validation                                   │ │
│  │ • Semantic cache lookup (Redis)                       │ │
│  │ • Query complexity classification                     │ │
│  │ • Model routing logic                                 │ │
│  │ • Response confidence scoring                         │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│              Inference Layer (Model Orchestration)          │
│  ┌──────────────┬──────────────┬──────────────┬──────────┐  │
│  │  Qwen3-8B    │  Qwen3-14B   │  Qwen3-32B   │ Qwen3-VL │  │
│  │  Q4 / 5-6GB  │  Q4 / 10-11GB│  Q4 / 19-20GB│ 7-8GB    │  │
│  │  llama.cpp   │  vLLM        │  vLLM        │ vLLM     │  │
│  └──────────────┴──────────────┴──────────────┴──────────┘  │
│  ┌──────────────┬──────────────┬──────────────┐              │
│  │  Qwen3-4B    │  Qwen3-MT    │  Qwen3-VL-32B│              │
│  │  Q4 / 2-3GB  │  8-10GB      │  Thinking    │              │
│  │  Edge Device │  Translation │  Mode / 22GB │              │
│  └──────────────┴──────────────┴──────────────┘              │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│         Storage & Caching Layer                             │
│  ┌──────────────┬──────────────┬──────────────┐             │
│  │  Redis       │  PostgreSQL  │  Vector DB   │             │
│  │  (Semantic   │  (Query      │  (Embeddings)│             │
│  │   Cache)     │   History)   │              │             │
│  └──────────────┴──────────────┴──────────────┘             │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│         Hardware Layer                                      │
│  ┌──────────────┬──────────────┬──────────────┐             │
│  │  RTX 5090    │  CPU: Ryzen  │  32GB VRAM   │             │
│  │  (Primary)   │  7800X3D     │  64GB System │             │
│  └──────────────┴──────────────┴──────────────┘             │
└─────────────────────────────────────────────────────────────┘
```

### Device-Specific Deployments

**Primary Server (RTX 5090, 32GB VRAM)**
- Runs: Qwen3-32B, Qwen3-14B, Qwen3-8B, Qwen3-VL, Qwen3-MT
- Inference framework: vLLM + TensorRT-LLM
- API endpoint: `http://localhost:8000` (local) or `100.x.x.x:8000` (Tailscale)

**M3 MacBook Air (16GB RAM)**
- Runs: Qwen3-4B (Metal acceleration), Qwen3-8B if needed
- Inference framework: llama.cpp with Metal backend
- Use case: Offline work, edge inference, mobile-like performance testing
- Fallback to server for larger models

**iPhone via Tailscale**
- Connects to main server API via Tailscale VPN mesh
- Access OpenWebUI at `http://<server-tailscale-ip>:8080`
- Latency: 50-200ms depending on network conditions
- No local model storage (bandwidth-prohibitive)

---

## Project Structure

```
qwen3-local-system/
├── docs/
│   ├── DEPLOYMENT.md
│   ├── ARCHITECTURE.md
│   ├── API_REFERENCE.md
│   └── TROUBLESHOOTING.md
├── backend/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py                    # FastAPI app entry point
│   │   ├── routes.py                  # API endpoints
│   │   ├── websocket_handler.py       # Real-time streaming
│   │   └── models.py                  # Pydantic schemas
│   ├── core/
│   │   ├── __init__.py
│   │   ├── router.py                  # Query routing logic
│   │   ├── cache_manager.py           # Semantic cache (Redis)
│   │   ├── complexity_classifier.py   # Query complexity scoring
│   │   └── confidence_scorer.py       # Response confidence estimation
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── model_loader.py            # Load/unload models
│   │   ├── vllm_handler.py            # vLLM inference
│   │   ├── llamacpp_handler.py        # llama.cpp inference
│   │   ├── vision_handler.py          # Qwen3-VL processing
│   │   └── translation_handler.py     # Qwen3-MT processing
│   ├── finetuning/
│   │   ├── __init__.py
│   │   ├── trainer.py                 # Unsloth QLoRA training
│   │   ├── data_processor.py          # Dataset preparation
│   │   ├── data_curator.py            # Chat export processing
│   │   └── checkpoint_manager.py      # Model checkpoint handling
│   ├── monitoring/
│   │   ├── __init__.py
│   │   ├── metrics.py                 # Cache hit rates, latency
│   │   ├── logger.py                  # Query logging
│   │   └── health_check.py            # System health endpoint
│   ├── config/
│   │   ├── __init__.py
│   │   ├── settings.py                # Environment variables
│   │   ├── model_config.yaml          # Model specifications
│   │   └── routing_rules.yaml         # Routing thresholds
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── redis_client.py            # Redis wrapper
│   │   ├── embeddings.py              # Sentence transformer wrapper
│   │   └── device_utils.py            # GPU/CPU detection
│   └── requirements.txt
├── frontend/
│   ├── public/
│   │   └── index.html
│   ├── src/
│   │   ├── components/
│   │   │   ├── ChatInterface.tsx
│   │   │   ├── ModelSelector.tsx
│   │   │   ├── SystemMetrics.tsx
│   │   │   └── FileUpload.tsx
│   │   ├── services/
│   │   │   ├── api.ts
│   │   │   ├── websocket.ts
│   │   │   └── storage.ts
│   │   ├── App.tsx
│   │   └── index.css
│   ├── package.json
│   └── tsconfig.json
├── docker/
│   ├── Dockerfile.backend
│   ├── Dockerfile.frontend
│   ├── Dockerfile.redis
│   └── docker-compose.yml
├── scripts/
│   ├── setup.sh                       # Initial setup script
│   ├── download_models.sh             # Model downloading
│   ├── export_chat_history.py         # Export from ChatGPT/Claude/Perplexity
│   ├── curate_dataset.py              # Interactive data curation
│   ├── finetune_model.py              # Start fine-tuning job
│   ├── benchmark.py                   # Performance testing
│   └── deploy.sh                      # Production deployment
├── tests/
│   ├── test_router.py
│   ├── test_cache.py
│   ├── test_inference.py
│   └── test_api.py
├── .env.example
├── docker-compose.yml
├── README.md
└── QUICK_START.md
```

---

## Core Specifications

### 1. Query Routing Engine (`core/router.py`)

**Purpose**: Intelligently route queries to optimal models based on complexity, cache availability, and performance requirements.

**Input**: User query, optional context, device capability flags
**Output**: Selected model, parameters, reasoning trace

**Algorithm**:
```python
def route_query(query: str, context: str = "", device: str = "server") -> RouteDecision:
    """
    Cascade routing strategy combining model selection and escalation.
    Returns optimal model for given query characteristics.
    
    Process:
    1. Check semantic cache (40-60% hit rate target)
    2. Classify query complexity (0.0-1.0 scale)
    3. Estimate required context length
    4. Select model from Pareto frontier
    5. Set confidence threshold for escalation
    """
    
    # Step 1: Cache lookup
    cache_result = semantic_cache.check(query)
    if cache_result and cache_result['confidence'] > 0.85:
        return RouteDecision(
            model="cache",
            latency_ms=10,
            cost_relative=0.001
        )
    
    # Step 2: Complexity classification
    complexity_score = complexity_classifier.score(
        query=query,
        context=context,
        features=['length', 'keywords', 'reasoning_required']
    )
    
    context_length = estimate_tokens(query + context)
    
    # Step 3: Model selection via Pareto frontier
    if device == "mobile" or device == "macbook":
        # Edge device: use 4B or 8B only
        if complexity_score < 0.3:
            return RouteDecision(model="qwen3-4b-gguf", ...)
        else:
            return RouteDecision(model="qwen3-8b-gguf", ...)
    
    # Server device: full routing
    if complexity_score < 0.2:
        return RouteDecision(model="qwen3-8b-q4", ...)
    elif complexity_score < 0.5:
        return RouteDecision(model="qwen3-14b-q4", ...)
    elif "explain" in query.lower() or requires_reasoning(query):
        return RouteDecision(
            model="qwen3-32b-thinking",
            use_thinking_mode=True,
            ...
        )
    else:
        return RouteDecision(model="qwen3-32b-q4", ...)
```

**Routing Rules Configuration** (`config/routing_rules.yaml`):
```yaml
routing:
  cache_hit_threshold: 0.10  # Cosine similarity
  cache_ttl_seconds: 86400   # 24 hours
  complexity_thresholds:
    simple: 0.2
    moderate: 0.5
    complex: 0.8
  
  model_selection:
    qwen3_4b:
      max_complexity: 0.2
      max_context_tokens: 2048
      target_latency_ms: 100
      cost_relative: 1.0
    
    qwen3_8b:
      max_complexity: 0.5
      max_context_tokens: 4096
      target_latency_ms: 200
      cost_relative: 1.5
    
    qwen3_14b:
      max_complexity: 0.8
      max_context_tokens: 8192
      target_latency_ms: 400
      cost_relative: 2.2
    
    qwen3_32b:
      max_complexity: 1.0
      max_context_tokens: 131072
      target_latency_ms: 800
      cost_relative: 3.5
      
    qwen3_32b_thinking:
      use_case: "reasoning_required"
      thinking_budget_tokens: 5000
      cost_relative: 4.0

escalation:
  low_confidence_threshold: 0.4
  escalate_to_larger_model: true
  confidence_scoring_method: "token_logits"
```

### 2. Semantic Cache Layer (`core/cache_manager.py`)

**Technology**: Redis + RedisVL + sentence-transformers

**Purpose**: Cache responses using semantic similarity, achieving 40-60% query reduction.

```python
class SemanticCacheManager:
    """
    Redis-based semantic caching using embeddings.
    
    Attributes:
        distance_threshold: Cosine similarity threshold (default 0.10)
        ttl_seconds: Time-to-live for cache entries (default 86400 = 24h)
        embedding_model: sentence-transformers model for embeddings
    """
    
    def __init__(self, redis_url: str, embedding_model: str):
        self.client = redis.from_url(redis_url)
        self.vectorizer = HFTextVectorizer(embedding_model)
        self.ns = VectorStoreIndex(self.client, name="qwen_cache")
    
    def check(self, prompt: str, threshold: float = 0.10) -> Optional[CacheHit]:
        """
        Check cache for semantically similar queries.
        Returns cached response if similarity > threshold.
        """
        embedding = self.vectorizer.embed(prompt)
        results = self.ns.search(embedding, k=1, distance_threshold=threshold)
        
        if results:
            return CacheHit(
                response=results[0]['response'],
                similarity=results[0]['distance'],
                cached_at=results[0]['timestamp'],
                hit_count=results[0]['hit_count'] + 1
            )
        return None
    
    def store(self, prompt: str, response: str, metadata: dict = None):
        """Store query-response pair with metadata."""
        embedding = self.vectorizer.embed(prompt)
        self.ns.add(
            {
                "prompt": prompt,
                "response": response,
                "embedding": embedding,
                "timestamp": datetime.now().isoformat(),
                "hit_count": 0,
                **metadata
            },
            ttl=self.ttl_seconds
        )
    
    def get_stats(self) -> CacheStats:
        """Return cache performance statistics."""
        # Implementation returns: hit_rate, avg_similarity, total_cached, etc.
```

**Cache Configuration** (`config/settings.py`):
```python
CACHE_CONFIG = {
    "redis_url": "redis://localhost:6379",
    "embedding_model": "redis/langcache-embed-v1",  # Optimized for semantic cache
    "distance_threshold": 0.10,  # Cosine similarity
    "ttl_general": 86400,        # 24 hours
    "ttl_timesensitive": 3600,   # 1 hour for news/updates
    "max_cache_size_mb": 10000,
}
```

### 3. Complexity Classification (`core/complexity_classifier.py`)

**Purpose**: Score query complexity (0.0-1.0) to determine which model to use.

```python
class ComplexityClassifier:
    """
    Estimates query complexity using multiple signals:
    - Query length and structure
    - Presence of reasoning keywords
    - Required context depth
    - Task type inference
    """
    
    def __init__(self):
        self.keywords_simple = ["what", "how", "list", "find", "search"]
        self.keywords_complex = ["explain", "analyze", "design", "optimize", "debate"]
        self.keywords_reasoning = ["why", "compare", "pros cons", "implications"]
    
    def score(self, query: str, context: str = "") -> float:
        """
        Compute complexity score (0.0-1.0).
        
        Factors:
        - Query length (longer = more complex)
        - Context length (deeper context = more complex)
        - Keyword presence (reasoning words increase complexity)
        - Estimated output length
        - Required reasoning steps
        """
        
        # Normalized factors (0.0-1.0)
        length_factor = min(len(query) / 200, 1.0)
        context_factor = min(len(context) / 1000, 1.0)
        
        keyword_factor = 0.0
        if any(kw in query.lower() for kw in self.keywords_complex):
            keyword_factor += 0.4
        if any(kw in query.lower() for kw in self.keywords_reasoning):
            keyword_factor += 0.3
        
        # Aggregate with weights
        complexity = (
            0.2 * length_factor +
            0.2 * context_factor +
            0.4 * keyword_factor +
            0.2 * self._estimate_reasoning_steps(query)
        )
        
        return min(complexity, 1.0)
    
    def _estimate_reasoning_steps(self, query: str) -> float:
        """Estimate number of reasoning steps required."""
        # Simple heuristic: count conjunctions and complex structures
        reasoning_indicators = query.count(" and ") + query.count(" or ") + query.count(",")
        return min(reasoning_indicators / 5, 1.0)
```

### 4. Model Configuration (`config/model_config.yaml`)

```yaml
models:
  qwen3_4b:
    name: "Qwen/Qwen3-4B-AWQ"
    quantization: "awq"
    vram_gb: 3
    framework: "llamacpp"
    max_tokens: 2048
    context_window: 32768
    use_case: "edge_device"
    deployment_targets:
      - "macbook"
      - "mobile"
    
  qwen3_8b:
    name: "Qwen/Qwen3-8B-AWQ"
    quantization: "awq"
    vram_gb: 6
    framework: "llamacpp"
    max_tokens: 4096
    context_window: 131072
    use_case: "fast_balanced"
    deployment_targets:
      - "server"
      - "macbook"
  
  qwen3_14b:
    name: "Qwen/Qwen3-14B-AWQ"
    quantization: "awq"
    vram_gb: 11
    framework: "vllm"
    max_tokens: 8192
    context_window: 131072
    use_case: "balanced"
    deployment_targets:
      - "server"
    vllm_config:
      gpu_memory_utilization: 0.90
      max_model_len: 8192
  
  qwen3_32b:
    name: "Qwen/Qwen3-32B-AWQ"
    quantization: "awq"
    vram_gb: 20
    framework: "vllm"
    max_tokens: 16384
    context_window: 131072
    use_case: "reasoning"
    deployment_targets:
      - "server"
    vllm_config:
      gpu_memory_utilization: 0.95
      max_model_len: 16384
      rope_scaling:
        type: "yarn"
        factor: 4.0
  
  qwen3_32b_thinking:
    name: "Qwen/Qwen3-32B-Thinking"
    quantization: "awq"
    vram_gb: 22
    framework: "vllm"
    max_tokens: 16384
    context_window: 131072
    use_case: "advanced_reasoning"
    deployment_targets:
      - "server"
    thinking_config:
      thinking_budget: 5000
      thinking_mode: true
  
  qwen3_vl:
    name: "Qwen/Qwen3-VL-7B-AWQ"
    quantization: "awq"
    vram_gb: 9
    framework: "vllm"
    max_tokens: 4096
    use_case: "vision"
    deployment_targets:
      - "server"
    capabilities:
      - "image_understanding"
      - "gui_automation"
      - "screenshot_analysis"
      - "ocr"
  
  qwen3_mt:
    name: "Qwen/Qwen3-32B-MT"
    quantization: "q4"
    vram_gb: 10
    framework: "vllm"
    max_tokens: 2048
    use_case: "translation"
    deployment_targets:
      - "server"
    languages_supported: 92
```

### 5. Fine-Tuning Pipeline (`finetuning/trainer.py`)

**Purpose**: Fine-tune Qwen3 models on curated chat history using QLoRA.

```python
class Qwen3Trainer:
    """
    QLoRA fine-tuning for Qwen3 models using Unsloth.
    
    Workflow:
    1. Load model in 4-bit
    2. Attach LoRA adapters (low-rank fine-tuning)
    3. Train on curated dataset
    4. Merge LoRA weights
    5. Quantize for production
    """
    
    def __init__(self, model_name: str, lora_rank: int = 128):
        self.model_name = model_name
        self.lora_rank = lora_rank
    
    def load_model(self):
        """Load model in 4-bit with LoRA adapters."""
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=2048,
            load_in_4bit=True,  # QLoRA quantization
            device_map="auto",
        )
        
        model = FastLanguageModel.get_peft_model(
            model,
            r=self.lora_rank,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        return model, tokenizer
    
    def train(self, dataset_path: str, output_dir: str, epochs: int = 3):
        """
        Fine-tune model on dataset.
        
        Args:
            dataset_path: Path to instruction-tuning dataset (JSON lines)
            output_dir: Directory for checkpoints
            epochs: Training epochs
        
        Dataset format:
            {"instruction": "...", "input": "...", "output": "..."}
        """
        model, tokenizer = self.load_model()
        
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=load_dataset("json", data_files=dataset_path),
            dataset_text_field="text",
            max_seq_length=2048,
            dataset_num_proc=4,
            packing=False,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,  # Effective batch = 8
            warmup_steps=100,
            num_train_epochs=epochs,
            learning_rate=2e-4,
            fp16=not torch.cuda.is_available(),
            bf16=torch.cuda.is_available(),
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=42,
            output_dir=output_dir,
            report_to="none",
            save_strategy="epoch",
        )
        
        trainer.train()
        return trainer
    
    def merge_and_quantize(self, checkpoint_dir: str, output_path: str):
        """Merge LoRA weights and quantize for deployment."""
        # Merge: adapter weights + base model
        # Quantize: 4-bit AWQ for production
```

### 6. Data Export & Curation (`scripts/export_chat_history.py`, `finetuning/data_curator.py`)

**Purpose**: Export chat history from multiple sources and curate into training dataset.

```python
def export_chatgpt_history(export_zip_path: str) -> List[Dict]:
    """
    Export ChatGPT conversations from official export ZIP.
    
    Process:
    1. Extract conversations.json from ZIP
    2. Parse conversation tree structure
    3. Filter by quality metrics (length, no errors, etc.)
    4. Convert to instruction-tuning format
    """
    conversations = []
    with zipfile.ZipFile(export_zip_path) as z:
        with z.open("conversations.json") as f:
            data = json.load(f)
    
    for conv in data:
        for node_id, node in conv["mapping"].items():
            if node["message"] and node["message"]["content"]["parts"]:
                role = node["message"]["author"]["role"]
                content = node["message"]["content"]["parts"][0]
                conversations.append({
                    "role": role,
                    "content": content,
                    "timestamp": node["message"].get("create_time")
                })
    
    return conversations

class DataCurator:
    """
    Interactive data curation tool.
    
    Features:
    - Filter by quality metrics (length, coherence, domain)
    - Remove PII/sensitive information
    - Format for instruction tuning
    - Track curator decisions for reproducibility
    """
    
    def __init__(self):
        self.rejected_patterns = []
        self.accepted_count = 0
    
    def curate_conversations(self, raw_conversations: List[Dict]) -> List[Dict]:
        """
        Interactively curate conversations.
        
        Filters:
        - Minimum length: 50 characters
        - Maximum length: 2000 characters (for efficiency)
        - Remove off-topic exchanges
        - Remove hallucinated responses
        - Remove extremely short Q&A pairs
        """
        curated = []
        
        for i, conv in enumerate(raw_conversations):
            print(f"\n--- Conversation {i+1}/{len(raw_conversations)} ---")
            print(f"Role: {conv['role']}")
            print(f"Content: {conv['content'][:200]}...")
            
            # Automatic quality checks
            if len(conv['content']) < 50:
                print("❌ Too short")
                continue
            
            if "I don't have" in conv['content'] or "I'm not sure" in conv['content']:
                print("⚠️ Low confidence response")
                user_input = input("Accept anyway? (y/n): ")
                if user_input.lower() != 'y':
                    continue
            
            # Manual acceptance
            user_input = input("Accept this? (y/n): ")
            if user_input.lower() == 'y':
                curated.append(conv)
                self.accepted_count += 1
            else:
                self.rejected_patterns.append(conv)
        
        return curated
    
    def format_for_training(self, curated_conversations: List[Dict]) -> List[Dict]:
        """Convert to instruction-tuning format."""
        training_data = []
        
        for conv in curated_conversations:
            training_data.append({
                "instruction": conv["instruction"],
                "input": conv.get("input", ""),
                "output": conv["output"]
            })
        
        return training_data
```

### 7. API Endpoints (`api/routes.py`)

```python
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible chat endpoint.
    
    Request:
        - messages: List of messages
        - model: (optional) Model to use
        - temperature: Sampling temperature
        - max_tokens: Max response length
    
    Returns:
        - choices: List of responses
        - usage: Token counts
        - metadata: Routing info, cache hit status
    """
    # Route query
    route = router.route_query(
        query=request.messages[-1]["content"],
        device=request.device or "server"
    )
    
    # Call appropriate model
    response = await call_model(
        model_name=route.model,
        messages=request.messages,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
    )
    
    # Cache response
    cache_manager.store(
        prompt=request.messages[-1]["content"],
        response=response["content"],
        metadata={
            "model": route.model,
            "cache_hit": route.model == "cache"
        }
    )
    
    return {
        "choices": [{"message": {"content": response["content"]}}],
        "usage": response["usage"],
        "metadata": {
            "model_used": route.model,
            "cache_hit": route.model == "cache",
            "latency_ms": response["latency_ms"]
        }
    }

@app.post("/v1/finetune/start")
async def start_finetuning(request: FinetuneRequest):
    """
    Start fine-tuning job on specified model.
    
    Request:
        - base_model: Model to fine-tune
        - dataset_path: Path to training data
        - epochs: Training epochs
        - lora_rank: LoRA rank
    """
    job = FinetuningJob(
        job_id=str(uuid.uuid4()),
        base_model=request.base_model,
        dataset_path=request.dataset_path,
        status="queued"
    )
    
    # Start in background
    asyncio.create_task(trainer.train(...))
    
    return {"job_id": job.job_id, "status": "queued"}

@app.get("/v1/models")
async def list_models():
    """List available models and their status."""
    return {
        "models": [
            {"name": "qwen3-32b", "status": "running", "vram_used": 20},
            {"name": "qwen3-14b", "status": "running", "vram_used": 11},
            {"name": "qwen3-8b", "status": "idle", "vram_used": 0},
        ]
    }

@app.get("/v1/cache/stats")
async def cache_stats():
    """Get cache performance statistics."""
    stats = cache_manager.get_stats()
    return {
        "hit_rate": stats.hit_rate,
        "total_cached": stats.total_cached,
        "avg_similarity": stats.avg_similarity,
        "cache_size_mb": stats.cache_size_mb
    }

@app.post("/v1/vision/analyze")
async def analyze_image(request: VisionRequest):
    """
    Analyze screenshot/image with Qwen3-VL.
    
    Use cases:
    - GUI automation (find button, menu location)
    - Screenshot understanding
    - OCR
    """
    response = vision_handler.analyze(
        image=request.image,
        prompt=request.prompt,
        return_bounding_boxes=request.return_bboxes
    )
    return response
```

### 8. Monitoring & Metrics (`monitoring/metrics.py`)

```python
class MetricsCollector:
    """
    Track system performance metrics for continuous optimization.
    
    Metrics:
    - Cache hit rate (target: 40-60%)
    - Model utilization by tier
    - Average latency by model
    - User satisfaction (implicit from query patterns)
    - Cost per query (relative units)
    """
    
    def __init__(self):
        self.db = PostgreSQL(dsn="postgresql://user:pass@localhost/metrics")
    
    def log_query(self, query: QueryLog):
        """Log query for analysis."""
        self.db.insert("query_logs", {
            "timestamp": datetime.now(),
            "query": query.text[:500],  # Truncate for storage
            "model_used": query.model,
            "latency_ms": query.latency_ms,
            "cache_hit": query.cache_hit,
            "user_id": query.user_id,
            "success": query.success
        })
    
    def get_daily_stats(self) -> DailyStats:
        """Aggregate statistics for the day."""
        results = self.db.query("""
            SELECT
                model_used,
                COUNT(*) as query_count,
                AVG(latency_ms) as avg_latency,
                SUM(CASE WHEN cache_hit THEN 1 ELSE 0 END)::FLOAT / COUNT(*) as cache_hit_rate,
                SUM(CASE WHEN success THEN 1 ELSE 0 END)::FLOAT / COUNT(*) as success_rate
            FROM query_logs
            WHERE date(timestamp) = CURRENT_DATE
            GROUP BY model_used
        """)
        return results
```

---

## Technology Stack

### Backend
- **Framework**: FastAPI (async, WebSocket support)
- **Inference**: vLLM + TensorRT-LLM (GPU inference), llama.cpp (CPU/edge)
- **LLM Training**: Unsloth (QLoRA) + Hugging Face Transformers
- **Caching**: Redis + RedisVL (semantic cache)
- **Database**: PostgreSQL (query logs, metrics)
- **Async**: Celery for background jobs (fine-tuning, model downloads)
- **Python Version**: 3.10+

### Frontend
- **Web UI**: React TypeScript + Tailwind CSS
- **Chat Interface**: OpenWebUI (or custom React component)
- **Mobile**: Browser-based via Tailscale (no native app needed)
- **State Management**: React Query (API caching) + Zustand (local state)

### Infrastructure
- **Containerization**: Docker + Docker Compose
- **Remote Access**: Tailscale VPN mesh
- **GPU Drivers**: CUDA 12.4+, cuDNN 8.8+
- **Quantization**: AutoAWQ (for AWQ models), GPTQ-for-LLaMA

---

## Implementation Phases

### Phase 1: Core Foundation
- [ ] Setup project structure and dependencies
- [ ] Configure vLLM for Qwen3-32B, Qwen3-14B, Qwen3-8B
- [ ] Implement basic routing logic (non-semantic)
- [ ] Setup Redis semantic cache
- [ ] Create FastAPI endpoints for `/chat/completions`
- [ ] Deploy with OpenWebUI

**Milestone**: Can chat with models locally, basic caching works

### Phase 2: Intelligent Routing
- [ ] Implement complexity classifier
- [ ] Implement Pareto-optimal routing
- [ ] Add semantic caching with embeddings
- [ ] Add confidence scoring and escalation
- [ ] Setup monitoring/metrics database

**Milestone**: System routes queries intelligently, cache hit rate > 30%

### Phase 3: Specialized Capabilities
- [ ] Deploy Qwen3-VL for vision tasks
- [ ] Deploy Qwen3-MT for translation
- [ ] Implement vision automation (screenshot analysis)
- [ ] Test multilingual support

**Milestone**: Can analyze images, translate text, handle multimodal queries

### Phase 4: Fine-Tuning Pipeline
- [ ] Export chat history from ChatGPT/Claude/Perplexity
- [ ] Build data curation UI
- [ ] Implement QLoRA training pipeline
- [ ] Setup checkpoint management
- [ ] Test fine-tuned models

**Milestone**: Can fine-tune custom models on your data, improved domain performance

### Phase 5: Remote Access & Edge Deployment
- [ ] Configure Tailscale for secure remote access
- [ ] Deploy M3 MacBook llama.cpp inference
- [ ] Test iPhone access via Tailscale VPN
- [ ] Setup device-specific model routing
- [ ] Optimize for mobile latency

**Milestone**: Can access system from anywhere, edge devices work

### Phase 6: Production Hardening
- [ ] Load testing and performance optimization
- [ ] Error handling and recovery
- [ ] Logging and debugging infrastructure
- [ ] Documentation and runbooks
- [ ] Cost tracking and optimization

**Milestone**: Production-ready system with monitoring, documentation

---

## Deployment Instructions

### Prerequisites
```bash
# Check NVIDIA GPU
nvidia-smi

# Check system resources
free -h
lsblk
```

### Initial Setup
```bash
# Clone repository
git clone https://github.com/yourusername/qwen3-local-system
cd qwen3-local-system

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r backend/requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your settings

# Download models (15-30 minutes)
bash scripts/download_models.sh

# Start Redis
docker run -d -p 6379:6379 redis:latest

# Start system
python -m backend.api.main
```

### Production Deployment
```bash
# Build Docker images
docker-compose build

# Start all services
docker-compose up -d

# Verify services
docker-compose ps
docker-compose logs -f backend

# Access at http://localhost:8080 (OpenWebUI)
# API at http://localhost:8000
```

### Tailscale Setup
```bash
# Install Tailscale on server
curl -fsSL https://tailscale.com/install.sh | sh

# Authenticate
sudo tailscale up

# Get Tailnet IP
tailscale ip -4

# On client devices, install Tailscale and connect to same Tailnet
# Then access: http://<server-tailnet-ip>:8080
```

---

## Configuration Files

### `.env` Template
```
# Model Configuration
PRIMARY_MODEL=qwen3-32b-awq
SECONDARY_MODEL=qwen3-14b-awq
CACHE_MODEL=qwen3-8b-q4

# Hardware
GPU_MEMORY_FRACTION=0.95
NUM_GPU_LAYERS=32
BATCH_SIZE=1

# Redis
REDIS_URL=redis://localhost:6379
CACHE_TTL=86400

# Fine-tuning
LORA_RANK=128
LORA_ALPHA=32
TRAINING_BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=4

# API
API_PORT=8000
API_HOST=0.0.0.0
MAX_TOKENS=4096

# Monitoring
LOG_LEVEL=INFO
METRICS_DB_URL=postgresql://user:pass@localhost/metrics

# Device
DEVICE_TYPE=server  # server | macbook | mobile
TAILSCALE_IP=100.0.0.1
```

---

## Success Criteria

### Performance Targets
- [ ] Cache hit rate: 40-60%
- [ ] Average latency (cached): <20ms
- [ ] Average latency (Qwen3-8B): <300ms
- [ ] Average latency (Qwen3-32B): <1000ms
- [ ] GPU memory utilization: 90-95%
- [ ] System uptime: >99.5%

### Feature Completeness
- [ ] Multi-model routing works correctly
- [ ] Semantic caching reduces API calls by 50%+
- [ ] Fine-tuning pipeline works end-to-end
- [ ] Vision analysis handles screenshots
- [ ] Translation works across 50+ languages
- [ ] Mobile access via Tailscale is latency-acceptable (<200ms)

### Code Quality
- [ ] 80%+ test coverage
- [ ] All endpoints documented
- [ ] Error handling implemented
- [ ] Logging comprehensive
- [ ] Modular, maintainable architecture

---

## Support & Next Steps

- [ ] Read full architecture docs in `docs/ARCHITECTURE.md`
- [ ] Follow quick start in `QUICK_START.md`
- [ ] Join Discord community for questions
- [ ] Review benchmark results in `docs/BENCHMARKS.md`
- [ ] Setup monitoring dashboard
