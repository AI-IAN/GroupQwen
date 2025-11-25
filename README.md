# Local Multi-Model AI Orchestration System

A production-grade local AI inference system that intelligently routes queries across multiple model families (Qwen3, OLMo3, and more) with semantic caching, fine-tuning capabilities, and multi-device remote access. Supports models from 4B to 70B+ parameters with automatic complexity-based routing.

## Features

- **Intelligent Query Routing**: Automatically routes queries to optimal models based on complexity
- **Semantic Caching**: 40-60% of queries served from cache with <10ms latency
- **Multi-Model Family Support**:
  - **Qwen3:** 4B, 8B, 14B, 32B, 32B-Thinking, VL (vision), MT (translation)
  - **OLMo3:** 7B, 7B-Instruct, 32B, 32B-Instruct (newly added)
  - Extensible architecture for adding new model families
- **Fine-Tuning Pipeline**: QLoRA-based fine-tuning on custom datasets
- **Multi-Device Access**: Server, MacBook, and mobile access via Tailscale
- **Vision Capabilities**: Screenshot analysis, OCR, GUI automation with Qwen3-VL
- **Translation**: 92-language support with Qwen3-MT
- **Production-Ready**: Docker deployment, monitoring, health checks, metrics

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend Layer                           │
│  ┌──────────────┬──────────────┬──────────────┐             │
│  │  OpenWebUI   │  Mobile Web  │  Custom UI   │             │
│  └──────────────┴──────────────┴──────────────┘             │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                  API Gateway (FastAPI)                      │
│  • Semantic cache lookup • Query routing                   │
│  • Complexity classification • Model orchestration         │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│              Inference Layer (Multi-Model)                  │
│  Qwen3: 4B • 8B • 14B • 32B • VL • MT                      │
│  OLMo3: 7B • 7B-Instruct • 32B • 32B-Instruct              │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

**For detailed installation instructions, see [GETTING_STARTED.md](GETTING_STARTED.md)**

### Prerequisites

- Python 3.10+
- NVIDIA GPU (8GB+ VRAM recommended) or Apple Silicon Mac
- Redis server
- 16GB+ system RAM

### Installation

```bash
# 1. Clone and setup
git clone https://github.com/yourusername/GroupQwen
cd GroupQwen
./scripts/setup.sh

# 2. Configure environment
nano .env

# 3. Start Redis
redis-server  # or: sudo systemctl start redis-server

# 4. Download models (optional for testing)
./scripts/download_models.sh

# 5. Start API server
source venv/bin/activate
python -m backend.api.main
```

The API will be available at `http://localhost:8000`

**See [GETTING_STARTED.md](GETTING_STARTED.md) for detailed setup and troubleshooting.**

### Docker Deployment

```bash
docker-compose up -d
```

## Usage

### Chat Completion

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Explain quantum computing"}
    ],
    "temperature": 0.7
  }'
```

### Vision Analysis

```bash
curl -X POST http://localhost:8000/v1/vision/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "image": "<base64_image>",
    "prompt": "Describe this image",
    "return_bboxes": false
  }'
```

### Translation

```bash
curl -X POST http://localhost:8000/v1/translate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, world!",
    "target_lang": "es"
  }'
```

## API Documentation

Interactive API documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Project Structure

```
GroupQwen/
├── backend/
│   ├── api/              # FastAPI application
│   ├── core/             # Routing, caching, classification
│   ├── inference/        # Model handlers (vLLM, llama.cpp)
│   ├── finetuning/       # Fine-tuning pipeline
│   ├── monitoring/       # Metrics, logging, health checks
│   ├── config/           # Configuration files
│   └── utils/            # Utilities
├── frontend/             # React TypeScript UI
├── docker/               # Docker configurations
├── scripts/              # Utility scripts
├── docs/                 # Documentation
└── tests/                # Test files
```

## Performance Targets

- **Cache Hit Rate**: 40-60%
- **Cached Query Latency**: <20ms
- **Small Models (4B-8B)**: <300ms
- **Medium Models (14B)**: <500ms
- **Large Models (32B)**: <1000ms
- **GPU Utilization**: 90-95%

**Note:** Actual performance depends on hardware. See benchmarks in `scripts/benchmark.py`.

## Configuration

Key configuration files:

- `.env`: Environment variables (device type, Redis URL, API settings)
- `backend/config/model_config.yaml`: Model specifications (Qwen3 + OLMo3 definitions)
- `backend/config/routing_rules.yaml`: Routing thresholds and model selection rules

### Supported Models

**Qwen3 Family:**
- qwen3_4b, qwen3_8b, qwen3_14b, qwen3_32b, qwen3_32b_thinking
- qwen3_vl (vision), qwen3_mt (translation)

**OLMo3 Family (newly added):**
- olmo3_7b, olmo3_7b_instruct
- olmo3_32b, olmo3_32b_instruct

All models support AWQ quantization for reduced VRAM usage.

## Fine-Tuning

1. Export chat history from ChatGPT/Claude:
```bash
python scripts/export_chat_history.py
```

2. Curate dataset:
```bash
python scripts/curate_dataset.py
```

3. Start fine-tuning:
```bash
curl -X POST http://localhost:8000/v1/finetune/start \
  -H "Content-Type: application/json" \
  -d '{
    "base_model": "qwen3-8b",
    "dataset_path": "./data/curated_dataset.jsonl",
    "epochs": 3
  }'
```

## Monitoring

- **Health Check**: `GET /v1/health`
- **Metrics**: `GET /v1/metrics`
- **Cache Stats**: `GET /v1/cache/stats`
- **Model Status**: `GET /v1/models`

## Development

### Running Tests

```bash
pytest tests/ -v
```

### Benchmarking

```bash
python scripts/benchmark.py
```

### Code Formatting

```bash
black backend/
flake8 backend/
```

## Multi-Device Access

### Tailscale Setup

1. Install Tailscale on server and client devices
2. Connect devices to same Tailnet
3. Access API via Tailscale IP: `http://<tailscale-ip>:8000`

## Development

**For developers continuing this project:**
- See [DEVELOPMENT_ROADMAP.md](DEVELOPMENT_ROADMAP.md) for development priorities and next steps
- See [GETTING_STARTED.md](GETTING_STARTED.md) for setup and troubleshooting
- See [SystemSpec.md](SystemSpec.md) for detailed technical specification

**Current Status:** ~45-50% complete. Core routing and caching work. Inference handlers need implementation.

### Running Tests

```bash
pytest tests/ -v
```

### Benchmarking Models

```bash
python scripts/benchmark.py
```

Compare Qwen3 vs OLMo3 performance across different model sizes.

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- **Alibaba Cloud** - Qwen3 model family
- **Allen Institute for AI** - OLMo3 model family
- **vLLM team** - GPU inference optimization
- **Unsloth** - Efficient fine-tuning
- **llama.cpp** - CPU/Metal inference

## Documentation

- **[GETTING_STARTED.md](GETTING_STARTED.md)** - Installation, setup, and troubleshooting
- **[DEVELOPMENT_ROADMAP.md](DEVELOPMENT_ROADMAP.md)** - Development priorities and next steps (for developers)
- **[SystemSpec.md](SystemSpec.md)** - Detailed technical specification and architecture
- **API Docs** - http://localhost:8000/docs (when server is running)

## Support

- Issues: [GitHub Issues](https://github.com/yourusername/GroupQwen/issues)
- Discussions: [GitHub Discussions](https://github.com/yourusername/GroupQwen/discussions)
