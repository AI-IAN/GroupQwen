# Qwen3 Local AI Orchestration System

A production-grade local AI inference system that intelligently routes queries across multiple Qwen3 models (4B through 32B variants) with semantic caching, fine-tuning capabilities, and multi-device remote access.

## Features

- **Intelligent Query Routing**: Automatically routes queries to optimal models based on complexity
- **Semantic Caching**: 40-60% of queries served from cache with <10ms latency
- **Multi-Model Support**: Qwen3-4B, 8B, 14B, 32B, VL, and MT variants
- **Fine-Tuning Pipeline**: QLoRA-based fine-tuning on custom datasets
- **Multi-Device Access**: Server, MacBook, and mobile access via Tailscale
- **Vision Capabilities**: Screenshot analysis, OCR, GUI automation with Qwen3-VL
- **Translation**: 92-language support with Qwen3-MT
- **Production-Ready**: Docker deployment, monitoring, health checks

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
│              Inference Layer                                │
│  Qwen3-8B • Qwen3-14B • Qwen3-32B • Qwen3-VL • Qwen3-MT    │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.10+
- NVIDIA GPU with 32GB VRAM (RTX 5090 or similar)
- CUDA 12.4+
- Redis server
- 64GB system RAM recommended

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/GroupQwen
cd GroupQwen
```

2. Run the setup script:
```bash
./scripts/setup.sh
```

3. Edit `.env` file with your configuration:
```bash
nano .env
```

4. Download models:
```bash
./scripts/download_models.sh
```

5. Start Redis (if not running):
```bash
redis-server
```

6. Start the API server:
```bash
python -m backend.api.main
```

The API will be available at `http://localhost:8000`

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
- **Qwen3-8B Latency**: <300ms
- **Qwen3-32B Latency**: <1000ms
- **GPU Utilization**: 90-95%

## Configuration

Key configuration files:

- `.env`: Environment variables
- `backend/config/model_config.yaml`: Model specifications
- `backend/config/routing_rules.yaml`: Routing thresholds

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

## Troubleshooting

See [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) for common issues.

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- Qwen team for the Qwen3 model family
- vLLM for GPU inference optimization
- Unsloth for efficient fine-tuning

## Support

- Documentation: [docs/](docs/)
- Issues: [GitHub Issues](https://github.com/yourusername/GroupQwen/issues)
- Discussions: [GitHub Discussions](https://github.com/yourusername/GroupQwen/discussions)
