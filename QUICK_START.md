# Quick Start Guide

Get the Qwen3 Local AI Orchestration System running in 10 minutes.

## Prerequisites

- Ubuntu 22.04 or similar
- Python 3.10+
- NVIDIA GPU (32GB VRAM recommended)
- Redis

## Installation Steps

### 1. Clone and Setup

```bash
git clone https://github.com/yourusername/GroupQwen
cd GroupQwen
./scripts/setup.sh
```

This will:
- Create a virtual environment
- Install Python dependencies
- Create necessary directories
- Copy `.env.example` to `.env`

### 2. Configure Environment

Edit `.env` file:

```bash
nano .env
```

Key settings:
```env
DEVICE_TYPE=server              # or 'macbook' for edge devices
REDIS_URL=redis://localhost:6379
LOG_LEVEL=INFO
```

### 3. Start Redis

```bash
# Ubuntu/Debian
sudo systemctl start redis-server

# Or run directly
redis-server
```

Verify Redis is running:
```bash
redis-cli ping
# Should return: PONG
```

### 4. Download Models (Optional)

For testing without models, skip this step. The system will use placeholder responses.

```bash
./scripts/download_models.sh
```

Select model to download:
- Option 2: Qwen3-8B-AWQ (6GB) - Recommended for testing
- Option 5: All models (40GB+) - Full deployment

### 5. Start the API Server

```bash
source venv/bin/activate
python -m backend.api.main
```

You should see:
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 6. Test the API

In a new terminal:

```bash
# Health check
curl http://localhost:8000/v1/health

# Chat completion
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Hello! How are you?"}
    ]
  }'
```

### 7. Access Documentation

Open your browser:
- API Docs: http://localhost:8000/docs
- Alternative Docs: http://localhost:8000/redoc

## Docker Deployment (Alternative)

If you prefer Docker:

```bash
# Build and start all services
docker-compose up -d

# Check logs
docker-compose logs -f backend

# Stop services
docker-compose down
```

## Troubleshooting

### Redis Connection Error

```bash
# Check if Redis is running
redis-cli ping

# Start Redis
sudo systemctl start redis-server
```

### GPU Not Detected

```bash
# Check GPU
nvidia-smi

# Verify CUDA
nvcc --version
```

### Import Errors

```bash
# Reinstall dependencies
source venv/bin/activate
pip install -r backend/requirements.txt
```

### Port Already in Use

```bash
# Change API_PORT in .env
API_PORT=8001

# Or kill existing process
lsof -ti:8000 | xargs kill -9
```

## Next Steps

1. **Configure Models**: Edit `backend/config/model_config.yaml`
2. **Adjust Routing**: Edit `backend/config/routing_rules.yaml`
3. **Monitor Performance**: Access `/v1/metrics` endpoint
4. **Run Benchmarks**: `python scripts/benchmark.py`
5. **Fine-tune Models**: See main README.md for fine-tuning guide

## Common Commands

```bash
# Activate virtual environment
source venv/bin/activate

# Start server
python -m backend.api.main

# Run tests
pytest tests/ -v

# Check logs
tail -f logs/app.log

# Clear cache
curl -X POST http://localhost:8000/v1/cache/clear
```

## Support

- Full Documentation: [README.md](README.md)
- System Spec: [SystemSpec.md](SystemSpec.md)
- Issues: GitHub Issues

---

**Time to first response**: ~10 minutes
**Production-ready**: Follow full setup in README.md
