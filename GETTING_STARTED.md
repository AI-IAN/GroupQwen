# Getting Started

Get the Local Multi-Model AI Orchestration System running in under 15 minutes.

---

## ⚡ Quick Start

### Prerequisites

- **Operating System:** Ubuntu 22.04+ or macOS
- **Python:** 3.10 or higher
- **GPU:** NVIDIA GPU with 8GB+ VRAM (recommended) or Apple Silicon Mac
- **RAM:** 16GB+ system memory
- **Redis:** Redis server (for caching)

---

## 📦 Installation

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/GroupQwen
cd GroupQwen
```

### 2. Run Setup Script

The setup script will create a virtual environment and install dependencies:

```bash
./scripts/setup.sh
```

This script:
- Creates a Python virtual environment (`venv/`)
- Installs all required packages
- Creates necessary directories
- Copies `.env.example` to `.env`

### 3. Configure Environment

Edit the `.env` file with your settings:

```bash
nano .env
```

**Key Configuration Options:**

```env
# Device type: "server" (GPU) or "macbook" (CPU/Metal)
DEVICE_TYPE=server

# Redis connection
REDIS_URL=redis://localhost:6379

# API settings
API_HOST=0.0.0.0
API_PORT=8000

# Logging
LOG_LEVEL=INFO
```

### 4. Start Redis

**Ubuntu/Debian:**
```bash
sudo systemctl start redis-server

# Verify it's running
redis-cli ping  # Should return: PONG
```

**macOS (Homebrew):**
```bash
brew services start redis

# Verify it's running
redis-cli ping  # Should return: PONG
```

**Docker:**
```bash
docker run -d -p 6379:6379 redis:latest
```

### 5. Download Models (Optional)

For initial testing, you can skip this step. The system will use placeholder responses.

To download actual models:

```bash
./scripts/download_models.sh
```

**Recommended for Testing:**
- **Option 2:** Qwen3-8B-AWQ (~6GB) - Fast, balanced performance
- **Option 8:** OLMo3-7B-Instruct (~5GB) - Alternative model for comparison

**For Full Deployment:**
- **Option 5:** All Qwen3 models (~40GB)
- Download OLMo3 models separately

### 6. Start the API Server

```bash
source venv/bin/activate
python -m backend.api.main
```

You should see:
```
INFO:     Started server process
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

---

## 🧪 Test the System

### Health Check

```bash
curl http://localhost:8000/v1/health
```

**Expected Response:**
```json
{
  "status": "healthy",
  "redis_connected": true,
  "gpu_available": true,
  "models_loaded": 0
}
```

### Chat Completion

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Hello! Explain quantum computing in simple terms."}
    ],
    "temperature": 0.7
  }'
```

### Translation (Qwen3-MT)

```bash
curl -X POST http://localhost:8000/v1/translate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, how are you?",
    "target_lang": "es"
  }'
```

### Check System Metrics

```bash
curl http://localhost:8000/v1/metrics
```

### Cache Statistics

```bash
curl http://localhost:8000/v1/cache/stats
```

---

## 📖 API Documentation

Once the server is running, access interactive API documentation:

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

---

## 🐳 Docker Deployment (Alternative)

If you prefer Docker:

```bash
# Build and start all services
docker-compose up -d

# Check logs
docker-compose logs -f backend

# Stop services
docker-compose down
```

---

## 🔧 Troubleshooting

### Redis Connection Error

**Problem:** `ConnectionError: Error connecting to Redis`

**Solution:**
```bash
# Check if Redis is running
redis-cli ping

# If not running, start it
sudo systemctl start redis-server  # Linux
brew services start redis          # macOS
```

### GPU Not Detected

**Problem:** `GPU not available, falling back to CPU`

**Solution:**
```bash
# Check GPU
nvidia-smi

# Verify CUDA installation
nvcc --version

# Reinstall PyTorch with CUDA support if needed
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Import Errors

**Problem:** `ModuleNotFoundError: No module named 'backend'`

**Solution:**
```bash
# Activate virtual environment
source venv/bin/activate

# Reinstall dependencies
pip install -r backend/requirements.txt
```

### Port Already in Use

**Problem:** `OSError: [Errno 98] Address already in use`

**Solution:**
```bash
# Find process using port 8000
lsof -ti:8000

# Kill the process
lsof -ti:8000 | xargs kill -9

# Or change the port in .env
API_PORT=8001
```

### Memory Issues

**Problem:** `CUDA out of memory`

**Solution:**
- Use smaller models (Qwen3-4B or Qwen3-8B instead of 32B)
- Enable quantization (AWQ or Q4)
- Reduce `max_model_len` in model config
- Close other GPU-intensive applications

---

## 🎯 What's Next?

Now that the system is running:

1. **Explore the API:** Try different endpoints via http://localhost:8000/docs
2. **Test Multiple Models:** Compare Qwen3 vs OLMo3 performance
3. **Monitor Performance:** Check cache hit rates and latencies at `/v1/metrics`
4. **Optimize Configuration:** Adjust routing rules and model settings
5. **Continue Development:** See `DEVELOPMENT_ROADMAP.md` for next steps

---

## 📚 Documentation

- **README.md** - Project overview and architecture
- **DEVELOPMENT_ROADMAP.md** - Development roadmap and priorities (for developers)
- **SystemSpec.md** - Detailed technical specification
- **.env.example** - Configuration options

---

## 🔐 Multi-Device Access (Optional)

### Access from Other Devices via Tailscale

1. **Install Tailscale** on server and client devices
2. **Connect both devices** to the same Tailnet
3. **Access API** from any device using Tailscale IP:
   ```
   http://<tailscale-ip>:8000
   ```

---

## 🚀 Common Commands

```bash
# Activate virtual environment
source venv/bin/activate

# Start API server
python -m backend.api.main

# Run tests
pytest tests/ -v

# Check code formatting
black backend/ --check
flake8 backend/

# View logs
tail -f logs/app.log

# Clear cache
curl -X POST http://localhost:8000/v1/cache/clear
```

---

## 💬 Need Help?

- **Documentation:** See other .md files in the repository
- **API Reference:** http://localhost:8000/docs
- **Issues:** Check GitHub Issues
- **System Spec:** See `SystemSpec.md` for detailed architecture

---

**Time to First Response:** ~15 minutes (with setup)
**Time to Production:** See `DEVELOPMENT_ROADMAP.md` for development priorities
