#!/bin/bash
# Setup script for Qwen3 Local AI System

set -e

echo "====================================="
echo "Qwen3 Local AI System - Setup"
echo "====================================="

# Check Python version
python_version=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
required_version="3.10"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "Error: Python 3.10+ required (found $python_version)"
    exit 1
fi

echo "✓ Python $python_version detected"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing Python dependencies..."
pip install -r backend/requirements.txt

echo "✓ Dependencies installed"

# Create directories
echo "Creating required directories..."
mkdir -p models checkpoints logs data

echo "✓ Directories created"

# Copy environment file
if [ ! -f ".env" ]; then
    echo "Creating .env file..."
    cp .env.example .env
    echo "✓ .env file created (please edit with your configuration)"
else
    echo "✓ .env file already exists"
fi

# Check for GPU
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo "✓ NVIDIA GPU detected"
else
    echo "⚠ No NVIDIA GPU detected (CPU mode will be used)"
fi

# Check for Redis
if command -v redis-cli &> /dev/null; then
    if redis-cli ping > /dev/null 2>&1; then
        echo "✓ Redis is running"
    else
        echo "⚠ Redis is not running. Start it with: redis-server"
    fi
else
    echo "⚠ Redis not installed. Install with: sudo apt install redis-server"
fi

echo ""
echo "====================================="
echo "Setup complete!"
echo "====================================="
echo ""
echo "Next steps:"
echo "1. Edit .env file with your configuration"
echo "2. Download models: ./scripts/download_models.sh"
echo "3. Start the server: python -m backend.api.main"
echo ""
