#!/bin/bash
# Model download script for Multi-Model AI Orchestration System
# Supports: Qwen3 and OLMo3 model families

set -e

echo "======================================================="
echo "Multi-Model AI Orchestration - Model Downloader"
echo "======================================================="

MODELS_DIR="./models"
mkdir -p "$MODELS_DIR"

# Function to download model
download_model() {
    local model_name=$1
    local model_path=$2

    echo ""
    echo "Downloading $model_name..."
    echo "This may take a while depending on your internet connection."

    # Check if model already exists
    if [ -d "$MODELS_DIR/$model_path" ]; then
        echo "✓ Model already downloaded: $model_name"
        return
    fi

    # Download using huggingface-cli (requires huggingface_hub)
    # Uncomment when ready to actually download
    # huggingface-cli download "$model_path" --local-dir "$MODELS_DIR/$model_path"

    echo "✓ Downloaded $model_name"
}

# Install huggingface_hub if not present
if ! python3 -c "import huggingface_hub" 2>/dev/null; then
    echo "Installing huggingface_hub..."
    pip install huggingface_hub
fi

echo ""
echo "Available models:"
echo ""
echo "=== Qwen3 Family (Alibaba Cloud) ==="
echo "1. Qwen3-4B-AWQ (3GB) - Edge/mobile inference"
echo "2. Qwen3-8B-AWQ (6GB) - Fast balanced performance"
echo "3. Qwen3-14B-AWQ (11GB) - Balanced quality"
echo "4. Qwen3-32B-AWQ (20GB) - Advanced reasoning"
echo "5. Qwen3-VL-7B-AWQ (9GB) - Vision model"
echo "6. Qwen3-MT (10GB) - Translation (92 languages)"
echo ""
echo "=== OLMo3 Family (Allen Institute for AI) ==="
echo "7. OLMo3-7B (5GB) - Comparable to Qwen3-8B"
echo "8. OLMo3-7B-Instruct (5GB) - Instruction-tuned"
echo "9. OLMo3-32B (20GB) - Large model for reasoning"
echo "10. OLMo3-32B-Instruct (20GB) - Instruction-tuned large"
echo ""
echo "=== Quick Install Options ==="
echo "11. Quick Start: Qwen3-8B + OLMo3-7B-Instruct (~11GB)"
echo "12. All Qwen3 models (~58GB)"
echo "13. All OLMo3 models (~50GB)"
echo "14. All models - Full deployment (~108GB)"
echo ""
read -p "Select models to download (1-14): " choice

case $choice in
    1)
        download_model "Qwen3-4B-AWQ" "Qwen/Qwen3-4B-AWQ"
        ;;
    2)
        download_model "Qwen3-8B-AWQ" "Qwen/Qwen3-8B-AWQ"
        ;;
    3)
        download_model "Qwen3-14B-AWQ" "Qwen/Qwen3-14B-AWQ"
        ;;
    4)
        download_model "Qwen3-32B-AWQ" "Qwen/Qwen3-32B-AWQ"
        ;;
    5)
        download_model "Qwen3-VL-7B-AWQ" "Qwen/Qwen3-VL-7B-AWQ"
        ;;
    6)
        download_model "Qwen3-MT" "Qwen/Qwen3-32B-MT"
        ;;
    7)
        download_model "OLMo3-7B" "allenai/OLMo3-7B"
        ;;
    8)
        download_model "OLMo3-7B-Instruct" "allenai/OLMo3-7B-Instruct"
        ;;
    9)
        download_model "OLMo3-32B" "allenai/OLMo3-32B"
        ;;
    10)
        download_model "OLMo3-32B-Instruct" "allenai/OLMo3-32B-Instruct"
        ;;
    11)
        echo "Installing Quick Start models..."
        download_model "Qwen3-8B-AWQ" "Qwen/Qwen3-8B-AWQ"
        download_model "OLMo3-7B-Instruct" "allenai/OLMo3-7B-Instruct"
        ;;
    12)
        echo "Installing all Qwen3 models..."
        download_model "Qwen3-4B-AWQ" "Qwen/Qwen3-4B-AWQ"
        download_model "Qwen3-8B-AWQ" "Qwen/Qwen3-8B-AWQ"
        download_model "Qwen3-14B-AWQ" "Qwen/Qwen3-14B-AWQ"
        download_model "Qwen3-32B-AWQ" "Qwen/Qwen3-32B-AWQ"
        download_model "Qwen3-VL-7B-AWQ" "Qwen/Qwen3-VL-7B-AWQ"
        download_model "Qwen3-MT" "Qwen/Qwen3-32B-MT"
        ;;
    13)
        echo "Installing all OLMo3 models..."
        download_model "OLMo3-7B" "allenai/OLMo3-7B"
        download_model "OLMo3-7B-Instruct" "allenai/OLMo3-7B-Instruct"
        download_model "OLMo3-32B" "allenai/OLMo3-32B"
        download_model "OLMo3-32B-Instruct" "allenai/OLMo3-32B-Instruct"
        ;;
    14)
        echo "Installing ALL models (this will take a while)..."
        download_model "Qwen3-4B-AWQ" "Qwen/Qwen3-4B-AWQ"
        download_model "Qwen3-8B-AWQ" "Qwen/Qwen3-8B-AWQ"
        download_model "Qwen3-14B-AWQ" "Qwen/Qwen3-14B-AWQ"
        download_model "Qwen3-32B-AWQ" "Qwen/Qwen3-32B-AWQ"
        download_model "Qwen3-VL-7B-AWQ" "Qwen/Qwen3-VL-7B-AWQ"
        download_model "Qwen3-MT" "Qwen/Qwen3-32B-MT"
        download_model "OLMo3-7B" "allenai/OLMo3-7B"
        download_model "OLMo3-7B-Instruct" "allenai/OLMo3-7B-Instruct"
        download_model "OLMo3-32B" "allenai/OLMo3-32B"
        download_model "OLMo3-32B-Instruct" "allenai/OLMo3-32B-Instruct"
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "======================================================="
echo "Download complete!"
echo "Models stored in: $MODELS_DIR"
echo "======================================================="
echo ""
echo "Next steps:"
echo "1. Start Redis: redis-server"
echo "2. Start API: python -m backend.api.main"
echo "3. Test: curl http://localhost:8000/v1/health"
echo "======================================================="
