#!/bin/bash
# Model download script for Qwen3 models

set -e

echo "====================================="
echo "Qwen3 Model Downloader"
echo "====================================="

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
echo "1. Qwen3-4B-AWQ (3GB)"
echo "2. Qwen3-8B-AWQ (6GB)"
echo "3. Qwen3-14B-AWQ (11GB)"
echo "4. Qwen3-32B-AWQ (20GB)"
echo "5. All models"
echo ""
read -p "Select models to download (1-5): " choice

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
        download_model "Qwen3-4B-AWQ" "Qwen/Qwen3-4B-AWQ"
        download_model "Qwen3-8B-AWQ" "Qwen/Qwen3-8B-AWQ"
        download_model "Qwen3-14B-AWQ" "Qwen/Qwen3-14B-AWQ"
        download_model "Qwen3-32B-AWQ" "Qwen/Qwen3-32B-AWQ"
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "====================================="
echo "Download complete!"
echo "====================================="
