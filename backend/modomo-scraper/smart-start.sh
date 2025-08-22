#!/bin/bash
set -e

echo "🚀 Starting Modomo AI Service..."

# Function to check if models exist
check_models() {
    echo "🔍 Checking for existing models in volume..."
    
    # Check for key model indicators
    if [ -d "/app/models/huggingface/transformers" ] && \
       [ -d "/app/models/sentence_transformers" ] && \
       [ -f "/app/models/sam2/sam2_hiera_large.pt" ]; then
        echo "✅ Models found in volume, skipping download"
        return 0
    else
        echo "📥 Models not found, will download to volume"
        return 1
    fi
}

# Download models only if they don't exist
download_models() {
    echo "🔄 Downloading models to persistent volume..."
    
    # Ensure directories exist
    mkdir -p /app/models/huggingface /app/models/sentence_transformers /app/models/sam2 /app/models/clip /app/models/yolo
    
    # Download core models to volume
    python -c "
import os
import torch
from transformers import CLIPModel, CLIPProcessor
from sentence_transformers import SentenceTransformer
import requests

print('Downloading CLIP model to volume...')
try:
    model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32', cache_dir='/app/models/clip')
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32', cache_dir='/app/models/clip')
    print('✅ CLIP model cached')
except Exception as e:
    print(f'⚠️ CLIP download failed: {e}')

print('Downloading sentence transformer to volume...')
try:
    st_model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder='/app/models/sentence_transformers')
    print('✅ Sentence transformer cached')
except Exception as e:
    print(f'⚠️ Sentence transformer download failed: {e}')

print('Downloading SAM2 checkpoint to volume...')
try:
    checkpoint_url = 'https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt'
    checkpoint_path = '/app/models/sam2/sam2_hiera_large.pt'
    if not os.path.exists(checkpoint_path):
        response = requests.get(checkpoint_url, stream=True)
        os.makedirs('/app/models/sam2', exist_ok=True)
        with open(checkpoint_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print('✅ SAM2 checkpoint downloaded')
    else:
        print('✅ SAM2 checkpoint already exists')
except Exception as e:
    print(f'⚠️ SAM2 download failed: {e}')

print('✅ Model download complete')
"
    
    echo "✅ All models downloaded to volume"
}

# Main logic
if check_models; then
    echo "🎯 Using existing models from volume"
else
    download_models
fi

echo "🎯 Starting FastAPI application..."
exec python main_refactored.py