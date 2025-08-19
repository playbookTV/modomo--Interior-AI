#!/bin/bash
set -e

echo "🚀 Starting Modomo AI Service with cached models..."

# Check if models volume is empty and copy cached models if needed
if [ ! -d "/app/models/clip" ] || [ ! -d "/app/models/huggingface" ]; then
    echo "📦 Volume appears empty, copying cached models..."
    mkdir -p /app/models/clip /app/models/huggingface /app/models/sentence_transformers
    
    # Copy cached models from Docker layer to persistent volume
    if [ -d "/app/model_cache_base/clip" ]; then
        cp -r /app/model_cache_base/clip/* /app/models/clip/ 2>/dev/null || echo "No CLIP models to copy"
    fi
    
    if [ -d "/app/model_cache_base/sentence_transformers" ]; then
        cp -r /app/model_cache_base/sentence_transformers/* /app/models/sentence_transformers/ 2>/dev/null || echo "No sentence transformer models to copy"
    fi
    
    echo "✅ Cached models copied to persistent volume"
else
    echo "✅ Using existing models from persistent volume"
fi

# Ensure all required directories exist
mkdir -p /app/models/sam2 /app/models/yolo /app/temp /app/downloads

# Download SAM2 checkpoints if not present (one-time download to volume)
if [ ! -f "/app/models/sam2/sam2_hiera_large.pt" ]; then
    echo "📥 Downloading SAM2 checkpoint (one-time)..."
    python -c "
import os
import requests
from pathlib import Path

# Download SAM2 checkpoint
sam2_dir = Path('/app/models/sam2')
sam2_dir.mkdir(exist_ok=True)

checkpoint_url = 'https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt'
checkpoint_path = sam2_dir / 'sam2_hiera_large.pt'

if not checkpoint_path.exists():
    print('Downloading SAM2 checkpoint...')
    response = requests.get(checkpoint_url, stream=True)
    with open(checkpoint_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f'SAM2 checkpoint saved to {checkpoint_path}')
else:
    print('SAM2 checkpoint already exists')
" || echo "⚠️ SAM2 download failed, will use fallback"
fi

# Start the application
echo "🎯 Starting FastAPI application..."
exec "$@"