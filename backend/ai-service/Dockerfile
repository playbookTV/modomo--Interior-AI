FROM python:3.11-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Download and cache AI models (optional - can be done at runtime)
RUN python -c "
import torch
from transformers import pipeline
from diffusers import StableDiffusionXLPipeline, ControlNetModel

# Download models to cache them in the Docker image
print('Downloading Stable Diffusion XL...')
try:
    pipe = StableDiffusionXLPipeline.from_pretrained(
        'stabilityai/stable-diffusion-xl-base-1.0',
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant='fp16'
    )
    del pipe
    print('✅ Stable Diffusion XL downloaded')
except Exception as e:
    print(f'⚠️ Failed to download SDXL: {e}')

print('Downloading ControlNet models...')
try:
    controlnet = ControlNetModel.from_pretrained(
        'diffusers/controlnet-depth-sdxl-1.0',
        torch_dtype=torch.float16
    )
    del controlnet
    print('✅ ControlNet depth downloaded')
except Exception as e:
    print(f'⚠️ Failed to download ControlNet: {e}')

print('Model download complete!')
"

# Copy application code
COPY . .

# Create non-root user
RUN adduser --disabled-password --gecos '' appuser
RUN chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

# Production image with GPU support
FROM base as gpu

USER root

# Install CUDA libraries if needed
# RUN pip install torch==2.1.1+cu121 torchvision==0.16.1+cu121 --index-url https://download.pytorch.org/whl/cu121

USER appuser

# Development image
FROM base as dev

USER root
RUN pip install --no-cache-dir jupyter ipython
USER appuser

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"] 