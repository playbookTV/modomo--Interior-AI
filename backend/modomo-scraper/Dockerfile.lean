FROM python:3.11-slim

# Install minimal system dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl wget git gcc g++ build-essential \
    libgl1-mesa-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /app

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8001
ENV AI_MODE=minimal

# Copy requirements and install ONLY essential packages
COPY requirements-minimal.txt ./
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements-minimal.txt

# DO NOT install AI models during build - they'll be downloaded at runtime to volume
# DO NOT download PyTorch/transformers during build - keep image small

# Create directory structure (but don't download anything)
RUN mkdir -p /app/temp /app/downloads /app/models

# Copy application code
COPY . .

# Set minimal environment for runtime model downloads
ENV PYTHONPATH=/app:/app/models
ENV MODEL_CACHE_DIR=/app/models
ENV TRANSFORMERS_CACHE=/app/models/huggingface
ENV HF_HOME=/app/models/huggingface

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Direct startup - models will download on first run to volume
CMD python main_railway.py