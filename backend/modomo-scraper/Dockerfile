FROM python:3.11-slim

# Install system dependencies for AI, SAM2, and Playwright
# Updated for Debian Trixie compatibility
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    curl \
    unzip \
    git \
    gcc \
    g++ \
    build-essential \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    libglx-mesa0 \
    mesa-common-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libnspr4 \
    libnss3 \
    libdbus-1-3 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libxkbcommon0 \
    libatspi2.0-0 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    libgbm1 \
    libcairo2 \
    libpango-1.0-0 \
    libasound2 \
    libgtk-3-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements files  
COPY requirements-railway-complete.txt requirements-railway-fixed.txt requirements-minimal.txt requirements-full.txt requirements-ai-stable.txt ./

# Always use full AI mode for production
ARG AI_MODE=full
ENV AI_MODE_BUILD=${AI_MODE}
ENV AI_MODE=full

# Install optimized dependencies for Railway deployment with caching
RUN echo "🚀 Installing optimized AI dependencies for Railway (with caching)..." && \
    pip install --upgrade pip && \
    pip install -r requirements-railway-complete.txt && \
    echo "🔥 Installing SAM2 and additional models..." && \
    pip install git+https://github.com/facebookresearch/segment-anything-2.git && \
    pip install git+https://github.com/openai/CLIP.git && \
    pip install iopath hydra-core omegaconf && \
    echo "✅ All optimized dependencies with enhanced classification installed (cached)"

# Install Playwright browsers for web scraping
RUN python -m playwright install-deps && \
    python -m playwright install chromium || echo "⚠️ Failed to install Playwright browsers"

# Copy application code
COPY . .

# Explicitly ensure models directory and __init__.py are copied
COPY models/ /app/models/
COPY models/__init__.py /app/models/__init__.py

# Verify models directory structure
RUN echo "🔍 Verifying models directory structure..." && \
    ls -la /app/models/ && \
    echo "🔍 __init__.py exists: $(test -f /app/models/__init__.py && echo 'YES' || echo 'NO')" && \
    echo "✅ Models directory verification complete"

# Create directories for temp files and models (Railway Volume will mount to /app/models)
RUN mkdir -p /app/temp /app/downloads /app/checkpoints

# Setup model directories - models will be stored on Railway Volume
RUN echo "📦 Setting up model directories for Railway Volume..." && \
    mkdir -p /app/models/sam2 /app/models/clip /app/models/grounding_dino && \
    echo "✅ Model directories ready for Railway Volume mount"

# Setup additional cache directories for Phase 2 optimization
RUN echo "🚀 Setting up Phase 2 cache directories..." && \
    mkdir -p /app/cache_volume/huggingface /app/cache_volume/spacy /app/cache_volume/playwright /app/cache_volume/pip /app/cache_volume/nltk /app/cache_volume/matplotlib && \
    echo "✅ Phase 2 cache directories ready"

# Set environment variables for enhanced classification system
ENV PYTHONPATH=/app:/app/models
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8001
ENV AI_MODE=${AI_MODE}
ENV CLASSIFICATION_ENHANCED=true
ENV KEYWORD_SYSTEM_VERSION=2.0
ENV MODEL_CACHE_DIR=/app/models
ENV SAM2_CHECKPOINT_DIR=/app/models/sam2
ENV CLIP_CACHE_DIR=/app/models/clip

# Phase 2 cache optimization environment variables
ENV TRANSFORMERS_CACHE=/app/cache_volume/huggingface
ENV HF_HOME=/app/cache_volume/huggingface
ENV SENTENCE_TRANSFORMERS_HOME=/app/cache_volume/huggingface/sentence_transformers
ENV SPACY_DATA=/app/cache_volume/spacy
ENV PLAYWRIGHT_BROWSERS_PATH=/app/cache_volume/playwright
ENV PIP_CACHE_DIR=/app/cache_volume/pip
ENV NLTK_DATA=/app/cache_volume/nltk
ENV MPLCONFIGDIR=/app/cache_volume/matplotlib

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Use Railway-optimized entry point
CMD python main_railway.py