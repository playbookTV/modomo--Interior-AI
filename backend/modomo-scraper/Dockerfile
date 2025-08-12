FROM python:3.11-slim

# Install system dependencies for AI, SAM2, and Playwright
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    curl \
    unzip \
    git \
    gcc \
    g++ \
    build-essential \
    libgl1-mesa-glx \
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
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements files
COPY requirements-minimal.txt requirements-full.txt requirements-ai-stable.txt ./

# Always use full AI mode for production
ARG AI_MODE=full
ENV AI_MODE_BUILD=${AI_MODE}
ENV AI_MODE=full

# Always install AI dependencies - no fallback to minimal
RUN echo "🤖 Installing AI dependencies with SAM2 for production..." && \
    pip install --upgrade pip && \
    pip install --no-cache-dir torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements-ai-stable.txt && \
    echo "🔥 Installing SAM2 and CLIP from source..." && \
    pip install --no-cache-dir git+https://github.com/facebookresearch/segment-anything-2.git && \
    pip install --no-cache-dir git+https://github.com/openai/CLIP.git && \
    pip install --no-cache-dir iopath hydra-core omegaconf && \
    echo "✅ All AI dependencies including SAM2 and CLIP installed successfully"

# Install Playwright browsers for web scraping
RUN python -m playwright install-deps && \
    python -m playwright install chromium || echo "⚠️ Failed to install Playwright browsers"

# Copy application code
COPY . .

# Create directories for temp files and models
RUN mkdir -p /app/temp /app/downloads /app/models /app/checkpoints

# Download SAM2 model checkpoints
RUN echo "📦 Downloading SAM2 model checkpoints..." && \
    cd /app/checkpoints && \
    wget -q https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt && \
    wget -q https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt && \
    echo "✅ SAM2 checkpoints downloaded successfully" || echo "⚠️ SAM2 checkpoint download failed, will use fallback"

# Set environment variables
ENV PYTHONPATH=/app
ENV PORT=8001
ENV AI_MODE=${AI_MODE}

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Use Railway-optimized entry point
CMD python main_railway.py