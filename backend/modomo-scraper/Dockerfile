FROM python:3.11-slim

# Install system dependencies for AI and Playwright
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    curl \
    unzip \
    gcc \
    g++ \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements files
COPY requirements-minimal.txt requirements-full.txt requirements-ai-stable.txt ./

# Install dependencies based on environment variable or build arg
ARG AI_MODE=basic
ENV AI_MODE_BUILD=${AI_MODE}

# Install dependencies with better error handling
RUN if [ "${AI_MODE}" = "full" ] || [ "${AI_MODE_BUILD}" = "full" ] ; then \
        echo "🤖 Installing stable AI dependencies..." && \
        pip install --no-cache-dir -r requirements-ai-stable.txt || \
        (echo "❌ AI installation failed, falling back to minimal" && pip install --no-cache-dir -r requirements-minimal.txt) ; \
    else \
        echo "💡 Installing minimal dependencies..." && \
        pip install --no-cache-dir -r requirements-minimal.txt ; \
    fi

# Copy application code
COPY . .

# Create directories for temp files and models
RUN mkdir -p /app/temp /app/downloads /app/models

# Set environment variables
ENV PYTHONPATH=/app
ENV PORT=8001
ENV AI_MODE=${AI_MODE}

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Use Railway-optimized entry point
CMD python main_railway.py