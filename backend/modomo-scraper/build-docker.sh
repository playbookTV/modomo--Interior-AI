#!/bin/bash

# Docker build optimization script for modomo-scraper
# Usage: ./build-docker.sh [minimal|optimized|original]

set -e

BUILD_TYPE=${1:-optimized}
IMAGE_NAME="modomo-scraper"
TIMESTAMP=$(date +%s)

echo "🚀 Building Docker image with $BUILD_TYPE configuration..."

case $BUILD_TYPE in
    "minimal")
        echo "📦 Building minimal image for development..."
        docker build -f Dockerfile.minimal -t ${IMAGE_NAME}:minimal-${TIMESTAMP} . --progress=plain
        docker tag ${IMAGE_NAME}:minimal-${TIMESTAMP} ${IMAGE_NAME}:minimal-latest
        echo "✅ Minimal build complete: ${IMAGE_NAME}:minimal-latest"
        ;;
    "optimized")
        echo "🔥 Building optimized image for production..."
        docker build -f Dockerfile.optimized -t ${IMAGE_NAME}:optimized-${TIMESTAMP} . --progress=plain
        docker tag ${IMAGE_NAME}:optimized-${TIMESTAMP} ${IMAGE_NAME}:optimized-latest
        echo "✅ Optimized build complete: ${IMAGE_NAME}:optimized-latest"
        ;;
    "original")
        echo "📋 Building with original Dockerfile..."
        docker build -f Dockerfile -t ${IMAGE_NAME}:original-${TIMESTAMP} . --progress=plain
        docker tag ${IMAGE_NAME}:original-${TIMESTAMP} ${IMAGE_NAME}:original-latest
        echo "✅ Original build complete: ${IMAGE_NAME}:original-latest"
        ;;
    *)
        echo "❌ Invalid build type. Use: minimal, optimized, or original"
        exit 1
        ;;
esac

echo "🎯 Build completed in $(date)"
echo "📊 Image sizes:"
docker images ${IMAGE_NAME} --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"