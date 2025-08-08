# LLM Demo - Interior Design AI Testing Environment

This directory contains a testing environment for evaluating AI models that generate enhanced interior designs from uploaded images. It connects to the existing ReRoom AI service backend.

## Structure

- `frontend/` - React testing webapp for model evaluation
- `datasets/` - Training and test image datasets  
- `scripts/` - Utility scripts for data processing and evaluation
- `examples/` - Sample images and expected results

## Goals

Based on the testing PRD, this environment will help evaluate:
- Model training performance (YOLO, Stable Diffusion, CLIP)
- Object generation and detection accuracy
- Style transfer quality assessment
- Price discovery integration
- Tagging and metadata extraction

## Quick Start

```bash
# The backend is already running on Railway
# https://reroom-production-dcb0.up.railway.app:6969

# Install demo dependencies
cd llm-demo
pnpm install

# Start demo frontend
cd frontend  
pnpm dev
```

## Backend Integration

This demo uses the existing ReRoom AI service (`backend/ai-service/main.py`) which includes:
- **Object Detection**: YOLOv8 for furniture identification  
- **Style Transfer**: Stable Diffusion + ControlNet for makeovers
- **Product Recognition**: CLIP + BLIP for price discovery
- **Batch Processing**: For handling multiple images
- **Redis Caching**: For performance optimization
- **Circuit Breakers**: For fault tolerance

## API Endpoints

The demo connects to these existing endpoints:
- `POST /makeover` - Generate room makeovers
- `POST /makeover/enhanced` - Advanced makeovers with custom parameters
- `POST /batch` - Batch process multiple images
- `GET /health/detailed` - Comprehensive system status
- `POST /analyze/bulk` - Bulk image analysis
- `POST /analyze/style-comparison` - Compare multiple styles

## Features

- Image upload and processing with existing AI models
- Enhanced image display with clickable detected objects
- Object metadata viewing (name, type, price, confidence)
- Style comparison tool (Modern, Scandinavian, Industrial, etc.)
- Model performance monitoring and debugging
- Batch processing interface
- Real-time inference logging