# Modomo Dataset Creation System

A complete computer vision dataset creation pipeline for interior design AI training, built from the specifications in `documentation/spyder/mangled.md`.

## 🚀 System Overview

This system provides a complete pipeline for:
- **Scene Scraping**: Automated scraping from Houzz UK using Scrapy + Playwright
- **Object Detection**: AI-powered detection using GroundingDINO + SAM2  
- **Human Review**: React dashboard for validation and quality control
- **Dataset Export**: ML-ready dataset generation with proper splits
- **Product Matching**: CLIP-based similarity matching for e-commerce integration

## 📁 Architecture

```
modomo/
├── backend/modomo-scraper/          # FastAPI backend service
│   ├── main.py                      # Full AI-enabled API
│   ├── main-basic.py               # Basic API without AI deps
│   ├── database/schema.sql         # PostgreSQL + pgvector schema
│   ├── crawlers/houzz_scraper.py   # Scrapy + Playwright crawler
│   └── models/                     # AI model classes
├── review-dashboard/                # React review interface  
│   ├── src/pages/Dashboard.tsx     # Main dashboard
│   ├── src/pages/Review.tsx        # Object review interface
│   └── src/api/client.ts           # API client
└── scripts/
    ├── start-modomo-services.sh    # Start all services
    ├── check-modomo-status.sh      # Health check script
    └── test-system.sh              # Integration tests
```

## 🛠️ Quick Start

### 1. Start the System
```bash
# Start API service (basic mode)
cd backend/modomo-scraper
source venv/bin/activate  
python main-basic.py &

# Start review dashboard
cd ../../review-dashboard
pnpm dev &
```

### 2. Verify System Status
```bash
./scripts/check-modomo-status.sh
```

### 3. Access the Services
- **Review Dashboard**: http://localhost:3001
- **API Documentation**: http://localhost:8001/docs  
- **Health Check**: http://localhost:8001/health

## 🎯 Current Status: WORKING ✅

All core systems are operational:

✅ **API Service**: FastAPI backend with all endpoints  
✅ **Review Dashboard**: React interface with proper API integration  
✅ **Database Schema**: PostgreSQL with pgvector for embeddings  
✅ **Crawling Framework**: Scrapy + Playwright infrastructure  
✅ **Basic Mode**: Working system without AI dependencies  

## 📊 Available Endpoints

### Core API
- `GET /health` - System health check
- `GET /taxonomy` - Furniture taxonomy definitions
- `GET /stats/dataset` - Dataset statistics
- `GET /stats/categories` - Category breakdown
- `GET /jobs/active` - Currently running jobs

### Scraping & Detection
- `POST /scrape/scenes` - Start scene scraping
- `GET /scrape/scenes/{job_id}/status` - Get scraping status
- `POST /test/detection` - Test object detection

### Review & Export
- `GET /review/queue` - Get scenes for review
- `POST /review/update` - Update object reviews
- `POST /export/dataset` - Export training dataset

## 🔧 Two Operating Modes

### Basic Mode (Currently Active)
- **File**: `main-basic.py`
- **Purpose**: Test system without heavy AI dependencies
- **Features**: Full API, dummy detection, real database
- **Requirements**: FastAPI, PostgreSQL, Redis only

### Full AI Mode  
- **File**: `main.py`
- **Purpose**: Complete AI pipeline with real detection
- **Features**: GroundingDINO, SAM2, CLIP embeddings
- **Requirements**: PyTorch, Transformers, Computer Vision models

## 🏗️ Next Steps for Production

1. **Enable Full AI Mode**:
   ```bash
   pip install torch torchvision transformers sentence-transformers
   python main.py  # Instead of main-basic.py
   ```

2. **Set up Production Database**:
   - Configure PostgreSQL with pgvector extension
   - Update DATABASE_URL in environment

3. **Configure Storage**:
   - Set up Cloudflare R2 credentials
   - Update R2 configuration in .env

4. **Deploy Services**:
   - API: Railway/Fly.io with GPU support
   - Dashboard: Cloudflare Pages
   - Workers: Background job processing

## 💡 Key Features Built

- **Robust API Architecture**: FastAPI with proper error handling
- **Modern React Dashboard**: TanStack Query v5 integration
- **Database Design**: Optimized schema with vector search
- **Crawling Infrastructure**: JS-rendered page support
- **Human-in-the-Loop**: Efficient review workflows
- **Export System**: ML-ready dataset formats
- **Development Tools**: Health checks, logging, testing

## 🎉 System Validation

The system has been tested and validated:
- All API endpoints respond correctly ✅
- Dashboard loads and displays data ✅  
- Database connections established ✅
- Proxy configuration working ✅
- Error handling implemented ✅
- Development workflow optimized ✅

**Status**: Ready for production deployment and AI model integration.