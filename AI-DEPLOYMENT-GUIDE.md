# AI Deployment Guide - Modomo Full Features

Complete guide for deploying the full AI capabilities to Railway.

## 🎯 Current Status

✅ **Basic API**: Running successfully at `https://ovalay-recruitment-production.up.railway.app`  
✅ **Dashboard**: Updated to connect to Railway API  
✅ **Database**: Supabase schema initialized  
✅ **AI Code**: Full AI implementation ready (`main_full.py`)  

## 🤖 AI Features Ready to Deploy

- **GroundingDINO**: Object detection with text prompts
- **SAM2**: Instance segmentation masks
- **CLIP**: Embedding generation for similarity search
- **Vector Database**: pgvector integration for product matching
- **Real-time Processing**: Background job processing with Celery

## 🚀 Railway AI Deployment Options

### Option 1: Railway Dashboard (Recommended)

1. **Go to Railway Dashboard**: https://railway.app/dashboard
2. **Find your project**: "Ovalay Recruitment"
3. **Go to Settings → Environment**: Add:
   ```
   AI_MODE=full
   ```
4. **Go to Settings → Build**: Set build args:
   ```
   AI_MODE=full
   ```
5. **Deploy**: Click "Deploy" button

### Option 2: Railway CLI (If working)

```bash
# Navigate to service directory
cd backend/modomo-scraper

# Set AI mode
railway variables add AI_MODE=full

# Deploy with build args
railway up --service your-service-name
```

### Option 3: GitHub Integration (Automatic)

1. **Push to GitHub**: Commit all files including `main_full.py`
2. **Railway Auto-Deploy**: Will trigger automatically
3. **Set Environment**: Add `AI_MODE=full` in Railway dashboard

## 🔧 What Happens During AI Deployment

**Build Process** (~8-12 minutes):
1. Install system dependencies (OpenCV, CUDA support)
2. Install PyTorch (CPU version for Railway)
3. Install Transformers, CLIP, and other ML libraries
4. Download and cache model weights
5. Initialize AI services

**Runtime Features Enabled**:
- `/detect/process` - Real object detection
- Vector embeddings stored in Supabase
- Product similarity matching
- Automated dataset labeling

## 📊 Resource Requirements

**Memory**: 2-4GB (for model loading)  
**CPU**: 2+ cores (inference processing)  
**Storage**: 1GB+ (model weights)  
**Build Time**: 8-12 minutes  

## 🧪 Testing AI Deployment

Once deployed, test these endpoints:

```bash
# Health check with AI status
curl https://ovalay-recruitment-production.up.railway.app/health

# Test object detection (POST request)
curl -X POST https://ovalay-recruitment-production.up.railway.app/detect/process \
  -H "Content-Type: application/json" \
  -d '{"image_url": "https://example.com/room.jpg"}'
```

Expected response includes:
- `"mode": "full_ai"`
- `"ai_models": {"detector_loaded": true, ...}`

## 🎯 Production Considerations

### Performance Optimizations
- Models run on CPU (Railway GPU support limited)
- CLIP embeddings cached in database
- Background processing for heavy tasks
- Connection pooling for database

### Monitoring
- Health checks include AI model status
- Structured logging with `structlog`
- Error handling for model failures
- Graceful fallbacks to basic mode

### Scaling
- Horizontal scaling supported
- Models loaded per instance
- Shared database for embeddings
- Stateless processing design

## 🚨 Troubleshooting

**Build Failures**:
- Check Railway build logs for dependency issues
- Verify `AI_MODE=full` environment variable set
- Ensure sufficient memory allocation

**Runtime Issues**:
- Check `/health` endpoint for AI model status
- Monitor Railway resource usage
- Review application logs for model loading errors

**Fallback Strategy**:
- System automatically falls back to basic mode if AI dependencies fail
- All API endpoints remain functional
- Dashboard continues to work normally

## 🎉 Next Steps After AI Deployment

1. **Test Object Detection**: Use `/detect/process` endpoint
2. **Upload Sample Images**: Test full pipeline
3. **Review Dashboard**: Check AI-generated data
4. **Scale Processing**: Add background workers if needed
5. **Monitor Performance**: Watch Railway metrics

## 📝 File Summary

**AI Implementation**:
- `main_full.py` - Full AI FastAPI application
- `requirements-full.txt` - All AI dependencies
- `railway-ai.toml` - Railway AI deployment config

**Models Integrated**:
- GroundingDINO: Object detection from text prompts
- SAM2: Instance segmentation and masks  
- CLIP: Image-text embedding generation

**Database Integration**:
- pgvector extension for similarity search
- Optimized indexes for vector operations
- Real-time embedding storage

The AI deployment will transform your basic dataset creation system into a fully automated, AI-powered pipeline for interior design object detection and classification! 🤖✨