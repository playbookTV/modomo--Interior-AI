# Modomo Railway Deployment Guide

Complete guide for deploying the Modomo Dataset Scraping System to Railway with Supabase and Cloudflare R2.

## 🎯 Architecture Overview

**Services:**
- **Railway**: Modomo Scraper API (FastAPI backend)
- **Supabase**: PostgreSQL database with pgvector  
- **Cloudflare R2**: Object storage for images and datasets
- **Cloudflare Pages**: Review dashboard hosting (separate deployment)

## 🚀 Railway Deployment Steps

### 1. Prerequisites
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login to Railway
railway login
```

### 2. Deploy the Backend
```bash
# Run the automated deployment script
./scripts/deploy-to-railway.sh
```

**Or manually:**
```bash
cd backend/modomo-scraper
railway init
railway up --detach
```

### 3. Set Environment Variables
The deployment script automatically sets:

```bash
DATABASE_URL_CLOUD="postgresql://postgres:qYs8XprwrHQoM7n6@db.nyeeewcpexqsqfzzmvyu.supabase.co:5432/postgres"
CLOUDFLARE_R2_ENDPOINT="https://9bbdb3861142e65685d23f4955b88ebe.r2.cloudflarestorage.com"
CLOUDFLARE_R2_BUCKET="reroom"
CLOUDFLARE_R2_ACCESS_KEY_ID="6c8abdff2cdad89323e36b258b1d0f4b"
CLOUDFLARE_R2_SECRET_ACCESS_KEY="35b7b4a1f586211d407b246b54898d2e50d13562cba7e7be6293d4b6ccea06c5"
CORS_ORIGINS="*"
ENVIRONMENT="production"
```

### 4. Initialize Database Schema
**Option A: Railway Console**
```bash
railway run python init_db.py
```

**Option B: Supabase SQL Editor**
1. Go to Supabase dashboard
2. Open SQL Editor  
3. Copy contents of `backend/modomo-scraper/database/schema.sql`
4. Execute the schema

### 5. Verify Deployment
```bash
# Check deployment status
railway status

# View logs
railway logs

# Get service URL
railway domain
```

## 🔧 Configuration Details

### Files Created for Railway:
- `railway.json` - Railway deployment configuration
- `Dockerfile` - Production-ready container
- `.env.example` - Environment variables template
- `init_db.py` - Database schema initialization
- `README-RAILWAY.md` - Deployment documentation

### Key Features:
- **Health Checks**: `/health` endpoint for Railway monitoring
- **Port Configuration**: Automatic PORT environment variable detection
- **Database Connection**: Optimized connection pooling for Supabase
- **Error Handling**: Graceful fallbacks and comprehensive logging
- **CORS Configuration**: Flexible cross-origin settings

## 📊 Expected Services URLs

After deployment, your services will be:

**API Service (Railway):**
```
https://modomo-scraper-production-xxxx.up.railway.app
```

**Endpoints:**
- `GET /health` - Health check
- `GET /docs` - API documentation
- `GET /stats/dataset` - Dataset statistics
- `POST /scrape/scenes` - Start scraping jobs

## 🌐 Dashboard Integration

Update your review dashboard's API configuration:

```typescript
// review-dashboard/src/api/client.ts
const API_BASE_URL = 'https://your-railway-service.up.railway.app'
```

Or set environment variable:
```bash
VITE_API_URL=https://your-railway-service.up.railway.app
```

## 🔍 Monitoring & Debugging

**Railway Commands:**
```bash
railway logs               # View application logs
railway ps                 # Check service status  
railway shell             # Access container shell
railway variables          # List environment variables
```

**Health Check URLs:**
```bash
curl https://your-service.up.railway.app/health
curl https://your-service.up.railway.app/stats/dataset
```

## 💡 Production Considerations

✅ **Completed:**
- Supabase PostgreSQL with pgvector support
- Cloudflare R2 storage integration
- Production-ready Docker configuration
- Health monitoring and logging
- CORS and security headers
- Database connection pooling

🚀 **Optional Enhancements:**
- Redis plugin for background job queuing
- Custom domain setup
- Monitoring/alerting integration
- Rate limiting for API endpoints

## 🎉 Deployment Status

**Railway Backend**: Ready for deployment  
**Database Schema**: Ready to initialize  
**Storage**: Cloudflare R2 configured  
**Environment**: All variables prepared  

**Next Action:** Run `./scripts/deploy-to-railway.sh` to deploy to Railway!