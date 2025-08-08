# Modomo Scraper - Railway Deployment Guide

Deploy the Modomo Dataset Scraping service to Railway with Supabase database and Cloudflare R2 storage.

## 🚀 Quick Railway Deployment

### 1. Connect Repository
```bash
# In your Railway dashboard:
# 1. Click "New Project"
# 2. Select "Deploy from GitHub repo"
# 3. Choose this repository
# 4. Set root directory to: backend/modomo-scraper
```

### 2. Environment Variables
Set these environment variables in Railway:

**Database (Supabase):**
```
DATABASE_URL_CLOUD=postgresql://postgres:qYs8XprwrHQoM7n6@db.nyeeewcpexqsqfzzmvyu.supabase.co:5432/postgres
```

**Storage (Cloudflare R2):**
```
CLOUDFLARE_R2_ENDPOINT=https://9bbdb3861142e65685d23f4955b88ebe.r2.cloudflarestorage.com
CLOUDFLARE_R2_BUCKET=reroom
CLOUDFLARE_R2_ACCESS_KEY_ID=6c8abdff2cdad89323e36b258b1d0f4b
CLOUDFLARE_R2_SECRET_ACCESS_KEY=35b7b4a1f586211d407b246b54898d2e50d13562cba7e7be6293d4b6ccea06c5
CLOUDFLARE_R2_PUBLIC_URL=https://photos.reroom.app
```

**Redis (Optional - Add Redis plugin in Railway):**
```
REDIS_URL=redis://default:password@redis-service:6379
```

**API Configuration:**
```
CORS_ORIGINS=https://your-dashboard.pages.dev,http://localhost:3001
DEBUG=false
LOG_LEVEL=info
```

### 3. Deploy
Railway will automatically:
- Build the Docker container
- Deploy the service
- Provide a public URL
- Handle SSL/HTTPS

## 🗄️ Database Setup

### Option 1: Initialize via Railway Console
After deployment, run this in Railway's console:
```bash
python -c "
import asyncio
import os
import asyncpg

async def init_db():
    conn = await asyncpg.connect(os.getenv('DATABASE_URL_CLOUD'))
    with open('database/schema.sql', 'r') as f:
        await conn.execute(f.read())
    print('✅ Database initialized')
    await conn.close()

asyncio.run(init_db())
"
```

### Option 2: Initialize via Supabase SQL Editor
1. Go to your Supabase dashboard
2. Open SQL Editor
3. Copy the contents of `database/schema.sql`
4. Execute the schema

## 🔧 Service Configuration

The service will automatically:
- **Port**: Railway assigns PORT environment variable
- **Health Check**: Available at `/health`
- **API Docs**: Available at `/docs`
- **Database**: Auto-connects to Supabase
- **Storage**: Auto-connects to Cloudflare R2

## 📊 Monitoring

Health check endpoints:
- `GET /health` - Service health
- `GET /stats/dataset` - Dataset statistics
- `GET /jobs/active` - Active background jobs

## 🌐 CORS Configuration

Set `CORS_ORIGINS` to include your dashboard URL:
```
CORS_ORIGINS=https://modomo-dashboard.pages.dev,http://localhost:3001
```

## 🎯 Production Considerations

1. **Database Connection Pooling**: Already configured in the app
2. **Background Jobs**: Uses Redis for job queuing
3. **File Storage**: All files stored in Cloudflare R2
4. **Monitoring**: Health checks and logging enabled
5. **Security**: CORS properly configured

## 📝 Environment Variables Reference

| Variable | Required | Description |
|----------|----------|-------------|
| `DATABASE_URL_CLOUD` | ✅ | Supabase PostgreSQL connection string |
| `CLOUDFLARE_R2_ENDPOINT` | ✅ | R2 storage endpoint |
| `CLOUDFLARE_R2_BUCKET` | ✅ | R2 bucket name |
| `CLOUDFLARE_R2_ACCESS_KEY_ID` | ✅ | R2 access key |
| `CLOUDFLARE_R2_SECRET_ACCESS_KEY` | ✅ | R2 secret key |
| `REDIS_URL` | ⚠️ | Redis for background jobs (optional) |
| `CORS_ORIGINS` | ⚠️ | Allowed origins for CORS |
| `RUNPOD_API_KEY` | ❌ | For AI processing (optional) |

## 🚀 Expected Railway URL
After deployment, your API will be available at:
```
https://your-service-name-production-xxxx.up.railway.app
```

Update your dashboard's API_BASE_URL to point to this Railway URL.