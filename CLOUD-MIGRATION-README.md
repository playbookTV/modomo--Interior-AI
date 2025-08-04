# 🚀 ReRoom Cloud Migration Guide

This guide will migrate your ReRoom app from local Docker infrastructure to a **best-of-breed cloud architecture** using Supabase, Clerk, Cloudflare R2, and RunPod.

## 🎯 Migration Overview

### **From: Local Docker Stack**
```
Current Architecture:
├── 🗄️ PostgreSQL (localhost:5432)
├── 📦 Redis (localhost:6379)
├── 🗂️ MinIO (localhost:9000)
├── 📸 Photo Service (localhost:3002)
├── 🧠 AI Service (localhost:8000)
└── 📱 Mobile App (localhost:8081)
```

### **To: Cloud-Native Architecture**
```
New Architecture:
├── 🗄️ Supabase (PostgreSQL + Real-time)
├── 🔐 Clerk (Authentication + User Management)
├── ☁️ Cloudflare R2 (S3-Compatible Storage + CDN)
├── 🧠 RunPod (GPU AI Processing)
├── 🚀 Railway (Backend Hosting)
└── 📱 Mobile App (Cloud Integrated)
```

## 💰 Cost Comparison

| Service | Current (Local) | New (Cloud) | Monthly Cost |
|---------|----------------|-------------|--------------|
| Database | Docker PostgreSQL | Supabase Pro | $25 |
| Storage | Docker MinIO | Cloudflare R2 | $15 |
| Auth | Local JWT | Clerk Pro | $25 |
| AI Processing | Local Python | RunPod GPU | $200-350 |
| Backend Hosting | Local | Railway | $20-50 |
| **Total** | **$0 (local only)** | **$285-465** | **Production Ready** |

## 🛠️ Prerequisites

Before starting the migration, ensure you have:

1. **Supabase Account**: [Sign up](https://supabase.com)
2. **Clerk Account**: [Sign up](https://clerk.com)
3. **Cloudflare Account**: [Sign up](https://cloudflare.com)
4. **RunPod Account**: [Sign up](https://runpod.io)
5. **Railway Account**: [Sign up](https://railway.app)

## 📋 Step-by-Step Migration

### **Phase 1: Get Your API Keys**

#### 1.1 Supabase Setup
1. Go to [Supabase Dashboard](https://supabase.com/dashboard)
2. Create a new project or use existing: `nyeeewcpexqsqfzzmvyu`
3. Go to **Settings → API**
4. Copy these values:
   - **URL**: `https://nyeeewcpexqsqfzzmvyu.supabase.co` ✅
   - **anon public key**: `[YOUR_SUPABASE_ANON_KEY]`
   - **service_role key**: `[YOUR_SUPABASE_SERVICE_ROLE_KEY]`

#### 1.2 Clerk Setup  
1. Go to [Clerk Dashboard](https://dashboard.clerk.com)
2. Your app is already configured with publishable key: `pk_test_aW5jbHVkZWQtaGVyb24tNTMuY2xlcmsuYWNjb3VudHMuZGV2JA` ✅
3. Get these additional keys:
   - **Secret Key**: `[YOUR_CLERK_SECRET_KEY]`
   - **Webhook Secret**: `[YOUR_CLERK_WEBHOOK_SECRET]`

#### 1.3 Cloudflare R2 Setup
1. Go to [Cloudflare Dashboard](https://dash.cloudflare.com)
2. Your bucket is already configured: `https://9bbdb3861142e65685d23f4955b88ebe.r2.cloudflarestorage.com/reroom` ✅
3. Go to **R2 → Manage R2 API Tokens**
4. Create a new token with R2:Object:Edit permissions
5. Copy these values:
   - **Access Key ID**: `[YOUR_R2_ACCESS_KEY]`
   - **Secret Access Key**: `[YOUR_R2_SECRET_KEY]`

#### 1.4 RunPod Setup
1. Go to [RunPod Console](https://www.runpod.io/console)
2. Your endpoint is already configured: `https://api.runpod.ai/v2/m6672a33qqoau2/run` ✅
3. Go to **Settings → API Keys**
4. Create a new API key
5. Copy: **API Key**: `[YOUR_RUNPOD_API_KEY]`

### **Phase 2: Update Environment Configuration**

Your `.env` file has been updated with the cloud configuration section. **Replace the placeholder values** with your actual API keys:

```bash
# Open your .env file and update these values:

# Supabase Database
SUPABASE_ANON_KEY=[YOUR_SUPABASE_ANON_KEY]
SUPABASE_SERVICE_ROLE_KEY=[YOUR_SUPABASE_SERVICE_ROLE_KEY]

# Clerk Authentication  
CLERK_SECRET_KEY=[YOUR_CLERK_SECRET_KEY]
CLERK_WEBHOOK_SECRET=[YOUR_CLERK_WEBHOOK_SECRET]

# Cloudflare R2 Storage
CLOUDFLARE_R2_ACCESS_KEY_ID=[YOUR_R2_ACCESS_KEY]
CLOUDFLARE_R2_SECRET_ACCESS_KEY=[YOUR_R2_SECRET_KEY]

# RunPod AI Service
RUNPOD_API_KEY=[YOUR_RUNPOD_API_KEY]
```

### **Phase 3: Run Automated Setup**

Once you've updated the `.env` file with your API keys, run the automated setup script:

```bash
# Run the cloud migration setup
./scripts/setup-cloud-migration.sh
```

This script will:
- ✅ Validate all your API keys
- ✅ Create the Supabase database schema
- ✅ Install backend dependencies
- ✅ Build and test the backend service
- ✅ Install mobile app dependencies
- ✅ Test cloud service connections

### **Phase 4: Deploy to Railway**

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login to Railway
railway login

# Navigate to backend
cd backend

# Link to your Railway project (already configured: reroom-production-dcb0.up.railway.app)
railway link

# Set environment variables in Railway
railway variables set NODE_ENV=production
railway variables set PORT=6969
railway variables set SUPABASE_URL=https://nyeeewcpexqsqfzzmvyu.supabase.co
railway variables set SUPABASE_ANON_KEY=[YOUR_SUPABASE_ANON_KEY]
railway variables set SUPABASE_SERVICE_ROLE_KEY=[YOUR_SUPABASE_SERVICE_ROLE_KEY]
railway variables set CLERK_SECRET_KEY=[YOUR_CLERK_SECRET_KEY]
railway variables set CLOUDFLARE_R2_ENDPOINT=https://9bbdb3861142e65685d23f4955b88ebe.r2.cloudflarestorage.com
railway variables set CLOUDFLARE_R2_BUCKET=reroom
railway variables set CLOUDFLARE_R2_ACCESS_KEY_ID=[YOUR_R2_ACCESS_KEY]
railway variables set CLOUDFLARE_R2_SECRET_ACCESS_KEY=[YOUR_R2_SECRET_KEY]
railway variables set RUNPOD_API_KEY=[YOUR_RUNPOD_API_KEY]
railway variables set RUNPOD_ENDPOINT=https://api.runpod.ai/v2/m6672a33qqoau2/run
railway variables set RAILWAY_BACKEND_URL=https://reroom-production-dcb0.up.railway.app:6969

# Deploy to Railway
railway up
```

### **Phase 5: Test Complete Pipeline**

```bash
# 1. Test backend health
curl https://reroom-production-dcb0.up.railway.app:6969/health

# 2. Start mobile app with cloud integration
cd mobile
npx expo start

# 3. Take a photo in the app and verify:
#    - Photo uploads to Cloudflare R2
#    - Metadata saves to Supabase
#    - AI makeover triggers on RunPod
#    - Real-time updates via Supabase
```

## 🔧 Key Files Created/Updated

### **Database Migration**
- `database/supabase-migration.sql` - Complete schema migration script

### **Backend Services**
- `backend/src/services/cloudflareR2.ts` - Cloudflare R2 integration
- `backend/src/services/supabaseService.ts` - Supabase database service
- `backend/src/services/runpodService.ts` - RunPod AI integration
- `backend/src/routes/cloudPhotos.ts` - Cloud photo API routes
- `backend/src/routes/makeovers.ts` - AI makeover management
- `backend/src/server.ts` - Railway backend server

### **Mobile App Integration**
- `mobile/app/_layout.tsx` - Clerk authentication wrapper
- `mobile/src/services/cloudPhotoService.ts` - Cloud photo service

### **Configuration Files**
- `backend/package.json` - Updated dependencies
- `backend/tsconfig.json` - TypeScript configuration
- `backend/railway.json` - Railway deployment config

## 📊 Service Health Monitoring

Once deployed, monitor your services:

### **Health Endpoints**
- **Backend**: `https://reroom-production-dcb0.up.railway.app:6969/health`
- **Supabase**: [Dashboard](https://supabase.com/dashboard/project/nyeeewcpexqsqfzzmvyu)
- **Railway**: [Dashboard](https://railway.app/dashboard)
- **Cloudflare**: [R2 Dashboard](https://dash.cloudflare.com)

### **Real-time Monitoring**
```bash
# Watch backend logs
railway logs

# Monitor Supabase real-time
# Go to Supabase Dashboard → Logs

# Check Cloudflare R2 usage
# Go to Cloudflare Dashboard → R2 → Analytics
```

## 🚨 Troubleshooting

### **Common Issues & Solutions**

#### Backend Won't Start
```bash
# Check environment variables
railway variables

# Check logs
railway logs

# Restart service
railway restart
```

#### Database Connection Issues
```bash
# Test Supabase connection
curl -H "apikey: YOUR_ANON_KEY" "https://nyeeewcpexqsqfzzmvyu.supabase.co/rest/v1/users"

# Check RLS policies in Supabase dashboard
```

#### Photo Upload Failures
```bash
# Test Cloudflare R2 access
# Check your R2 API token permissions

# Verify CORS settings in R2 dashboard
```

#### AI Processing Not Working
```bash
# Check RunPod endpoint status
curl -H "Authorization: Bearer YOUR_API_KEY" "https://api.runpod.ai/v2/m6672a33qqoau2/status"

# Verify webhook callback URL is accessible
```

## 🎉 Success Metrics

Your migration is successful when:

✅ **Backend Health**: All services report "healthy" status  
✅ **Photo Upload**: Photos successfully upload to Cloudflare R2  
✅ **Database**: Metadata correctly saves to Supabase  
✅ **Authentication**: Users can sign in via Clerk  
✅ **AI Processing**: RunPod jobs complete successfully  
✅ **Real-time Updates**: Mobile app receives live makeover progress  

## 🚀 Go Live Checklist

Before switching to production:

- [ ] All API keys are production-ready (not test keys)
- [ ] Domain configured for Railway backend
- [ ] SSL certificates enabled
- [ ] Rate limiting configured appropriately
- [ ] Monitoring and alerting set up
- [ ] Backup strategy implemented
- [ ] Error logging configured (Sentry recommended)

## 📞 Support

Need help with the migration?

- **Technical Issues**: Check service dashboards and logs
- **API Problems**: Verify your API keys and permissions
- **Performance**: Monitor usage metrics in each service dashboard

---

**🎯 This migration transforms ReRoom from a local development setup to a globally-scalable, production-ready AI interior design platform!**