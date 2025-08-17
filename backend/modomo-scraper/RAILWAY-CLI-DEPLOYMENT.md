# Railway CLI Deployment Guide for Celery Workers

## Quick Setup with Railway CLI

Since you use `railway up` instead of GitHub integration, here's the streamlined CLI approach:

### Step 1: Deploy Import Worker Service

```bash
# Navigate to the modomo-scraper directory
cd /Users/leslieisah/modomo/backend/modomo-scraper

# Create a new Railway service for the import worker
railway service create celery-import-worker

# Switch to the new service
railway service use celery-import-worker

# Set the configuration file for this service
cp railway-import-worker.toml railway.toml

# Deploy the worker service
railway up
```

### Step 2: Configure Environment Variables

```bash
# Copy environment variables from your main service
# Get them from your main service first:
railway service use modomo-scraper-main
railway variables

# Switch back to worker service and set variables:
railway service use celery-import-worker
railway variables set REDIS_URL="<your_redis_url>"
railway variables set SUPABASE_URL="<your_supabase_url>"
railway variables set SUPABASE_ANON_KEY="<your_supabase_key>"
railway variables set AI_MODE="basic"
railway variables set PYTHONUNBUFFERED="1"
```

### Step 3: Deploy and Monitor

```bash
# Deploy with the correct configuration
railway up

# Monitor the deployment
railway logs

# Check if worker is consuming jobs
railway logs --tail
```

## Expected Log Output

When successful, you should see:
```
[INFO] Connected to redis://...
[INFO] mingle: searching for neighbor nodes...
[INFO] celery@... ready.
[INFO] Consuming from queues: import, scraping
```

## Verify Jobs Are Processing

1. **Check job status via API:**
   ```bash
   curl https://ovalay-recruitment-production.up.railway.app/jobs/active
   ```

2. **Monitor pending jobs:**
   The 4 pending import jobs should start processing immediately.

## Service Management

```bash
# List all services in your project
railway service list

# Switch between services
railway service use modomo-scraper-main      # Main FastAPI app
railway service use celery-import-worker     # Import worker

# View logs for specific service
railway logs --service celery-import-worker

# Restart a service
railway service restart celery-import-worker
```

## Configuration Files Used

- **`railway-import-worker.toml`** - Optimized for import/scraping tasks
- **`railway-worker.toml`** - Full AI worker (optional, for later)
- **`railway.toml`** - Main FastAPI application

## Quick Commands Summary

```bash
# 1. Create and deploy worker service
railway service create celery-import-worker
railway service use celery-import-worker
cp railway-import-worker.toml railway.toml
railway up

# 2. Set environment variables (copy from main service)
railway variables set REDIS_URL="<value>"
railway variables set SUPABASE_URL="<value>"
railway variables set SUPABASE_ANON_KEY="<value>"

# 3. Monitor deployment
railway logs --tail
```

## Troubleshooting

**If deployment fails:**
```bash
# Check build logs
railway logs --build

# Verify environment variables
railway variables

# Check service status
railway status
```

**If worker doesn't start:**
```bash
# Check runtime logs
railway logs --runtime

# Verify Redis connection
railway variables | grep REDIS_URL
```

This approach will get your Celery import worker running to process the 4 pending jobs (151 total images) that are currently stuck.