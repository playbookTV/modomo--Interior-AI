# Railway Celery Multi-Service Deployment Guide

## Problem Statement

Railway doesn't support Docker Compose, but Celery requires background worker processes to consume job queues. The current deployment only runs the FastAPI server, leaving Celery jobs stuck in "pending" status.

## Solution: Multiple Railway Services

Deploy **separate Railway services** within the same project:
1. **Main Service**: FastAPI application server
2. **Worker Service**: Celery background workers
3. **Redis Service**: Railway Redis addon (shared between services)

## Current Status

### ✅ Working Components:
- FastAPI application deployed and running
- DatasetImporter frontend creating jobs successfully
- Redis queue receiving jobs
- Database tracking job status

### ❌ Missing Component:
- **Celery workers not running** → Jobs stuck in "pending" status

## Deployment Architecture

```
Railway Project: modomo-scraper
├── Service 1: FastAPI App (main)
│   ├── Start Command: python main_railway.py
│   ├── Config: railway.toml
│   └── URL: https://ovalay-recruitment-production.up.railway.app
├── Service 2: Celery Worker (new)
│   ├── Start Command: celery -A celery_app worker --loglevel=info -Q import,scraping
│   ├── Config: railway-import-worker.toml
│   └── Purpose: Process import/scraping jobs
├── Service 3: AI Worker (optional)
│   ├── Start Command: celery -A celery_app worker --loglevel=info -Q ai_processing,detection
│   ├── Config: railway-worker.toml  
│   └── Purpose: Process AI-intensive tasks
└── Redis Addon (shared)
    └── Provides: Message broker + result backend
```

## Setup Instructions

### Step 1: Create Import Worker Service

1. **In Railway Dashboard:**
   - Go to your `modomo-scraper` project
   - Click "New Service" → "GitHub Repo"
   - Connect the same repository (`modomo/backend/modomo-scraper`)

2. **Configure Worker Service:**
   - **Service Name**: `celery-import-worker`
   - **Root Directory**: `backend/modomo-scraper`
   - **Start Command**: `celery -A celery_app worker --loglevel=info --concurrency=1 -Q import,scraping`

3. **Environment Variables:**
   Copy these from your main service:
   ```
   REDIS_URL=<your_redis_url>
   SUPABASE_URL=<your_supabase_url>
   SUPABASE_ANON_KEY=<your_supabase_key>
   AI_MODE=basic
   PYTHONUNBUFFERED=1
   ```

### Step 2: Deploy Configuration Files

Use the provided configuration files:

- **`railway-import-worker.toml`**: Optimized for import/scraping tasks
- **`railway-worker.toml`**: Full AI worker with all queues
- **`railway.toml`**: Current main application config

### Step 3: Verify Deployment

1. **Check Worker Logs:**
   ```
   [INFO] Connected to redis://...
   [INFO] mingle: searching for neighbor nodes...
   [INFO] celery@... ready.
   ```

2. **Test Job Processing:**
   - Submit a new import job via DatasetImporter
   - Watch job status change from "pending" → "running" → "completed"

3. **Monitor via API:**
   ```bash
   curl https://ovalay-recruitment-production.up.railway.app/jobs/active
   ```

## Service Configurations

### Import Worker Service (Recommended)
```toml
# railway-import-worker.toml
[deploy]
startCommand = "celery -A celery_app worker --loglevel=info --concurrency=1 -Q import,scraping"
restartPolicyType = "on_failure"
restartPolicyMaxRetries = 10
```

**Purpose**: Specifically handles dataset import and scraping jobs
**Queues**: `import`, `scraping`
**Resources**: Lightweight, optimized for I/O tasks

### Full AI Worker Service (Optional)
```toml
# railway-worker.toml  
[deploy]
startCommand = "celery -A celery_app worker --loglevel=info --concurrency=2 -Q ai_processing,detection,color_processing,classification"
```

**Purpose**: Handles AI-intensive tasks (YOLO, SAM2, embeddings)
**Queues**: `ai_processing`, `detection`, `color_processing`, `classification`
**Resources**: Higher memory/CPU for AI models

## Environment Variables Setup

### Shared Variables (Required for both services):
```bash
REDIS_URL=<railway_redis_url>
SUPABASE_URL=<your_supabase_url>
SUPABASE_ANON_KEY=<your_supabase_key>
PYTHONUNBUFFERED=1
PYTHONDONTWRITEBYTECODE=1
```

### Worker-Specific Variables:
```bash
# Import Worker
AI_MODE=basic
CELERY_WORKER_CONCURRENCY=1

# AI Worker  
AI_MODE=full
CLASSIFICATION_ENHANCED=true
MODEL_CACHE_DIR=/app/models
SAM2_CHECKPOINT_DIR=/app/models/sam2
```

## Queue Assignment Strategy

### Current Job Distribution:
- **`import` queue**: HuggingFace dataset imports → Import Worker
- **`scraping` queue**: Houzz web scraping → Import Worker  
- **`ai_processing` queue**: YOLO/SAM2 detection → AI Worker
- **`detection` queue**: Object detection → AI Worker
- **`color_processing` queue**: Color analysis → AI Worker
- **`classification` queue**: Scene classification → AI Worker

### Queue Routing (from `celery_app.py`):
```python
task_routes = {
    "tasks.import_tasks.*": {"queue": "import"},
    "tasks.scraping_tasks.*": {"queue": "scraping"},
    "tasks.detection_tasks.*": {"queue": "ai_processing"},
    "tasks.color_tasks.*": {"queue": "color_processing"},
    "tasks.classification_tasks.*": {"queue": "classification"},
}
```

## Troubleshooting

### Common Issues:

1. **Jobs Still Pending After Worker Deployment**
   ```bash
   # Check worker logs in Railway dashboard
   # Verify REDIS_URL is correctly set
   # Ensure worker is consuming correct queues
   ```

2. **Worker Service Fails to Start**
   ```bash
   # Check environment variables are set
   # Verify celery_app.py can be imported
   # Check Dockerfile builds successfully
   ```

3. **Memory Issues with AI Worker**
   ```bash
   # Reduce concurrency to 1
   # Set AI_MODE=basic for testing
   # Monitor Railway metrics
   ```

### Debug Commands:

```bash
# Check active Celery workers
celery -A celery_app inspect active

# Check queue lengths  
celery -A celery_app inspect reserved

# Monitor worker stats
celery -A celery_app inspect stats
```

## Monitoring

### Railway Dashboard:
- Monitor CPU/Memory usage per service
- Check deployment logs for errors
- Track service health and restarts

### API Endpoints:
- `GET /jobs/active` - Active job status
- `GET /jobs/history` - Job history
- `GET /health` - System health check

### Current Pending Jobs:
```bash
# These jobs are waiting for workers:
Job ID: 0e4e301c-5920-41e2-9c14-0a82d50cde00 (50 images)
Job ID: ca9937d1-d1bb-4e13-92c2-9b6b18c8448a (50 images)  
Job ID: 8d0018d1-4b72-44ac-87bd-8105b39846a6 (50 images)
Job ID: a0a78bbc-ef60-41bb-9aca-1a099c820485 (1 image - test)
```

## Expected Results

After deploying the import worker service:

1. **Immediate**: Pending jobs should start processing
2. **DatasetImporter**: Real-time progress bars will work
3. **Job Status**: Transitions from "pending" → "running" → "completed"
4. **Dataset Growth**: New scenes/objects will appear in database
5. **Review Queue**: Processed scenes available for review

## Next Steps

1. **Deploy Import Worker**: Start with `railway-import-worker.toml`
2. **Monitor Initial Jobs**: Watch the 4 pending jobs get processed
3. **Test New Imports**: Use DatasetImporter to verify end-to-end workflow
4. **Scale if Needed**: Add AI worker service for heavy processing

## Cost Considerations

- **Import Worker**: ~$5-10/month (lightweight service)
- **AI Worker**: ~$20-30/month (higher resources for AI models)
- **Total**: Minimal increase for significant functionality gain

The multiple service approach is the **standard Railway pattern** for background task processing and provides better resource isolation and scaling flexibility.