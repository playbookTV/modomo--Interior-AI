# PyTorch Dependency Fix for Production Worker

## Issue
Production Heroku worker was failing with `No module named 'torch'` error when trying to run AI detection tasks.

## Root Cause
The `modomo-scraper` worker has minimal dependencies (no PyTorch) but was configured to handle AI detection tasks that require heavy AI libraries.

## Solution
1. **Graceful Dependency Handling**: Modified `detection_tasks.py` to check for PyTorch availability and gracefully skip AI tasks when not available
2. **Queue Separation**: Removed `ai_processing` queue from lightweight worker Procfile
3. **Memory Optimization**: Added Celery memory limits to prevent quota exceeded errors

## Changes Made

### 1. Detection Tasks (`tasks/detection_tasks.py`)
- Added PyTorch import checks in `get_detection_service()`
- Modified all detection functions to return "skipped" status instead of failing
- Tasks now complete successfully with appropriate messaging

### 2. Worker Configuration (`Procfile`)
```bash
# Before
worker: celery -A celery_app worker --loglevel=info --concurrency=1 -Q import,scraping,ai_processing,color_processing,classification

# After 
worker: celery -A celery_app worker --loglevel=info --concurrency=1 -Q import,scraping,color_processing,classification
```

### 3. Memory Optimization (`celery_app.py`)
```python
# Added memory limits
worker_max_tasks_per_child=100,  # Restart after 100 tasks
worker_max_memory_per_child=400000,  # Restart if memory exceeds 400MB
```

## Deployment
The lightweight worker now:
- ✅ Handles import/scraping/color/classification tasks
- ⏭️ Skips AI detection tasks gracefully  
- 🔄 Automatically restarts to prevent memory leaks
- 📊 Stays within 512MB memory quota

## AI Processing
AI detection tasks should be handled by a separate worker with full AI dependencies:
- Full `ai-service/requirements.txt` with PyTorch
- Higher memory allocation (2GB+)
- Only handles `ai_processing` queue

## Testing
Deploy and monitor logs - should see:
```
[WARNING] PyTorch not available - AI detection disabled in this worker
[WARNING] AI detection service not available - skipping detection task
```
Instead of crashes and retry loops.