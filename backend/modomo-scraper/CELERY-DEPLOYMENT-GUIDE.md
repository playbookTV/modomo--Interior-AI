# Celery Implementation Guide - Modomo Dataset Scraper

## Overview

This guide documents the complete Celery implementation for the Modomo Dataset Scraping System. Celery provides distributed task queue capabilities for handling background processing of AI detection, color analysis, scraping, and classification tasks.

## Architecture

### Core Components

1. **Celery App** (`celery_app.py`) - Main Celery application configuration
2. **Task Modules** (`tasks/`) - Organized task implementations by domain
3. **FastAPI Integration** (`main_refactored.py`) - API endpoints with Celery integration
4. **Monitoring & Management** - Comprehensive monitoring endpoints and Flower UI
5. **Docker Infrastructure** (`docker-compose.celery.yml`) - Production deployment setup

### Task Organization

```
tasks/
├── __init__.py              # Base task class and common utilities
├── color_tasks.py           # Color processing and analysis
├── detection_tasks.py       # AI detection pipeline (YOLO, SAM2, etc.)
├── scraping_tasks.py        # Houzz scraping and data collection
├── classification_tasks.py  # Image classification and scene analysis
└── import_tasks.py          # HuggingFace dataset imports
```

## Installation & Setup

### 1. Install Dependencies

```bash
# Install Celery and related packages
pip install -r requirements-celery.txt

# Core Celery dependencies included:
# - celery[redis]==5.3.4
# - flower==2.0.1
# - celery-beat==2.5.0
# - datasets>=2.14.0,<2.19.0
# - websockets>=10.0,<12.0

# Alternative: Minimal installation (avoids potential gradio conflicts)
pip install -r requirements-minimal-celery.txt
```

#### Dependency Conflict Resolution

If you encounter websockets/gradio-client conflicts:

```bash
# Option 1: Use minimal requirements (no HuggingFace datasets)
pip install -r requirements-minimal-celery.txt

# Option 2: Install core Celery only
pip install "celery[redis]>=5.3.0" flower redis supabase "websockets>=10.0,<12.0"

# Option 3: Force compatible versions
pip install "websockets>=11.0,<12.0" --force-reinstall
```

### 2. Environment Configuration

Ensure these environment variables are set:

```bash
# Required
REDIS_URL=redis://localhost:6379/0
SUPABASE_URL=your_supabase_url
SUPABASE_ANON_KEY=your_supabase_key

# Optional
CELERY_BROKER_URL=redis://localhost:6379/0  # Defaults to REDIS_URL
CELERY_RESULT_BACKEND=redis://localhost:6379/0  # Defaults to REDIS_URL
```

### 3. Redis Setup

#### Local Development
```bash
# Install and start Redis
brew install redis  # macOS
sudo apt-get install redis-server  # Ubuntu

# Start Redis
redis-server
```

#### Docker
```bash
# Start Redis with Docker
docker run -d -p 6379:6379 redis:7-alpine
```

## Running Celery

### Development Mode

#### Start Workers
```bash
# General worker (handles all tasks)
celery -A celery_app worker --loglevel=info --concurrency=2

# Specialized workers
celery -A celery_app worker --loglevel=info --concurrency=1 -Q ai_processing,detection
celery -A celery_app worker --loglevel=info --concurrency=1 -Q scraping,import
celery -A celery_app worker --loglevel=info --concurrency=2 -Q color_processing,classification
```

#### Start Beat Scheduler (for periodic tasks)
```bash
celery -A celery_app beat --loglevel=info
```

#### Start Flower Monitoring
```bash
celery -A celery_app flower --port=5555
# Access at http://localhost:5555
```

### Production Mode (Docker)

#### Start All Services
```bash
docker-compose -f docker-compose.celery.yml up -d
```

#### Scale Workers
```bash
# Scale AI workers for heavy processing
docker-compose -f docker-compose.celery.yml up -d --scale celery-ai-worker=3

# Scale general workers
docker-compose -f docker-compose.celery.yml up -d --scale celery-worker=2
```

## Task Types & Usage

### 1. Color Processing Tasks

#### Bulk Color Processing
```python
# API: POST /process/colors?limit=100
from tasks.color_tasks import run_color_processing_job

# Programmatic usage
task = run_color_processing_job.delay(job_id="color_001", limit=100)
result = task.get()  # Blocks until complete
```

#### Single Object Color Analysis
```python
from tasks.color_tasks import process_single_object_colors

task = process_single_object_colors.delay(
    object_id="obj_123",
    image_url="https://example.com/image.jpg"
)
```

### 2. AI Detection Pipeline

#### Complete Detection Pipeline
```python
# API: POST /detect/process?image_url=...
from tasks.detection_tasks import run_detection_pipeline

task = run_detection_pipeline.delay(
    job_id="detect_001",
    image_url="https://example.com/room.jpg",
    scene_id="scene_123"
)
```

#### Batch Scene Processing
```python
from tasks.detection_tasks import process_scenes_batch

task = process_scenes_batch.delay(
    job_id="batch_001",
    scene_ids=["scene_1", "scene_2", "scene_3"]
)
```

### 3. Scraping Tasks

#### Houzz Scraping Job
```python
# API: POST /scraper/start?limit=50
from tasks.scraping_tasks import run_scraping_job

task = run_scraping_job.delay(
    job_id="scrape_001",
    limit=50,
    room_types=["living_room", "bedroom"]
)
```

#### HuggingFace Dataset Import
```python
# API: POST /import/huggingface-dataset
from tasks.scraping_tasks import import_huggingface_dataset

task = import_huggingface_dataset.delay(
    job_id="import_001",
    dataset="modomo/interior-design",
    offset=0,
    limit=1000,
    include_detection=True
)
```

### 4. Classification Tasks

#### Scene Reclassification
```python
# API: POST /classify/reclassify?limit=100
from tasks.classification_tasks import run_scene_reclassification_job

task = run_scene_reclassification_job.delay(
    job_id="classify_001",
    limit=100,
    force_reclassify=False
)
```

#### Single Scene Classification
```python
from tasks.classification_tasks import classify_single_scene

task = classify_single_scene.delay(
    scene_id="scene_123",
    image_url="https://example.com/room.jpg"
)
```

## Monitoring & Management

### API Endpoints

#### Job Status & Progress
```bash
# Get job status with Celery task info
GET /jobs/{job_id}/celery-status

# Get all active Celery tasks
GET /jobs/celery/active-tasks

# Cancel a running job
POST /jobs/{job_id}/cancel
```

#### Worker Monitoring
```bash
# Get worker status and statistics
GET /jobs/celery/workers

# Get queue information and lengths
GET /jobs/celery/queues

# Comprehensive monitoring dashboard
GET /jobs/celery/dashboard
```

#### Queue Management
```bash
# Purge a specific queue (admin only)
POST /jobs/celery/purge-queue/{queue_name}
```

### Flower Web Interface

Access the Flower monitoring interface at:
- Development: http://localhost:5555
- Production: http://your-domain:5555

#### Features Available in Flower:
- Real-time task monitoring
- Worker status and statistics
- Queue length monitoring
- Task history and results
- Worker control (restart, shutdown)
- Task routing visualization

### Progress Tracking

All long-running tasks implement progress tracking:

```python
# Example progress update in a task
self.update_state(
    state='PROGRESS',
    meta={
        'current': 50,
        'total': 100,
        'progress': 50.0,
        'message': 'Processing images...'
    }
)
```

## Queue Configuration

### Queue Types
- **celery** (default): General tasks
- **ai_processing**: AI-intensive tasks (YOLO, SAM2, embeddings)
- **detection**: Object detection and segmentation
- **scraping**: Web scraping tasks
- **import**: Data import operations
- **color_processing**: Color analysis tasks
- **classification**: Image classification tasks

### Task Routing
Tasks are automatically routed to appropriate queues based on their type:

```python
# Example routing in celery_app.py
task_routes = {
    'tasks.detection_tasks.*': {'queue': 'ai_processing'},
    'tasks.color_tasks.*': {'queue': 'color_processing'},
    'tasks.scraping_tasks.*': {'queue': 'scraping'},
    'tasks.classification_tasks.*': {'queue': 'classification'},
}
```

## Error Handling & Retry Logic

### Automatic Retries
All tasks implement automatic retry with exponential backoff:

```python
@celery_app.task(
    bind=True,
    autoretry_for=(Exception,),
    retry_kwargs={'max_retries': 3, 'countdown': 60}
)
def my_task(self, ...):
    # Task implementation
    pass
```

### Error Recovery
- **Database connection failures**: Automatic retry with connection reset
- **External API failures**: Exponential backoff retry
- **Memory errors**: Task termination with cleanup
- **Network timeouts**: Configurable retry attempts

### Dead Letter Handling
Failed tasks are logged and can be inspected:

```bash
# View failed tasks in Flower
# Or check logs for error details
docker-compose -f docker-compose.celery.yml logs celery-worker
```

## Performance Optimization

### Worker Concurrency
- **AI Workers**: concurrency=1 (GPU/memory intensive)
- **General Workers**: concurrency=2-4 (CPU bound)
- **Scraping Workers**: concurrency=1 (rate limiting)

### Memory Management
```python
# Memory optimization settings in celery_app.py
worker_max_tasks_per_child = 100  # Restart worker after 100 tasks
worker_max_memory_per_child = 2000000  # 2GB memory limit
```

### Queue Prioritization
```python
# Priority routing for urgent tasks
task_routes = {
    'tasks.detection_tasks.process_urgent': {'queue': 'ai_processing', 'priority': 9},
    'tasks.color_tasks.process_batch': {'queue': 'color_processing', 'priority': 1},
}
```

## Troubleshooting

### Common Issues

#### Dependency Conflicts (websockets/gradio-client)

**Error**: `Cannot install gradio and websockets==12.0 because these package versions have conflicting dependencies`

**Solutions**:
```bash
# Solution 1: Use compatible websockets version
pip install "websockets>=11.0,<12.0"

# Solution 2: Use minimal requirements without HuggingFace
pip install -r requirements-minimal-celery.txt

# Solution 3: Remove conflicting packages
pip uninstall gradio gradio-client -y
pip install -r requirements-celery.txt

# Solution 4: Create clean virtual environment
python -m venv venv-celery
source venv-celery/bin/activate  # Linux/Mac
# or
venv-celery\Scripts\activate  # Windows
pip install -r requirements-minimal-celery.txt
```

#### Depth Estimation Model Conflicts

**Error**: `Cannot install depth-anything-v2 and huggingface-hub because of conflicting dependencies`

**Approach**: The system uses a fallback strategy:
1. **Primary**: Depth Anything V2 (if available)
2. **Fallback**: ZoeDepth (Railway-compatible)

**Post-deployment installation**:
```bash
# After successful deployment, optionally install Depth Anything V2
./install-depth-anything-v2.sh

# Manual installation
pip install git+https://github.com/badayvedat/Depth-Anything-V2.git@badayvedat-patch-1
```

**Note**: The application will work with ZoeDepth if Depth Anything V2 is unavailable.

#### Workers Not Starting
```bash
# Check Redis connection
redis-cli ping

# Check Celery configuration
celery -A celery_app inspect stats

# Check logs
docker-compose -f docker-compose.celery.yml logs celery-worker
```

#### Tasks Not Being Processed
```bash
# Check active workers
celery -A celery_app inspect active

# Check queue lengths
redis-cli llen celery
redis-cli llen ai_processing

# Purge stuck queues
celery -A celery_app purge
```

#### Memory Issues
```bash
# Monitor worker memory usage
docker stats

# Restart workers if needed
docker-compose -f docker-compose.celery.yml restart celery-ai-worker
```

### Debug Mode

Enable debug logging:
```python
# In celery_app.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Deployment Considerations

### Resource Requirements
- **Redis**: 512MB-1GB RAM minimum
- **General Workers**: 1-2GB RAM per worker
- **AI Workers**: 4-8GB RAM per worker (+ GPU if available)
- **Storage**: Sufficient disk space for task results and logs

### Scaling Guidelines
- Start with 1 AI worker, 2 general workers
- Scale AI workers based on detection queue length
- Scale general workers based on overall system load
- Monitor memory usage and adjust concurrency accordingly

### Security
- Use Redis AUTH in production
- Restrict Flower access (authentication required)
- Validate all task inputs
- Implement rate limiting for API endpoints

## Integration Examples

### FastAPI Integration
```python
# Example endpoint with Celery task
@app.post("/process-image")
async def process_image(image_url: str):
    job_id = generate_job_id()
    
    try:
        from tasks.detection_tasks import run_detection_pipeline
        task = run_detection_pipeline.delay(job_id, image_url)
        
        return {
            "job_id": job_id,
            "task_id": task.id,
            "status": "started",
            "message": "Processing started"
        }
    except ImportError:
        # Fallback to synchronous processing
        return await process_image_sync(image_url)
```

### Frontend Integration
```javascript
// Poll task status
async function pollTaskStatus(jobId) {
    const response = await fetch(`/jobs/${jobId}/celery-status`);
    const status = await response.json();
    
    if (status.job_data.status === 'completed') {
        return status;
    } else if (status.job_data.status === 'failed') {
        throw new Error(status.job_data.error_message);
    } else {
        // Continue polling
        setTimeout(() => pollTaskStatus(jobId), 2000);
    }
}
```

## Additional Resources

- [Celery Documentation](https://docs.celeryproject.org/)
- [Flower Documentation](https://flower.readthedocs.io/)
- [Redis Documentation](https://redis.io/documentation)
- [FastAPI Background Tasks](https://fastapi.tiangolo.com/tutorial/background-tasks/)

## Conclusion

This Celery implementation provides a robust, scalable background task processing system for the Modomo Dataset Scraper. It supports:

- ✅ Distributed task processing across multiple workers
- ✅ Real-time progress tracking and monitoring
- ✅ Automatic error handling and retry logic
- ✅ Queue-based task routing and prioritization
- ✅ Comprehensive monitoring and management tools
- ✅ Docker-based production deployment
- ✅ Graceful fallback for development environments

The system is ready for production use and can scale horizontally to handle increased workloads.