"""
Celery configuration for Modomo background tasks
"""
import os
from celery import Celery
from config.settings import settings

# Create Celery app
celery_app = Celery(
    "modomo_tasks",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=[
        "tasks.color_tasks",
        "tasks.detection_tasks", 
        "tasks.scraping_tasks",
        "tasks.import_tasks",
        "tasks.classification_tasks"
    ]
)

# Celery configuration
celery_app.conf.update(
    # Task serialization
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    
    # Timezone
    timezone="UTC",
    enable_utc=True,
    
    # Task tracking
    task_track_started=True,
    task_send_sent_event=True,
    
    # Performance settings
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_disable_rate_limits=False,
    
    # Memory optimization
    worker_max_tasks_per_child=100,  # Restart worker after 100 tasks to prevent memory leaks
    worker_max_memory_per_child=400000,  # Restart worker if memory exceeds 400MB
    
    # Rate limiting
    task_annotations={
        "*": {"rate_limit": "10/s"},
        "tasks.color_tasks.run_color_processing_job": {"rate_limit": "5/s"},
        "tasks.detection_tasks.run_detection_pipeline": {"rate_limit": "3/s"},
        "tasks.scraping_tasks.run_scraping_job": {"rate_limit": "2/s"},
    },
    
    # Retry settings
    task_default_retry_delay=60,
    task_max_retries=3,
    
    # Result expiration
    result_expires=3600,  # 1 hour
    
    # Worker settings
    worker_send_task_events=True,
    task_send_events=True,
    
    # Queue routing
    task_routes={
        "tasks.color_tasks.*": {"queue": "color_processing"},
        "tasks.detection_tasks.*": {"queue": "ai_processing"},
        "tasks.scraping_tasks.*": {"queue": "scraping"},
        "tasks.import_tasks.*": {"queue": "import"},
        "tasks.classification_tasks.*": {"queue": "classification"},
    },
    
    # Error handling
    task_reject_on_worker_lost=True,
    task_ignore_result=False,
)

# Beat schedule for periodic tasks (optional)
celery_app.conf.beat_schedule = {
    "cleanup-failed-jobs": {
        "task": "tasks.maintenance_tasks.cleanup_failed_jobs",
        "schedule": 3600.0,  # Every hour
    },
    "health-check": {
        "task": "tasks.maintenance_tasks.health_check",
        "schedule": 300.0,  # Every 5 minutes
    },
}

if __name__ == "__main__":
    celery_app.start()