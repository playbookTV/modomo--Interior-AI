"""
Base task functionality for Modomo Celery tasks
"""
import structlog
from typing import Dict, Any, Optional
from celery import current_task
from celery.exceptions import Retry

# Import services
database_service = None
job_service = None

try:
    from supabase import create_client
    from config.settings import settings
    from services.database_service import DatabaseService
    from services.job_service import JobService
    
    # Initialize global services for tasks
    supabase_client = None
    
    # Initialize Supabase with error handling
    if settings.SUPABASE_URL and settings.SUPABASE_ANON_KEY:
        try:
            supabase_client = create_client(
                supabase_url=settings.SUPABASE_URL,
                supabase_key=settings.SUPABASE_ANON_KEY
            )
            database_service = DatabaseService(supabase_client)
            print(f"✅ Database service initialized successfully")
        except Exception as db_error:
            print(f"❌ Database service initialization failed: {db_error}")
            database_service = None
    else:
        print(f"⚠️  Missing Supabase configuration - database service disabled")
    
    # Initialize Redis/Job service
    try:
        import redis
        redis_client = redis.from_url(settings.REDIS_URL, socket_timeout=10)
        job_service = JobService(redis_client)
        print(f"✅ Job service initialized successfully")
    except Exception as redis_error:
        print(f"❌ Job service initialization failed: {redis_error}")
        job_service = JobService(None)
        
except ImportError as e:
    print(f"❌ Critical service import failed in tasks: {e}")
    database_service = None
    job_service = None

logger = structlog.get_logger(__name__)

class BaseTask:
    """Base class for all Modomo tasks with common functionality"""
    
    @staticmethod
    def update_job_progress(
        job_id: str, 
        status: str = "running", 
        processed: int = 0, 
        total: int = 1, 
        message: str = None,
        error_message: str = None
    ):
        """Update job progress in both Redis and database"""
        try:
            # Update Redis if available
            if job_service and job_service.is_available():
                job_service.update_job(
                    job_id=job_id,
                    processed=processed,
                    total=total,
                    status=status,
                    message=message or f"Processed {processed}/{total}"
                )
            
            # Update database if available
            if database_service:
                try:
                    import asyncio
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(
                        database_service.update_job_progress(
                            job_id=job_id,
                            processed_items=processed,
                            total_items=total,
                            status=status,
                            error_message=error_message
                        )
                    )
                    loop.close()
                    logger.debug(f"✅ Updated database for job {job_id}")
                except Exception as db_error:
                    logger.warning(f"⚠️  Database update failed for job {job_id}: {db_error}")
            else:
                logger.debug(f"⚠️  Database service not available for job {job_id}")
                
        except Exception as e:
            logger.warning(f"Failed to update job progress for {job_id}: {e}")
    
    @staticmethod
    def update_celery_progress(processed: int, total: int, message: str = None):
        """Update Celery task progress"""
        try:
            if current_task:
                current_task.update_state(
                    state="PROGRESS",
                    meta={
                        "current": processed,
                        "total": total,
                        "progress": int((processed / total) * 100) if total > 0 else 0,
                        "message": message or f"Processed {processed}/{total}"
                    }
                )
        except Exception as e:
            logger.warning(f"Failed to update Celery progress: {e}")
    
    @staticmethod
    def handle_task_error(job_id: str, error: Exception, processed: int = 0, total: int = 1):
        """Handle task errors consistently"""
        error_message = str(error)
        logger.error(f"Task failed for job {job_id}: {error_message}")
        
        # Update job status to failed
        BaseTask.update_job_progress(
            job_id=job_id,
            status="failed",
            processed=processed,
            total=total,
            error_message=error_message
        )
        
        # Re-raise the error for Celery's retry mechanism
        raise error
    
    @staticmethod
    def complete_job(job_id: str, processed: int, total: int, result: Dict[str, Any] = None):
        """Mark job as completed"""
        BaseTask.update_job_progress(
            job_id=job_id,
            status="completed",
            processed=processed,
            total=total,
            message=f"Successfully completed {processed}/{total} items"
        )
        
        logger.info(f"Job {job_id} completed successfully: {processed}/{total}")
        return result or {"status": "completed", "processed": processed, "total": total}