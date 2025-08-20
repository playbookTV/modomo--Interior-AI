"""
Job tracking service using Redis for background task management
"""
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
try:
    import redis
except ImportError:
    redis = None
import structlog

logger = structlog.get_logger(__name__)


class JobService:
    """Service for managing background jobs with Redis"""
    
    def __init__(self, redis_client = None):
        self.redis = redis_client
    
    def is_available(self) -> bool:
        """Check if Redis is available for job tracking"""
        return self.redis is not None
    
    def create_job(
        self,
        job_id: str,
        job_type: str,
        total: int,
        message: str = "",
        **kwargs
    ) -> bool:
        """Create a new job in Redis"""
        if not self.redis:
            logger.warning("Redis not available - job tracking disabled")
            return False
        
        try:
            job_data = {
                "job_id": job_id,
                "status": "pending",
                "job_type": job_type,
                "message": message,
                "total": str(total),
                "processed": "0",
                "progress": "0",
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
                **{k: str(v) for k, v in kwargs.items()}
            }
            
            job_key = f"job:{job_id}"
            self.redis.hset(job_key, mapping=job_data)
            
            # Set expiration (24 hours)
            self.redis.expire(job_key, 86400)
            
            logger.info(f"✅ Created job {job_id} in Redis")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create job {job_id} in Redis: {e}")
            return False
    
    def update_job(
        self,
        job_id: str,
        processed: Optional[int] = None,
        total: Optional[int] = None,
        status: Optional[str] = None,
        message: Optional[str] = None,
        **kwargs
    ) -> bool:
        """Update job progress and status"""
        if not self.redis:
            return False
        
        try:
            job_key = f"job:{job_id}"
            
            # Get current job data
            current_data = self.redis.hgetall(job_key)
            if not current_data:
                logger.warning(f"Job {job_id} not found in Redis")
                return False
            
            # Prepare updates
            updates = {"updated_at": datetime.utcnow().isoformat()}
            
            if processed is not None:
                updates["processed"] = str(processed)
            if total is not None:
                updates["total"] = str(total)
            if status is not None:
                updates["status"] = status
            if message is not None:
                updates["message"] = message
            
            # Calculate progress if we have the numbers
            current_processed = int(updates.get("processed", current_data.get(b"processed", b"0")))
            current_total = int(updates.get("total", current_data.get(b"total", b"1")))
            
            if current_total > 0:
                progress = (current_processed / current_total) * 100
                updates["progress"] = str(int(progress))
            
            # Add any additional kwargs
            updates.update({k: str(v) for k, v in kwargs.items()})
            
            # Update Redis
            self.redis.hset(job_key, mapping=updates)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to update job {job_id}: {e}")
            return False
    
    def get_job(self, job_id: str) -> Optional[Dict[str, str]]:
        """Get job status and details"""
        if not self.redis:
            return None
        
        try:
            job_key = f"job:{job_id}"
            job_data = self.redis.hgetall(job_key)
            
            if not job_data:
                return None
            
            # Convert bytes to strings for JSON serialization
            return {
                key.decode() if isinstance(key, bytes) else key: 
                value.decode() if isinstance(value, bytes) else value 
                for key, value in job_data.items()
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get job {job_id}: {e}")
            return None
    
    def get_active_jobs(self) -> List[Dict[str, str]]:
        """Get all currently active jobs"""
        if not self.redis:
            return []
        
        try:
            job_keys = self.redis.keys("job:*")
            active_jobs = []
            
            for job_key in job_keys:
                try:
                    job_data = self.redis.hgetall(job_key)
                    
                    # Check if job_data is actually a hash/dict and not a string
                    if not job_data or not hasattr(job_data, 'items'):
                        logger.debug(f"Skipping malformed job data for {job_key}: {type(job_data)}")
                        continue
                    
                    # Check status
                    status_bytes = job_data.get(b"status", b"")
                    if isinstance(status_bytes, bytes):
                        status = status_bytes.decode()
                    else:
                        status = str(status_bytes)
                    
                    if status in ["pending", "running", "processing"]:
                        # Convert bytes to strings for JSON serialization
                        job_status = {}
                        for key, value in job_data.items():
                            key_str = key.decode() if isinstance(key, bytes) else str(key)
                            value_str = value.decode() if isinstance(value, bytes) else str(value)
                            job_status[key_str] = value_str
                        
                        # Ensure job_id is included from Redis key if missing
                        if "job_id" not in job_status:
                            job_status["job_id"] = job_key.decode().replace("job:", "")
                        
                        active_jobs.append(job_status)
                        
                except Exception as job_error:
                    logger.warning(f"Failed to process job {job_key}: {job_error}")
                    continue
            
            return active_jobs
            
        except Exception as e:
            logger.error(f"❌ Failed to get active jobs: {e}")
            return []
    
    def get_recent_errors(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent job errors"""
        if not self.redis:
            return []
        
        try:
            job_keys = self.redis.keys("job:*")
            recent_errors = []
            
            for job_key in job_keys:
                try:
                    job_data = self.redis.hgetall(job_key)
                    
                    # Check if job_data is actually a hash/dict and not a string
                    if not job_data or not hasattr(job_data, 'items'):
                        logger.debug(f"Skipping malformed job data for {job_key}: {type(job_data)}")
                        continue
                    
                    # Convert bytes to strings for proper comparison and access
                    job_status = {}
                    for key, value in job_data.items():
                        key_str = key.decode() if isinstance(key, bytes) else str(key)
                        value_str = value.decode() if isinstance(value, bytes) else str(value)
                        job_status[key_str] = value_str
                    
                    if job_status.get("status") in ["failed", "error"]:
                        recent_errors.append({
                            "job_id": job_key.decode().replace("job:", ""),
                            "status": job_status.get("status"),
                            "error_message": job_status.get("message", ""),
                            "updated_at": job_status.get("updated_at", ""),
                            "processed": int(job_status.get("processed", 0)),
                            "total": int(job_status.get("total", 0))
                        })
                        
                except Exception as job_error:
                    logger.warning(f"Failed to process error job {job_key}: {job_error}")
                    continue
            
            # Sort by updated_at desc, limit to most recent
            recent_errors.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
            return recent_errors[:limit]
            
        except Exception as e:
            logger.error(f"❌ Failed to get recent errors: {e}")
            return []
    
    def complete_job(self, job_id: str, message: str = "Job completed successfully") -> bool:
        """Mark a job as completed"""
        return self.update_job(
            job_id=job_id,
            status="completed",
            message=message,
            completed_at=datetime.utcnow().isoformat()
        )
    
    def fail_job(self, job_id: str, error_message: str) -> bool:
        """Mark a job as failed"""
        return self.update_job(
            job_id=job_id,
            status="failed",
            message=error_message,
            failed_at=datetime.utcnow().isoformat()
        )
    
    def cleanup_malformed_jobs(self) -> int:
        """Clean up malformed job data in Redis"""
        if not self.redis:
            return 0
        
        cleaned_count = 0
        try:
            job_keys = self.redis.keys("job:*")
            
            for job_key in job_keys:
                try:
                    job_data = self.redis.hgetall(job_key)
                    
                    # If job_data is not a proper hash/dict, delete it
                    if not job_data or not hasattr(job_data, 'items'):
                        logger.info(f"Cleaning up malformed job data: {job_key}")
                        self.redis.delete(job_key)
                        cleaned_count += 1
                        
                except Exception as e:
                    logger.warning(f"Error checking job {job_key}, deleting: {e}")
                    self.redis.delete(job_key)
                    cleaned_count += 1
                    
        except Exception as e:
            logger.error(f"Failed to cleanup malformed jobs: {e}")
        
        if cleaned_count > 0:
            logger.info(f"✅ Cleaned up {cleaned_count} malformed job records")
        
        return cleaned_count