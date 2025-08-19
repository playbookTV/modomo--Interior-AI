"""
Job management and tracking API routes
"""
from fastapi import APIRouter, HTTPException, Query, Depends
from typing import List, Dict, Any
import structlog

from services.job_service import JobService
from services.database_service import DatabaseService

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("/active", response_model=None)
async def get_active_jobs(
    job_service: JobService = Depends(),
    db_service: DatabaseService = Depends()
) -> List[Dict[str, Any]]:
    """Get currently active jobs from Redis and Database"""
    all_jobs = []
    redis_jobs = []
    db_jobs = []
    
    # Get jobs from Redis (real-time active tracking)
    if job_service.is_available():
        redis_jobs = job_service.get_active_jobs()
    
    # Also get active jobs from database (persistent tracking)
    if db_service and db_service.supabase:
        try:
            result = db_service.supabase.table("scraping_jobs").select("*").in_(
                "status", ["pending", "running", "processing"]
            ).order("created_at", desc=True).execute()
            
            if result.data:
                for job in result.data:
                    # Convert to match Redis format for consistency
                    db_job = {
                        "job_id": job["job_id"],
                        "status": job["status"],
                        "progress": str(job.get("progress", 0)),
                        "total": str(job.get("total_items", 0)),
                        "processed": str(job.get("processed_items", 0)),
                        "message": f"{job.get('job_type', 'processing').title()} job",
                        "created_at": job.get("created_at", ""),
                        "updated_at": job.get("updated_at", ""),
                        "job_type": job.get("job_type", "processing"),
                        "error_message": job.get("error_message"),
                        "parameters": job.get("parameters", {})
                    }
                    db_jobs.append(db_job)
                    logger.debug(f"Found active DB job: {job['job_id']}")
        except Exception as e:
            logger.error(f"Failed to get active jobs from database: {e}")
    
    # Combine and deduplicate (Redis takes precedence for active jobs)
    redis_job_ids = {job["job_id"] for job in redis_jobs}
    all_jobs.extend(redis_jobs)
    
    # Add DB jobs that aren't already in Redis
    for db_job in db_jobs:
        if db_job["job_id"] not in redis_job_ids:
            all_jobs.append(db_job)
    
    logger.info(f"Returning {len(all_jobs)} active jobs ({len(redis_jobs)} from Redis, {len([j for j in db_jobs if j['job_id'] not in redis_job_ids])} additional from DB)")
    return all_jobs


@router.get("/{job_id}/status", response_model=None)
async def get_job_status(job_id: str, job_service: JobService = Depends()):
    """Get the status and progress of a specific job"""
    if not job_service.is_available():
        raise HTTPException(status_code=503, detail="Job tracking not available")
    
    job_data = job_service.get_job(job_id)
    
    if not job_data:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return job_data


@router.get("/errors/recent", response_model=None)
async def get_recent_job_errors(job_service: JobService = Depends()):
    """Get recent job errors for frontend display"""
    if not job_service.is_available():
        return {"errors": [], "message": "Error tracking not available"}
    
    try:
        recent_errors = job_service.get_recent_errors(limit=10)
        
        return {
            "errors": recent_errors,
            "total_error_jobs": len(recent_errors)
        }
        
    except Exception as e:
        logger.error(f"Failed to get recent errors: {e}")
        return {"errors": [], "error": str(e)}


@router.get("/history", response_model=None)
async def get_job_history(
    limit: int = Query(50, description="Number of historical jobs to return"),
    offset: int = Query(0, description="Offset for pagination"),
    status: str = Query(None, description="Filter by status: completed, failed, all"),
    job_type: str = Query(None, description="Filter by job type: scenes, import, processing, detection"),
    db_service: DatabaseService = Depends()
):
    """Get historical job data from database"""
    if not db_service or not db_service.supabase:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        query = db_service.supabase.table("scraping_jobs").select("*")
        
        # Apply filters
        if status and status != "all":
            query = query.eq("status", status)
        elif not status:
            # Default: show completed and failed jobs (historical)
            query = query.in_("status", ["completed", "failed"])
            
        if job_type:
            query = query.eq("job_type", job_type)
        
        # Order by most recent first and apply pagination
        result = query.order("created_at", desc=True).range(offset, offset + limit - 1).execute()
        
        # Get total count for pagination info
        count_query = db_service.supabase.table("scraping_jobs").select("job_id", count="exact")
        if status and status != "all":
            count_query = count_query.eq("status", status)
        elif not status:
            count_query = count_query.in_("status", ["completed", "failed"])
        if job_type:
            count_query = count_query.eq("job_type", job_type)
            
        count_result = count_query.execute()
        total_jobs = count_result.count if count_result.count else 0
        
        # Convert database format to match frontend expectations
        historical_jobs = []
        for job in result.data:
            job_data = {
                "job_id": job["job_id"],
                "status": job["status"],
                "job_type": job.get("job_type", "processing"),
                "message": f"{job.get('job_type', 'processing').title()} job",
                "progress": job.get("progress", 0),
                "total": job.get("total_items", 0),
                "processed": job.get("processed_items", 0),
                "created_at": job.get("created_at"),
                "updated_at": job.get("updated_at"),
                "started_at": job.get("started_at"),
                "completed_at": job.get("completed_at"),
                "error_message": job.get("error_message"),
                "parameters": job.get("parameters", {}),
                # Add computed fields
                "dataset": job.get("parameters", {}).get("dataset"),
                "features": []  # Could be computed from parameters if needed
            }
            
            # Add duration if job is completed
            if job.get("completed_at") and job.get("started_at"):
                try:
                    from datetime import datetime
                    start = datetime.fromisoformat(job["started_at"].replace('Z', '+00:00'))
                    end = datetime.fromisoformat(job["completed_at"].replace('Z', '+00:00'))
                    duration_seconds = (end - start).total_seconds()
                    job_data["duration_seconds"] = duration_seconds
                except:
                    pass
            
            historical_jobs.append(job_data)
        
        return {
            "jobs": historical_jobs,
            "total": total_jobs,
            "limit": limit,
            "offset": offset,
            "has_more": offset + limit < total_jobs
        }
        
    except Exception as e:
        logger.error(f"Failed to get job history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve job history: {str(e)}")


@router.post("/{job_id}/retry", response_model=None)
async def retry_job(
    job_id: str, 
    job_service: JobService = Depends(), 
    db_service: DatabaseService = Depends()
):
    """Retry a failed or stuck job"""
    try:
        # Get job details from database
        if not db_service or not db_service.supabase:
            raise HTTPException(status_code=503, detail="Database not available")
        
        # Check if job exists and is retryable
        result = db_service.supabase.table("scraping_jobs").select("*").eq("job_id", job_id).execute()
        
        if not result.data:
            raise HTTPException(status_code=404, detail="Job not found")
        
        job = result.data[0]
        
        # Check if job can be retried
        if job["status"] not in ["pending", "failed", "error"]:
            raise HTTPException(
                status_code=400, 
                detail=f"Job with status '{job['status']}' cannot be retried"
            )
        
        # Create new job with same parameters
        import uuid
        from tasks.scraping_tasks import import_huggingface_dataset, run_scraping_job
        from tasks.detection_tasks import run_detection_pipeline
        from datetime import datetime
        
        new_job_id = str(uuid.uuid4())
        job_type = job.get("job_type", "processing")
        parameters = job.get("parameters", {})
        
        # Determine which task to restart based on job type and parameters
        if job_type == "import" and "dataset" in parameters:
            # Restart HuggingFace dataset import
            dataset = parameters["dataset"]
            offset = parameters.get("offset", 0)
            limit = parameters.get("limit", 10)
            include_detection = parameters.get("include_detection", True)
            
            # Queue the import task
            import_huggingface_dataset.apply_async(
                args=[new_job_id, dataset, offset, limit, include_detection],
                queue="import"
            )
            
        elif job_type == "scenes":
            # Restart scene scraping
            limit = parameters.get("limit", 10)
            room_types = parameters.get("room_types", [])
            
            # Queue the scraping task
            run_scraping_job.apply_async(
                args=[new_job_id, limit, room_types],
                queue="scraping"
            )
            
        elif job_type == "detection":
            # Restart detection processing
            image_url = parameters.get("image_url")
            scene_id = parameters.get("scene_id")
            
            if image_url and scene_id:
                run_detection_pipeline.apply_async(
                    args=[new_job_id, image_url, scene_id],
                    queue="ai_processing"
                )
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Missing required parameters for detection job retry"
                )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Job type '{job_type}' retry not implemented"
            )
        
        # Mark original job as cancelled
        db_service.supabase.table("scraping_jobs").update({
            "status": "cancelled",
            "error_message": f"Job cancelled and retried as {new_job_id}",
            "updated_at": datetime.utcnow().isoformat()
        }).eq("job_id", job_id).execute()
        
        logger.info(f"✅ Job {job_id} retried as {new_job_id}")
        
        return {
            "status": "success",
            "message": f"Job retried successfully",
            "original_job_id": job_id,
            "new_job_id": new_job_id,
            "job_type": job_type
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retry job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retry job: {str(e)}")


@router.post("/{job_id}/cancel", response_model=None)
async def cancel_job(job_id: str, db_service: DatabaseService = Depends()):
    """Cancel a running job"""
    try:
        if not db_service or not db_service.supabase:
            raise HTTPException(status_code=503, detail="Database not available")
        
        from datetime import datetime
        
        # Update job status to cancelled
        result = db_service.supabase.table("scraping_jobs").update({
            "status": "cancelled",
            "error_message": "Job cancelled by user",
            "updated_at": datetime.utcnow().isoformat()
        }).eq("job_id", job_id).execute()
        
        if not result.data:
            raise HTTPException(status_code=404, detail="Job not found")
        
        logger.info(f"✅ Job {job_id} cancelled")
        
        return {
            "status": "success",
            "message": "Job cancelled successfully",
            "job_id": job_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to cancel job: {str(e)}")


@router.post("/retry-pending", response_model=None)
async def retry_pending_jobs(
    job_type: str = Query(None, description="Filter by job type"),
    older_than_hours: int = Query(1, description="Retry jobs older than X hours"),
    limit: int = Query(50, description="Maximum jobs to retry"),
    db_service: DatabaseService = Depends()
):
    """Bulk retry pending jobs that are stuck"""
    try:
        if not db_service or not db_service.supabase:
            raise HTTPException(status_code=503, detail="Database not available")
        
        from datetime import datetime, timedelta
        
        # Calculate cutoff time
        cutoff_time = datetime.utcnow() - timedelta(hours=older_than_hours)
        
        # Build query for stuck pending jobs
        query = db_service.supabase.table("scraping_jobs").select("*").eq("status", "pending").lt("created_at", cutoff_time.isoformat())
        
        if job_type:
            query = query.eq("job_type", job_type)
        
        # Get stuck jobs
        result = query.limit(limit).execute()
        
        if not result.data:
            return {
                "retried_jobs": 0,
                "new_job_ids": [],
                "skipped_jobs": 0,
                "message": "No pending jobs found to retry"
            }
        
        retried_jobs = []
        skipped_jobs = 0
        
        for job in result.data:
            try:
                # Retry each job individually (simulate the retry logic)
                job_id = job["job_id"]
                
                # Create new job with same parameters
                import uuid
                from tasks.scraping_tasks import import_huggingface_dataset, run_scraping_job
                from tasks.detection_tasks import run_detection_pipeline
                
                new_job_id = str(uuid.uuid4())
                job_type_val = job.get("job_type", "processing")
                parameters = job.get("parameters", {})
                
                # Determine which task to restart based on job type and parameters
                if job_type_val == "import" and "dataset" in parameters:
                    # Restart HuggingFace dataset import
                    dataset = parameters["dataset"]
                    offset = parameters.get("offset", 0)
                    limit_val = parameters.get("limit", 10)
                    include_detection = parameters.get("include_detection", True)
                    
                    # Queue the import task
                    import_huggingface_dataset.apply_async(
                        args=[new_job_id, dataset, offset, limit_val, include_detection],
                        queue="import"
                    )
                    
                elif job_type_val == "scenes":
                    # Restart scene scraping
                    limit_val = parameters.get("limit", 10)
                    room_types = parameters.get("room_types", [])
                    
                    # Queue the scraping task
                    run_scraping_job.apply_async(
                        args=[new_job_id, limit_val, room_types],
                        queue="scraping"
                    )
                
                # Mark original job as cancelled
                db_service.supabase.table("scraping_jobs").update({
                    "status": "cancelled",
                    "error_message": f"Job cancelled and retried as {new_job_id}",
                    "updated_at": datetime.utcnow().isoformat()
                }).eq("job_id", job_id).execute()
                
                retried_jobs.append(new_job_id)
                logger.info(f"✅ Bulk retry: Job {job_id} retried as {new_job_id}")
                
            except Exception as e:
                logger.warning(f"Failed to retry job {job['job_id']}: {e}")
                skipped_jobs += 1
        
        return {
            "retried_jobs": len(retried_jobs),
            "new_job_ids": retried_jobs,
            "skipped_jobs": skipped_jobs,
            "message": f"Retried {len(retried_jobs)} jobs, skipped {skipped_jobs}"
        }
        
    except Exception as e:
        logger.error(f"Bulk retry failed: {e}")
        raise HTTPException(status_code=500, detail=f"Bulk retry failed: {str(e)}")