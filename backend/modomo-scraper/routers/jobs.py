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


@router.get("/active")
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


@router.get("/{job_id}/status")
async def get_job_status(job_id: str, job_service: JobService = Depends()):
    """Get the status and progress of a specific job"""
    if not job_service.is_available():
        raise HTTPException(status_code=503, detail="Job tracking not available")
    
    job_data = job_service.get_job(job_id)
    
    if not job_data:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return job_data


@router.get("/errors/recent")
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


@router.get("/history")
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