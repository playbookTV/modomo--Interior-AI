"""
Job management and tracking API routes
"""
from fastapi import APIRouter, HTTPException, Query, Depends
from typing import List, Dict, Any, Optional
import structlog
import uuid

from core.dependencies import get_job_service, get_database_service, get_detection_service
# Services imported via dependencies

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("/active", response_model=None)
async def get_active_jobs(
    job_service = Depends(get_job_service),
    db_service = Depends(get_database_service)
) -> List[Dict[str, Any]]:
    """Get currently active jobs from Redis and Database"""
    
    all_jobs = []
    redis_jobs = []
    db_jobs = []
    
    # Get jobs from Redis (real-time active tracking)
    if job_service and job_service.is_available():
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
async def get_job_status(
    job_id: str, 
    job_service = Depends(get_job_service)
):
    """Get the status and progress of a specific job"""
    if not job_service or not job_service.is_available():
        raise HTTPException(status_code=503, detail="Job tracking not available")
    
    job_data = job_service.get_job(job_id)
    
    if not job_data:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return job_data


@router.get("/errors/recent", response_model=None)
async def get_recent_job_errors(
    job_service = Depends(get_job_service)
):
    """Get recent job errors for frontend display"""
    if not job_service or not job_service.is_available():
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
):
    """Get historical job data from database"""
    db_service = get_database_service()
    
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
async def retry_job(job_id: str):
    """Retry a failed or stuck job"""
    try:
        job_service = get_job_service()
        db_service = get_database_service()
        
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
async def cancel_job(job_id: str):
    """Cancel a running job"""
    try:
        db_service = get_database_service()
        
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


@router.post("/generate-maps/batch", response_model=None)
async def generate_maps_batch(
    limit: int = Query(10, description="Number of scenes to process"),
    map_types: List[str] = Query(["depth", "edge"], description="Types of maps to generate"),
    force_regenerate: bool = Query(False, description="Force regenerate existing maps"),
    db_service = Depends(get_database_service),
    job_service = Depends(get_job_service)
):
    """Batch generate depth and edge maps for scenes"""
    try:
        if not db_service or not db_service.supabase:
            raise HTTPException(status_code=503, detail="Database not available")
        
        job_id = str(uuid.uuid4())
        
        # Create job tracking
        if job_service:
            job_service.create_job(
                job_id=job_id,
                job_type="map_generation", 
                total=limit,
                message="Starting batch map generation"
            )
        
        # Get scenes that need map generation
        query = db_service.supabase.table("scenes").select("scene_id, image_url, depth_map_r2_key, edge_map_r2_key")
        
        if not force_regenerate:
            # Only get scenes without existing maps
            query = query.or_("depth_map_r2_key.is.null,edge_map_r2_key.is.null")
        
        result = query.limit(limit).execute()
        
        if not result.data:
            if job_service:
                job_service.complete_job(job_id, "No scenes found for map generation")
            return {
                "job_id": job_id,
                "message": "No scenes found for map generation",
                "scenes_processed": 0,
                "maps_generated": {}
            }
        
        scenes_to_process = result.data
        logger.info(f"🗺️ Starting batch map generation for {len(scenes_to_process)} scenes")
        
        # Import map generation dependencies
        from models.map_generator import MapGenerator, MapGenerationConfig
        from models.depth_estimator import DepthAnythingV2
        from models.edge_detector import EdgeDetector
        
        # Initialize map generator
        config = MapGenerationConfig(
            upload_to_r2=True,
            keep_local_copies=False,
            max_concurrent_maps=1
        )
        
        try:
            depth_estimator = DepthAnythingV2()
            edge_detector = EdgeDetector()
            map_generator = MapGenerator(depth_estimator, edge_detector, config)
        except Exception as e:
            logger.error(f"Failed to initialize map generator: {e}")
            if job_service:
                job_service.fail_job(job_id, f"Failed to initialize map generator: {e}")
            raise HTTPException(status_code=500, detail=f"Map generator initialization failed: {e}")
        
        maps_generated = {"depth": 0, "edge": 0}
        scenes_processed = 0
        errors = []
        
        for i, scene in enumerate(scenes_to_process):
            try:
                if job_service:
                    job_service.update_job(
                        job_id,
                        processed=i,
                        total=len(scenes_to_process),
                        message=f"Generating maps for scene {i+1}/{len(scenes_to_process)}"
                    )
                
                scene_id = scene["scene_id"]
                image_url = scene["image_url"]
                
                logger.info(f"🗺️ Processing scene {scene_id} ({i+1}/{len(scenes_to_process)})")
                
                # Generate maps
                result = await map_generator.generate_all_maps(
                    image_url,
                    scene_id,
                    map_types
                )
                
                if result["success"] and result["r2_keys"]:
                    # Update scene record with R2 keys
                    updates = {}
                    
                    for map_type, r2_key in result["r2_keys"].items():
                        if map_type == "depth":
                            updates["depth_map_r2_key"] = r2_key
                            maps_generated["depth"] += 1
                        elif map_type == "edge":
                            updates["edge_map_r2_key"] = r2_key
                            maps_generated["edge"] += 1
                    
                    if updates:
                        db_service.supabase.table("scenes").update(updates).eq("scene_id", scene_id).execute()
                        logger.info(f"✅ Updated scene {scene_id} with map keys: {updates}")
                
                scenes_processed += 1
                
            except Exception as scene_error:
                error_msg = f"Scene {scene.get('scene_id', 'unknown')}: {str(scene_error)}"
                errors.append(error_msg)
                logger.error(f"❌ Map generation failed for scene: {error_msg}")
        
        # Cleanup
        if 'map_generator' in locals():
            map_generator.cleanup()
        
        message = f"Batch map generation completed: {scenes_processed} scenes processed, {sum(maps_generated.values())} total maps generated"
        
        if job_service:
            job_service.complete_job(job_id, message)
        
        return {
            "job_id": job_id,
            "message": message,
            "scenes_processed": scenes_processed,
            "maps_generated": maps_generated,
            "errors": errors,
            "total_scenes": len(scenes_to_process)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch map generation failed: {e}")
        if 'job_service' in locals() and job_service:
            job_service.fail_job(job_id, str(e))
        raise HTTPException(status_code=500, detail=f"Batch map generation failed: {str(e)}")


@router.post("/retry-pending", response_model=None)
async def retry_pending_jobs(
    job_type: str = Query(None, description="Filter by job type"),
    older_than_hours: int = Query(1, description="Retry jobs older than X hours"),
    limit: int = Query(50, description="Maximum jobs to retry"),
):
    """Bulk retry pending jobs that are stuck"""
    try:
        db_service = get_database_service()
        
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


@router.get("/performance/status", response_model=None)
async def get_performance_status():
    """Get system performance status and monitoring information"""
    try:
        # Import performance monitor
        try:
            from utils.performance_monitor import get_performance_monitor
            monitor = get_performance_monitor()
            performance_available = True
        except ImportError:
            logger.warning("Performance monitor not available")
            performance_available = False
            monitor = None
        
        status = {
            "performance_monitoring": performance_available,
            "timestamp": f"{uuid.uuid4()}",  # Using UUID for unique timestamp
            "status": "healthy"
        }
        
        if performance_available and monitor:
            try:
                # Get performance metrics
                system_info = monitor.get_system_info()
                recommendations = monitor.get_recommendations()
                
                status.update({
                    "system": system_info,
                    "recommendations": recommendations,
                    "memory_usage": system_info.get("memory_usage"),
                    "gpu_available": system_info.get("gpu_available", False),
                    "gpu_memory": system_info.get("gpu_memory"),
                    "cpu_count": system_info.get("cpu_count"),
                    "estimated_times": recommendations.get("estimated_times", {}),
                    "warnings": recommendations.get("warnings", [])
                })
                
                # Add service-specific status
                detection_service = get_detection_service()
                db_service = get_database_service()
                job_service = get_job_service()
                
                status["services"] = {
                    "detection_service": detection_service is not None,
                    "database_service": db_service is not None,
                    "job_service": job_service is not None
                }
                
            except Exception as monitor_error:
                logger.error(f"Failed to get performance metrics: {monitor_error}")
                status["monitor_error"] = str(monitor_error)
        else:
            # Fallback status without performance monitoring
            import psutil
            
            status.update({
                "system": {
                    "cpu_percent": psutil.cpu_percent(interval=1),
                    "memory_percent": psutil.virtual_memory().percent,
                    "cpu_count": psutil.cpu_count(),
                    "gpu_available": False
                },
                "fallback_mode": True
            })
        
        return status
        
    except Exception as e:
        logger.error(f"Performance status check failed: {e}")
        return {
            "performance_monitoring": False,
            "status": "error",
            "error": str(e),
            "timestamp": f"{uuid.uuid4()}"
        }