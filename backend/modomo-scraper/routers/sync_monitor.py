"""
Synchronization monitoring router for FE/Celery/Redis/Railway coordination
Tracks the complete pipeline: Import → AI Detection → Storage Operations
"""
import structlog
from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import asyncio

logger = structlog.get_logger(__name__)

sync_router = APIRouter(prefix="/sync", tags=["Synchronization"])

# Import services
from services.job_service import JobService
from services.database_service import DatabaseService
from main_refactored import _job_service, _database_service, celery_app
from tasks.hybrid_processing import run_ai_detection_batch

@sync_router.get("/status",
                summary="System Sync Status",
                description="Get comprehensive sync status for FE/Celery/Redis/Railway components")
async def get_system_sync_status():
    """
    Get comprehensive sync status for FE/Celery/Redis/Railway
    Shows health and connectivity of all four components
    """
    try:
        sync_status = {
            "timestamp": datetime.utcnow().isoformat(),
            "components": {},
            "pipeline_health": "unknown",
            "active_imports": 0,
            "active_detections": 0,
            "pending_handoffs": 0
        }
        
        # 1. Frontend Status (implied healthy if API is responding)
        sync_status["components"]["frontend"] = {
            "status": "healthy",
            "message": "API responding, frontend can communicate",
            "last_check": datetime.utcnow().isoformat()
        }
        
        # 2. Redis Status
        if _job_service and _job_service.is_available():
            try:
                active_jobs = _job_service.get_active_jobs()
                import_jobs = [j for j in active_jobs if j.get("job_type") == "import"]
                detection_jobs = [j for j in active_jobs if "detection" in j.get("job_id", "")]
                
                sync_status["components"]["redis"] = {
                    "status": "healthy",
                    "message": f"Connected, tracking {len(active_jobs)} active jobs",
                    "active_jobs": len(active_jobs),
                    "import_jobs": len(import_jobs),
                    "detection_jobs": len(detection_jobs)
                }
                sync_status["active_imports"] = len(import_jobs)
                sync_status["active_detections"] = len(detection_jobs)
                
            except Exception as redis_error:
                sync_status["components"]["redis"] = {
                    "status": "error",
                    "message": f"Redis error: {redis_error}",
                    "active_jobs": 0
                }
        else:
            sync_status["components"]["redis"] = {
                "status": "unavailable",
                "message": "Redis job service not available",
                "active_jobs": 0
            }
        
        # 3. Celery Status (check worker health)
        try:
            # Check active Celery tasks
            inspect = celery_app.control.inspect()
            active_tasks = inspect.active()
            reserved_tasks = inspect.reserved()
            
            total_active = sum(len(tasks) for tasks in (active_tasks or {}).values())
            total_reserved = sum(len(tasks) for tasks in (reserved_tasks or {}).values())
            
            sync_status["components"]["celery"] = {
                "status": "healthy" if active_tasks is not None else "disconnected",
                "message": f"Workers responding, {total_active} active, {total_reserved} reserved",
                "active_tasks": total_active,
                "reserved_tasks": total_reserved,
                "workers": list((active_tasks or {}).keys())
            }
            
        except Exception as celery_error:
            sync_status["components"]["celery"] = {
                "status": "error",
                "message": f"Celery connection failed: {celery_error}",
                "active_tasks": 0,
                "reserved_tasks": 0,
                "workers": []
            }
        
        # 4. Railway Status (database connectivity)
        if _database_service:
            try:
                db_test = await _database_service.test_connection()
                
                # Check for pending handoffs (import completed, detection not started)
                pending_handoffs = await get_pending_handoffs()
                sync_status["pending_handoffs"] = len(pending_handoffs)
                
                sync_status["components"]["railway"] = {
                    "status": "healthy" if db_test["status"] == "success" else "error",
                    "message": f"Database connected, {sync_status['pending_handoffs']} pending handoffs",
                    "database": db_test["status"],
                    "can_read": db_test.get("can_read", False),
                    "can_insert": db_test.get("can_insert", False)
                }
                
            except Exception as railway_error:
                sync_status["components"]["railway"] = {
                    "status": "error",
                    "message": f"Railway database error: {railway_error}",
                    "database": "error"
                }
        else:
            sync_status["components"]["railway"] = {
                "status": "unavailable",
                "message": "Database service not initialized",
                "database": "unavailable"
            }
        
        # 5. Overall Pipeline Health Assessment
        component_statuses = [comp["status"] for comp in sync_status["components"].values()]
        
        if all(status == "healthy" for status in component_statuses):
            sync_status["pipeline_health"] = "healthy"
        elif any(status == "error" for status in component_statuses):
            sync_status["pipeline_health"] = "degraded"
        else:
            sync_status["pipeline_health"] = "partial"
        
        return sync_status
        
    except Exception as e:
        logger.error(f"❌ Sync status check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Sync status check failed: {e}")


@sync_router.get("/pipeline/active")
async def get_active_pipeline_jobs():
    """
    Get all jobs currently in the pipeline with their stage information
    Shows: Import → AI Detection → Storage Operations flow
    """
    try:
        pipeline_jobs = {
            "import_stage": [],      # Celery import jobs
            "handoff_stage": [],     # Completed imports waiting for AI detection
            "detection_stage": [],   # Active AI detection jobs
            "storage_stage": [],     # Storage operations (maps/masks)
            "completed": []          # Fully completed jobs
        }
        
        # Get active Redis jobs
        if _job_service and _job_service.is_available():
            active_redis_jobs = _job_service.get_active_jobs()
            
            for job in active_redis_jobs:
                job_id = job.get("job_id", "")
                job_type = job.get("job_type", "")
                status = job.get("status", "")
                
                if job_type == "import" and status in ["pending", "running"]:
                    pipeline_jobs["import_stage"].append({
                        "job_id": job_id,
                        "status": status,
                        "progress": job.get("progress", 0),
                        "message": job.get("message", ""),
                        "stage": "celery_import"
                    })
        
        # Get database jobs for detection and storage stages
        if _database_service:
            # Active detection jobs
            detection_jobs_result = _database_service.supabase.table("scraping_jobs").select(
                "job_id, status, progress, processed_items, total_items, updated_at, parameters"
            ).in_("status", ["running", "processing"]).like("job_id", "%detection%").execute()
            
            for job in detection_jobs_result.data or []:
                pipeline_jobs["detection_stage"].append({
                    "job_id": job["job_id"],
                    "status": job["status"],
                    "progress": job["progress"],
                    "processed_items": job["processed_items"],
                    "total_items": job["total_items"],
                    "stage": "railway_ai_detection",
                    "updated_at": job["updated_at"]
                })
            
            # Check for pending handoffs
            pending_handoffs = await get_pending_handoffs()
            pipeline_jobs["handoff_stage"] = pending_handoffs
            
            # Recent completed jobs (last 10)
            completed_jobs_result = _database_service.supabase.table("scraping_jobs").select(
                "job_id, status, progress, completed_at, parameters"
            ).eq("status", "completed").order("completed_at", desc=True).limit(10).execute()
            
            for job in completed_jobs_result.data or []:
                pipeline_jobs["completed"].append({
                    "job_id": job["job_id"],
                    "status": job["status"],
                    "progress": job["progress"],
                    "completed_at": job["completed_at"],
                    "stage": "fully_completed",
                    "processing_mode": job.get("parameters", {}).get("processing_mode", "unknown")
                })
        
        return pipeline_jobs
        
    except Exception as e:
        logger.error(f"❌ Pipeline jobs check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Pipeline jobs check failed: {e}")


@sync_router.get("/detection/triggers")
async def get_ai_detection_triggers():
    """
    Monitor when AI detection is being/has been triggered
    Shows real-time AI detection activity and triggers
    """
    try:
        detection_activity = {
            "active_detections": [],
            "recent_triggers": [],
            "detection_queue_depth": 0,
            "avg_detection_time": 0,
            "success_rate": 0
        }
        
        if not _database_service:
            raise HTTPException(status_code=503, detail="Database service unavailable")
        
        # 1. Currently active AI detection jobs
        active_detection_result = _database_service.supabase.table("scraping_jobs").select(
            "job_id, status, progress, started_at, updated_at, parameters, processed_items, total_items"
        ).in_("status", ["running", "processing"]).like("job_id", "%detection%").execute()
        
        for job in active_detection_result.data or []:
            # Calculate detection duration
            started_at = datetime.fromisoformat(job["started_at"].replace("Z", "+00:00"))
            duration_minutes = (datetime.utcnow().replace(tzinfo=started_at.tzinfo) - started_at).total_seconds() / 60
            
            detection_activity["active_detections"].append({
                "job_id": job["job_id"],
                "progress": job["progress"],
                "processed_items": job["processed_items"],
                "total_items": job["total_items"],
                "duration_minutes": round(duration_minutes, 1),
                "original_job": job.get("parameters", {}).get("original_import_job", "unknown"),
                "processing_mode": job.get("parameters", {}).get("processing_mode", "unknown")
            })
        
        # 2. Recent AI detection triggers (last 24 hours)
        yesterday = (datetime.utcnow() - timedelta(hours=24)).isoformat()
        recent_triggers_result = _database_service.supabase.table("scraping_jobs").select(
            "job_id, status, started_at, completed_at, parameters, processed_items"
        ).like("job_id", "%detection%").gte("started_at", yesterday).order("started_at", desc=True).execute()
        
        detection_times = []
        successful_detections = 0
        total_detections = 0
        
        for job in recent_triggers_result.data or []:
            total_detections += 1
            
            trigger_info = {
                "job_id": job["job_id"],
                "status": job["status"],
                "started_at": job["started_at"],
                "original_job": job.get("parameters", {}).get("original_import_job", "unknown"),
                "scenes_processed": job["processed_items"]
            }
            
            if job["status"] == "completed":
                successful_detections += 1
                if job["completed_at"] and job["started_at"]:
                    started = datetime.fromisoformat(job["started_at"].replace("Z", "+00:00"))
                    completed = datetime.fromisoformat(job["completed_at"].replace("Z", "+00:00"))
                    duration = (completed - started).total_seconds() / 60
                    detection_times.append(duration)
                    trigger_info["duration_minutes"] = round(duration, 1)
            
            detection_activity["recent_triggers"].append(trigger_info)
        
        # 3. Calculate metrics
        if detection_times:
            detection_activity["avg_detection_time"] = round(sum(detection_times) / len(detection_times), 1)
        
        if total_detections > 0:
            detection_activity["success_rate"] = round((successful_detections / total_detections) * 100, 1)
        
        detection_activity["detection_queue_depth"] = len(detection_activity["active_detections"])
        
        return detection_activity
        
    except Exception as e:
        logger.error(f"❌ AI detection triggers check failed: {e}")
        raise HTTPException(status_code=500, detail=f"AI detection triggers check failed: {e}")


@sync_router.post("/handoff/trigger")
async def trigger_pending_handoffs(background_tasks: BackgroundTasks):
    """
    Manually trigger AI detection for completed import jobs that are waiting for handoff
    This ensures the pipeline doesn't get stuck
    """
    try:
        if not _database_service:
            raise HTTPException(status_code=503, detail="Database service unavailable")
        
        # Get all pending handoffs
        pending_handoffs = await get_pending_handoffs()
        
        if not pending_handoffs:
            return {
                "status": "success",
                "message": "No pending handoffs found",
                "triggered_jobs": 0
            }
        
        triggered_count = 0
        
        for handoff in pending_handoffs:
            try:
                job_id = handoff["job_id"]
                
                # Get scenes imported by this job
                scenes_result = _database_service.supabase.table("scenes").select(
                    "scene_id, image_url"
                ).like("houzz_id", f"hf_%").execute()  # This would need refinement based on job parameters
                
                if scenes_result.data:
                    # Schedule AI detection as background task
                    background_tasks.add_task(
                        run_ai_detection_batch, 
                        scenes_result.data[-handoff.get("processed_items", 10):], 
                        job_id
                    )
                    triggered_count += 1
                    logger.info(f"🚀 Triggered AI detection for job {job_id}")
                
            except Exception as job_error:
                logger.error(f"❌ Failed to trigger handoff for job {handoff['job_id']}: {job_error}")
                continue
        
        return {
            "status": "success",
            "message": f"Triggered AI detection for {triggered_count} pending jobs",
            "triggered_jobs": triggered_count,
            "pending_handoffs": len(pending_handoffs)
        }
        
    except Exception as e:
        logger.error(f"❌ Handoff trigger failed: {e}")
        raise HTTPException(status_code=500, detail=f"Handoff trigger failed: {e}")


@sync_router.get("/storage/operations")
async def get_storage_operations_status():
    """
    Monitor storage operations for maps and masks
    Shows DB and R2 storage activity
    """
    try:
        storage_status = {
            "recent_maps_stored": 0,
            "recent_masks_stored": 0,
            "storage_health": "unknown",
            "db_operations": 0,
            "r2_operations": 0
        }
        
        if not _database_service:
            raise HTTPException(status_code=503, detail="Database service unavailable")
        
        # Check recent storage activity (last hour)
        one_hour_ago = (datetime.utcnow() - timedelta(hours=1)).isoformat()
        
        # Count recent detected objects with masks (indicates mask storage)
        masks_result = _database_service.supabase.table("detected_objects").select(
            "object_id", count="exact"
        ).not_.is_("mask_r2_key", "null").gte("created_at", one_hour_ago).execute()
        
        storage_status["recent_masks_stored"] = masks_result.count or 0
        
        # Count recent scenes (indicates potential map generation)
        scenes_result = _database_service.supabase.table("scenes").select(
            "scene_id", count="exact"
        ).gte("created_at", one_hour_ago).execute()
        
        storage_status["recent_maps_stored"] = scenes_result.count or 0
        
        # Estimate DB and R2 operations
        storage_status["db_operations"] = storage_status["recent_masks_stored"] + storage_status["recent_maps_stored"]
        storage_status["r2_operations"] = storage_status["recent_masks_stored"]  # Each mask is stored in R2
        
        # Determine storage health
        if storage_status["db_operations"] > 0 or storage_status["r2_operations"] > 0:
            storage_status["storage_health"] = "active"
        else:
            storage_status["storage_health"] = "idle"
        
        return storage_status
        
    except Exception as e:
        logger.error(f"❌ Storage operations check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Storage operations check failed: {e}")


async def get_pending_handoffs() -> List[Dict[str, Any]]:
    """
    Helper function to find import jobs that completed but haven't triggered AI detection
    These are jobs stuck in the handoff stage
    """
    try:
        if not _database_service:
            return []
        
        # Find completed import jobs from last 24 hours
        yesterday = (datetime.utcnow() - timedelta(hours=24)).isoformat()
        completed_imports = _database_service.supabase.table("scraping_jobs").select(
            "job_id, status, completed_at, processed_items, parameters"
        ).eq("status", "completed").not_.like("job_id", "%detection%").gte("completed_at", yesterday).execute()
        
        pending_handoffs = []
        
        for job in completed_imports.data or []:
            job_id = job["job_id"]
            
            # Check if corresponding detection job exists
            detection_job_id = f"{job_id}_detection"
            detection_exists = _database_service.supabase.table("scraping_jobs").select(
                "job_id"
            ).eq("job_id", detection_job_id).execute()
            
            # If no detection job exists, this is a pending handoff
            if not detection_exists.data:
                pending_handoffs.append({
                    "job_id": job_id,
                    "completed_at": job["completed_at"],
                    "processed_items": job["processed_items"],
                    "parameters": job.get("parameters", {}),
                    "waiting_for": "ai_detection"
                })
        
        return pending_handoffs
        
    except Exception as e:
        logger.error(f"❌ Error finding pending handoffs: {e}")
        return []