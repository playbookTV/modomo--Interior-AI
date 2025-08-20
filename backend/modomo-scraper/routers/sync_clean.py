"""
Clean sync monitoring router without circular imports
Tracks pipeline coordination between Frontend/Celery/Redis/Railway
"""
import structlog
from fastapi import APIRouter, Depends
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

from core.dependencies import get_job_service, get_database_service

logger = structlog.get_logger(__name__)
sync_router = APIRouter(prefix="/sync", tags=["Synchronization"])


@sync_router.get("/status", response_model=None)
async def get_system_sync_status(
    job_service = Depends(get_job_service),
    db_service = Depends(get_database_service)
):
    """Get comprehensive sync status for all system components"""
    sync_status = {
        "timestamp": datetime.utcnow().isoformat(),
        "components": {},
        "overall_health": "unknown"
    }
    
    # Check Redis/Job Service
    if job_service and job_service.is_available():
        try:
            active_jobs = job_service.get_active_jobs()
            sync_status["components"]["redis"] = {
                "status": "healthy",
                "active_jobs": len(active_jobs),
                "message": "Job tracking operational"
            }
        except Exception as e:
            sync_status["components"]["redis"] = {
                "status": "degraded", 
                "error": str(e),
                "message": "Job tracking issues"
            }
    else:
        sync_status["components"]["redis"] = {
            "status": "unavailable",
            "message": "Job service not available"
        }
    
    # Check Database
    if db_service and db_service.supabase:
        try:
            # Test database connectivity
            test_result = db_service.supabase.table("scenes").select("scene_id").limit(1).execute()
            sync_status["components"]["database"] = {
                "status": "healthy",
                "message": "Database connectivity verified"
            }
        except Exception as e:
            sync_status["components"]["database"] = {
                "status": "degraded",
                "error": str(e),
                "message": "Database connectivity issues"
            }
    else:
        sync_status["components"]["database"] = {
            "status": "unavailable", 
            "message": "Database service not available"
        }
    
    # Determine overall health
    statuses = [comp["status"] for comp in sync_status["components"].values()]
    if all(s == "healthy" for s in statuses):
        sync_status["overall_health"] = "healthy"
    elif any(s == "healthy" for s in statuses):
        sync_status["overall_health"] = "degraded"
    else:
        sync_status["overall_health"] = "critical"
    
    return sync_status


@sync_router.get("/pipeline/active", response_model=None)
async def get_pipeline_status(
    job_service = Depends(get_job_service),
    db_service = Depends(get_database_service)
):
    """Monitor active data processing pipeline"""
    pipeline_status = {
        "active_jobs": [],
        "recent_completions": [],
        "pipeline_health": "unknown"
    }
    
    # Get active jobs
    if job_service and job_service.is_available():
        try:
            active_jobs = job_service.get_active_jobs()
            pipeline_status["active_jobs"] = active_jobs
        except Exception as e:
            logger.warning(f"Failed to get active jobs: {e}")
    
    # Get recent completions from database
    if db_service and db_service.supabase:
        try:
            recent_jobs = db_service.supabase.table("scraping_jobs").select(
                "job_id, status, job_type, completed_at"
            ).in_("status", ["completed", "failed"]).order(
                "completed_at", desc=True
            ).limit(10).execute()
            
            pipeline_status["recent_completions"] = recent_jobs.data or []
        except Exception as e:
            logger.warning(f"Failed to get recent completions: {e}")
    
    # Assess pipeline health
    total_active = len(pipeline_status["active_jobs"])
    if total_active == 0:
        pipeline_status["pipeline_health"] = "idle"
    elif total_active < 5:
        pipeline_status["pipeline_health"] = "normal"
    elif total_active < 10:
        pipeline_status["pipeline_health"] = "busy"
    else:
        pipeline_status["pipeline_health"] = "overloaded"
    
    return pipeline_status


@sync_router.get("/storage/status", response_model=None)
async def get_storage_sync_status(
    db_service = Depends(get_database_service)
):
    """Check storage system synchronization"""
    storage_status = {
        "database_scenes": 0,
        "database_objects": 0,
        "sync_issues": [],
        "last_sync_check": datetime.utcnow().isoformat()
    }
    
    if db_service and db_service.supabase:
        try:
            # Count scenes
            scenes_result = db_service.supabase.table("scenes").select("scene_id", count="exact").execute()
            storage_status["database_scenes"] = scenes_result.count or 0
            
            # Count objects
            objects_result = db_service.supabase.table("detected_objects").select("object_id", count="exact").execute()
            storage_status["database_objects"] = objects_result.count or 0
            
            # Check for sync issues (scenes without objects, etc.)
            orphaned_scenes = db_service.supabase.rpc("count_scenes_without_objects").execute()
            if orphaned_scenes.data and orphaned_scenes.data > 0:
                storage_status["sync_issues"].append(f"{orphaned_scenes.data} scenes without detected objects")
                
        except Exception as e:
            storage_status["sync_issues"].append(f"Database query error: {str(e)}")
    
    return storage_status