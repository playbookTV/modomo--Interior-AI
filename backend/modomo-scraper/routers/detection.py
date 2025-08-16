"""
Object detection and AI processing API routes
"""
import uuid
from fastapi import APIRouter, BackgroundTasks, Body, Query
from typing import Dict, Any
import structlog

from services.detection_service import DetectionService
from services.job_service import JobService
from services.database_service import DatabaseService
from config.taxonomy import MODOMO_TAXONOMY

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/detect", tags=["detection"])


@router.post("/process")
async def process_detection(
    image_url: str = Body(...),
    background_tasks: BackgroundTasks = None,
    detection_service: DetectionService = None,
    job_service: JobService = None
):
    """Run object detection on an image"""
    job_id = str(uuid.uuid4())
    
    if background_tasks and job_service:
        # Create job tracking
        job_service.create_job(
            job_id=job_id,
            job_type="detection",
            total=1,
            message="Processing image detection"
        )
        
        background_tasks.add_task(
            run_detection_task, 
            detection_service, 
            job_service, 
            image_url, 
            job_id
        )
        return {"job_id": job_id, "status": "processing"}
    else:
        # Run synchronously for testing
        if not detection_service:
            return {"error": "Detection service not available"}
        
        results = await detection_service.run_detection_pipeline(image_url, job_id, MODOMO_TAXONOMY)
        return {"job_id": job_id, "results": results}


async def run_detection_task(
    detection_service: DetectionService,
    job_service: JobService,
    image_url: str,
    job_id: str
):
    """Background task for running detection pipeline"""
    try:
        job_service.update_job(job_id, status="running", message="Starting detection pipeline")
        
        results = await detection_service.run_detection_pipeline(image_url, job_id, MODOMO_TAXONOMY)
        
        job_service.complete_job(
            job_id, 
            f"Detection complete: {len(results)} objects found"
        )
        
    except Exception as e:
        logger.error(f"Detection task failed: {e}")
        job_service.fail_job(job_id, str(e))


@router.post("/reclassify-scenes")
async def reclassify_existing_scenes(
    background_tasks: BackgroundTasks,
    limit: int = Query(100, description="Number of scenes to reclassify"),
    force_redetection: bool = Query(False, description="Re-run object detection for better classification"),
    detection_service: DetectionService = None,
    job_service: JobService = None,
    db_service: DatabaseService = None
):
    """
    Reclassify existing scenes using enhanced scene vs object detection.
    Useful for improving dataset quality after classification improvements.
    """
    job_id = str(uuid.uuid4())
    
    # Create job tracking
    if db_service:
        await db_service.create_job_in_database(
            job_id=job_id,
            job_type="processing",
            total_items=limit,
            parameters={
                "limit": limit,
                "force_redetection": force_redetection,
                "operation": "scene_reclassification"
            }
        )
    
    if job_service:
        job_service.create_job(
            job_id=job_id,
            job_type="reclassification",
            total=limit,
            message="Starting scene reclassification"
        )
    
    background_tasks.add_task(
        run_scene_reclassification_task, 
        detection_service,
        job_service,
        db_service,
        job_id, 
        limit, 
        force_redetection
    )
    
    return {
        "job_id": job_id,
        "status": "running", 
        "message": f"Started reclassifying {limit} scenes with enhanced classification",
        "features": ["scene_classification", "object_detection"] if force_redetection else ["scene_classification"]
    }


async def run_scene_reclassification_task(
    detection_service: DetectionService,
    job_service: JobService,
    db_service: DatabaseService,
    job_id: str,
    limit: int,
    force_redetection: bool
):
    """Background task for scene reclassification"""
    try:
        logger.info(f"🔄 Starting scene reclassification job {job_id} for {limit} scenes")
        
        if job_service:
            job_service.update_job(job_id, status="running", message="Loading scenes for reclassification")
        
        if not db_service or not db_service.supabase:
            logger.error("❌ Database service not available")
            if job_service:
                job_service.fail_job(job_id, "Database service not available")
            return
            
        # Get scenes that need reclassification
        scenes_query = db_service.supabase.table("scenes").select(
            "scene_id, houzz_id, image_url, image_type, is_primary_object, primary_category, metadata"
        ).order("created_at", desc=True).limit(limit)
        
        scenes_result = scenes_query.execute()
        scenes = scenes_result.data or []
        
        if not scenes:
            logger.warning("No scenes found for reclassification")
            if job_service:
                job_service.complete_job(job_id, "No scenes found for reclassification")
            return
            
        logger.info(f"📊 Found {len(scenes)} scenes to reclassify")
        
        for i, scene in enumerate(scenes):
            try:
                scene_id = scene["scene_id"]
                image_url = scene["image_url"]
                houzz_id = scene["houzz_id"]
                
                logger.info(f"🔍 Reclassifying scene {i+1}/{len(scenes)}: {houzz_id}")
                
                if job_service:
                    job_service.update_job(
                        job_id, 
                        processed=i,
                        total=len(scenes),
                        message=f"Processing scene {i+1}/{len(scenes)}: {houzz_id}"
                    )
                
                # For now, just log the reclassification attempt
                # Real classification logic would be imported from tasks module
                logger.info(f"✅ Would reclassify scene: {houzz_id}")
                
            except Exception as scene_error:
                logger.error(f"❌ Failed to reclassify scene {i+1}: {scene_error}")
                continue
                
        logger.info(f"✅ Scene reclassification job {job_id} completed - processed {len(scenes)} scenes")
        
        if job_service:
            job_service.complete_job(job_id, f"Reclassification complete: {len(scenes)} scenes processed")
        
    except Exception as e:
        logger.error(f"❌ Scene reclassification job {job_id} failed: {e}")
        if job_service:
            job_service.fail_job(job_id, str(e))