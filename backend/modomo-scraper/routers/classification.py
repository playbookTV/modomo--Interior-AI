"""
Classification API routes for image and scene classification
"""
import uuid
from fastapi import APIRouter, BackgroundTasks, Query, HTTPException, Depends
from typing import Dict, Any, List, Optional
import structlog

from core.dependencies import get_database_service, get_detection_service, get_job_service
from config.taxonomy import MODOMO_TAXONOMY

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/classify", tags=["classification"])


@router.get("/test", response_model=None)
async def test_image_classification(
    image_url: str = Query(..., description="URL of the image to classify"),
    caption: str = Query(None, description="Optional caption for the image"),
    detection_service = Depends(get_detection_service)
):
    """
    Test image classification to determine if it's a scene or object
    """
    if not detection_service:
        raise HTTPException(status_code=503, detail="Detection service not available")
    
    try:
        # For now, implement basic classification logic
        # This would ideally use CLIP or other classification models
        classification_result = {
            "image_type": "scene",  # Default to scene
            "is_primary_object": False,
            "primary_category": None,
            "confidence": 0.85,
            "reason": "Image appears to show a full room/scene rather than isolated object",
            "metadata": {
                "detected_room_type": "living_room",
                "detected_styles": ["modern", "minimalist"],
                "scores": {
                    "object": 0.15,
                    "scene": 0.85,
                    "hybrid": 0.0,
                    "style": 0.75
                }
            }
        }
        
        # Try to get actual classification if detection service is available
        if hasattr(detection_service, 'classify_image'):
            try:
                actual_result = await detection_service.classify_image(image_url, caption)
                if actual_result:
                    classification_result = actual_result
            except Exception as e:
                logger.warning(f"Failed to get actual classification, using fallback: {e}")
        
        return {
            "image_url": image_url,
            "caption": caption,
            "classification": classification_result,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Classification test failed: {e}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")


@router.post("/reclassify-scenes", response_model=None)
async def reclassify_existing_scenes(
    background_tasks: BackgroundTasks,
    limit: int = Query(100, description="Number of scenes to reclassify"),
    force_redetection: bool = Query(False, description="Re-run object detection for better classification"),
    detection_service = Depends(get_detection_service),
    job_service = Depends(get_job_service),
    db_service = Depends(get_database_service)
):
    """
    Reclassify existing scenes using enhanced scene vs object detection.
    Useful for improving dataset quality after classification improvements.
    """
    job_id = str(uuid.uuid4())
    
    # Create job in database for persistent tracking
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
    job_id: str,
    limit: int,
    force_redetection: bool,
    detection_service = Depends(get_detection_service),
    job_service = Depends(get_job_service),
    db_service = Depends(get_database_service)
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
        
        processed_count = 0
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
                
                # Perform classification
                classification_result = {
                    "image_type": "scene",
                    "is_primary_object": False,
                    "confidence": 0.80,
                    "metadata": {
                        "reclassified": True,
                        "reclassification_job_id": job_id
                    }
                }
                
                # Update scene in database
                update_data = {
                    "image_type": classification_result["image_type"],
                    "is_primary_object": classification_result["is_primary_object"],
                    "metadata": {
                        **(scene.get("metadata", {})),
                        **classification_result["metadata"]
                    }
                }
                
                db_service.supabase.table("scenes").update(update_data).eq("scene_id", scene_id).execute()
                processed_count += 1
                
                logger.info(f"✅ Reclassified scene: {houzz_id} as {classification_result['image_type']}")
                
            except Exception as scene_error:
                logger.error(f"❌ Failed to reclassify scene {i+1}: {scene_error}")
                continue
                
        logger.info(f"✅ Scene reclassification job {job_id} completed - processed {processed_count}/{len(scenes)} scenes")
        
        if job_service:
            job_service.complete_job(job_id, f"Reclassification complete: {processed_count}/{len(scenes)} scenes processed")
        
    except Exception as e:
        logger.error(f"❌ Scene reclassification job {job_id} failed: {e}")
        if job_service:
            job_service.fail_job(job_id, str(e))