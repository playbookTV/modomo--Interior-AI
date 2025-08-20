"""
Dataset export API routes
"""
import uuid
import json
from datetime import datetime
from fastapi import APIRouter, BackgroundTasks, Query, HTTPException, Depends
from typing import Dict, Any, List, Optional
import structlog

from core.dependencies import get_database_service, get_job_service
from config.taxonomy import MODOMO_TAXONOMY, get_all_categories

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/export", tags=["export"])


@router.post(", response_model=None/dataset")
async def start_dataset_export(
    background_tasks: BackgroundTasks,
    train_ratio: float = Query(0.7, description="Training set ratio (0.0-1.0)"),
    val_ratio: float = Query(0.2, description="Validation set ratio (0.0-1.0)"),
    test_ratio: float = Query(0.1, description="Test set ratio (0.0-1.0)"),
    db_service = Depends(get_database_service),
    job_service = Depends(get_job_service)
):
    """
    Start dataset export job with train/validation/test splits
    """
    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 0.001:
        raise HTTPException(
            status_code=400, 
            detail=f"Ratios must sum to 1.0, got {total_ratio}"
        )
    
    if any(ratio < 0 or ratio > 1 for ratio in [train_ratio, val_ratio, test_ratio]):
        raise HTTPException(
            status_code=400,
            detail="All ratios must be between 0.0 and 1.0"
        )
    
    export_id = str(uuid.uuid4())
    
    # Create job in database for persistent tracking
    if db_service:
        await db_service.create_job_in_database(
            job_id=export_id,
            job_type="export",
            total_items=0,  # Will be updated when we know total scenes
            parameters={
                "train_ratio": train_ratio,
                "val_ratio": val_ratio,
                "test_ratio": test_ratio,
                "export_type": "dataset_split"
            }
        )
    
    # Create job in Redis for real-time tracking
    if job_service:
        job_service.create_job(
            job_id=export_id,
            job_type="export",
            total=0,  # Will be updated
            message="Initializing dataset export"
        )
    
    # Start background export task
    background_tasks.add_task(
        run_dataset_export_task,
        export_id,
        train_ratio,
        val_ratio,
        test_ratio,
        db_service,
        job_service
    )
    
    return {
        "export_id": export_id,
        "status": "running",
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio
    }


@router.get(", response_model=None/")
async def get_all_exports(
    db_service = Depends(get_database_service)
):
    """
    Get all dataset export jobs
    """
    if not db_service or not db_service.supabase:
        raise HTTPException(status_code=503, detail="Database service not available")
    
    try:
        # Get export jobs from database
        result = db_service.supabase.table("scraping_jobs").select("*").eq(
            "job_type", "export"
        ).order("created_at", desc=True).execute()
        
        exports = []
        for job in result.data or []:
            export_data = {
                "export_id": job["job_id"],
                "status": job["status"],
                "created_at": job.get("created_at"),
                "completed_at": job.get("completed_at"),
                "parameters": job.get("parameters", {}),
                "progress": job.get("progress", 0),
                "total_items": job.get("total_items", 0),
                "processed_items": job.get("processed_items", 0),
                "error_message": job.get("error_message")
            }
            
            # Add computed fields
            params = job.get("parameters", {})
            export_data.update({
                "train_ratio": params.get("train_ratio", 0.7),
                "val_ratio": params.get("val_ratio", 0.2),
                "test_ratio": params.get("test_ratio", 0.1),
                "export_type": params.get("export_type", "dataset_split")
            })
            
            exports.append(export_data)
        
        return exports
        
    except Exception as e:
        logger.error(f"Failed to get export jobs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve exports: {str(e)}")


@router.get(", response_model=None/{export_id}/status")
async def get_export_status(
    export_id: str,
    job_service = Depends(get_job_service)
):
    """
    Get the status of a specific export job
    """
    if not job_service.is_available():
        raise HTTPException(status_code=503, detail="Job tracking not available")
    
    job_data = job_service.get_job(export_id)
    
    if not job_data:
        raise HTTPException(status_code=404, detail="Export job not found")
    
    return job_data


async def run_dataset_export_task(
    export_id: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    db_service = Depends(get_database_service),
    job_service = Depends(get_job_service)
):
    """Background task for dataset export"""
    try:
        logger.info(f"📦 Starting dataset export job {export_id}")
        
        if job_service:
            job_service.update_job(export_id, status="running", message="Loading approved scenes")
        
        if not db_service or not db_service.supabase:
            logger.error("❌ Database service not available")
            if job_service:
                job_service.fail_job(export_id, "Database service not available")
            return
        
        # Get approved scenes with their objects
        scenes_query = db_service.supabase.table("scenes").select(
            """
            scene_id, houzz_id, image_url, room_type, caption, metadata,
            image_type, is_primary_object, primary_category,
            detected_objects(object_id, category, bbox, confidence, approved, metadata)
            """
        ).eq("approved", True).order("created_at", desc=False)
        
        scenes_result = scenes_query.execute()
        scenes = scenes_result.data or []
        
        if not scenes:
            logger.warning("No approved scenes found for export")
            if job_service:
                job_service.complete_job(export_id, "No approved scenes found for export")
            return
        
        total_scenes = len(scenes)
        logger.info(f"📊 Found {total_scenes} approved scenes for export")
        
        # Update job total
        if job_service:
            job_service.update_job(export_id, total=total_scenes, message=f"Processing {total_scenes} scenes")
        
        # Calculate split sizes
        train_size = int(total_scenes * train_ratio)
        val_size = int(total_scenes * val_ratio)
        test_size = total_scenes - train_size - val_size
        
        # Create splits
        train_scenes = scenes[:train_size]
        val_scenes = scenes[train_size:train_size + val_size]
        test_scenes = scenes[train_size + val_size:]
        
        export_data = {
            "export_id": export_id,
            "created_at": datetime.utcnow().isoformat(),
            "total_scenes": total_scenes,
            "splits": {
                "train": {
                    "count": len(train_scenes),
                    "ratio": train_ratio,
                    "scenes": [format_scene_for_export(scene) for scene in train_scenes]
                },
                "validation": {
                    "count": len(val_scenes),
                    "ratio": val_ratio,
                    "scenes": [format_scene_for_export(scene) for scene in val_scenes]
                },
                "test": {
                    "count": len(test_scenes),
                    "ratio": test_ratio,
                    "scenes": [format_scene_for_export(scene) for scene in test_scenes]
                }
            },
            "taxonomy": MODOMO_TAXONOMY,
            "categories": get_all_categories(),
            "statistics": {
                "total_objects": sum(len(scene.get("detected_objects", [])) for scene in scenes),
                "approved_objects": sum(
                    len([obj for obj in scene.get("detected_objects", []) if obj.get("approved")])
                    for scene in scenes
                ),
                "room_types": list(set(scene.get("room_type") for scene in scenes if scene.get("room_type"))),
                "categories": list(set(
                    obj.get("category") 
                    for scene in scenes 
                    for obj in scene.get("detected_objects", [])
                    if obj.get("category") and obj.get("approved")
                ))
            }
        }
        
        # For now, just log the export completion
        # In a real implementation, you'd save this to R2/S3 storage
        logger.info(f"✅ Dataset export {export_id} completed:")
        logger.info(f"   - Train: {len(train_scenes)} scenes")
        logger.info(f"   - Validation: {len(val_scenes)} scenes") 
        logger.info(f"   - Test: {len(test_scenes)} scenes")
        logger.info(f"   - Total objects: {export_data['statistics']['total_objects']}")
        logger.info(f"   - Approved objects: {export_data['statistics']['approved_objects']}")
        
        if job_service:
            job_service.complete_job(export_id, f"Export complete: {total_scenes} scenes exported")
        
        # Update database job with results
        if db_service:
            await db_service.update_job_progress(
                job_id=export_id,
                processed_items=total_scenes,
                total_items=total_scenes,
                status="completed"
            )
        
    except Exception as e:
        logger.error(f"❌ Dataset export job {export_id} failed: {e}")
        if job_service:
            job_service.fail_job(export_id, str(e))


def format_scene_for_export(scene: Dict[str, Any]) -> Dict[str, Any]:
    """Format scene data for export"""
    return {
        "scene_id": scene["scene_id"],
        "houzz_id": scene["houzz_id"],
        "image_url": scene["image_url"],
        "room_type": scene.get("room_type"),
        "caption": scene.get("caption"),
        "image_type": scene.get("image_type", "scene"),
        "is_primary_object": scene.get("is_primary_object", False),
        "primary_category": scene.get("primary_category"),
        "metadata": scene.get("metadata", {}),
        "objects": [
            {
                "object_id": obj["object_id"],
                "category": obj["category"],
                "bbox": obj["bbox"],
                "confidence": obj["confidence"],
                "approved": obj.get("approved", False),
                "metadata": obj.get("metadata", {})
            }
            for obj in scene.get("detected_objects", [])
            if obj.get("approved")
        ]
    }