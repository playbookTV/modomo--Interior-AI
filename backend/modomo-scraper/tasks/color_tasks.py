"""
Phase 1: Color processing Celery tasks
"""
import structlog
from typing import Dict, Any, List
from celery import current_task
from celery_app import celery_app
from tasks import BaseTask, database_service

logger = structlog.get_logger(__name__)

@celery_app.task(bind=True, autoretry_for=(Exception,), retry_kwargs={'max_retries': 3, 'countdown': 60})
def run_color_processing_job(self, job_id: str, limit: int):
    """Background task for color processing of existing objects"""
    processed = 0
    total_objects = 0
    
    try:
        logger.info(f"🎨 Starting color processing job {job_id} for {limit} objects")
        BaseTask.update_job_progress(job_id, "running", 0, limit, "Initializing color processing...")
        
        # Get objects that need color processing
        objects_data = get_objects_for_color_processing(limit)
        total_objects = len(objects_data)
        
        if total_objects == 0:
            logger.info(f"No objects found for color processing")
            return BaseTask.complete_job(job_id, 0, 0, {"message": "No objects need color processing"})
        
        logger.info(f"Found {total_objects} objects for color processing")
        BaseTask.update_job_progress(job_id, "running", 0, total_objects, f"Processing {total_objects} objects...")
        
        # Process each object
        for i, obj in enumerate(objects_data):
            try:
                object_id = obj["object_id"]
                scene_id = obj["scene_id"]
                
                # Get scene image URL
                scene_data = get_scene_data(scene_id)
                if not scene_data or not scene_data.get("image_url"):
                    logger.warning(f"No image URL for scene {scene_id}")
                    continue
                
                image_url = scene_data["image_url"]
                bbox = obj.get("bbox", [])
                
                # Extract colors for this object
                color_data = extract_object_colors(image_url, bbox)
                
                if color_data and not color_data.get("error"):
                    # Update object with color data
                    update_success = update_object_colors(object_id, color_data)
                    
                    if update_success:
                        processed += 1
                        logger.debug(f"✅ Processed colors for object {object_id}")
                    else:
                        logger.warning(f"⚠️ Failed to update colors for object {object_id}")
                else:
                    logger.warning(f"⚠️ Color extraction failed for object {object_id}: {color_data.get('error') if color_data else 'Unknown error'}")
                
                # Update progress
                progress_message = f"Processed {processed} of {total_objects} objects"
                BaseTask.update_job_progress(job_id, "running", processed, total_objects, progress_message)
                BaseTask.update_celery_progress(processed, total_objects, progress_message)
                
                # Log progress every 10 objects
                if (i + 1) % 10 == 0:
                    logger.info(f"Color processing progress: {processed}/{total_objects} objects")
                    
            except Exception as obj_error:
                logger.error(f"Error processing object {obj.get('object_id', 'unknown')}: {obj_error}")
                continue
        
        # Complete the job
        result = {
            "processed": processed,
            "total": total_objects,
            "success_rate": f"{(processed/total_objects)*100:.1f}%" if total_objects > 0 else "0%",
            "message": f"Color processing completed: {processed}/{total_objects} objects"
        }
        
        logger.info(f"🎨 Color processing job {job_id} completed: {processed}/{total_objects}")
        return BaseTask.complete_job(job_id, processed, total_objects, result)
        
    except Exception as e:
        logger.error(f"❌ Color processing job {job_id} failed: {e}")
        BaseTask.handle_task_error(job_id, e, processed, total_objects or limit)
        raise

def get_objects_for_color_processing(limit: int) -> List[Dict[str, Any]]:
    """Get objects that need color processing"""
    try:
        if not database_service:
            return []
        
        # Get objects without color metadata
        result = database_service.supabase.table("detected_objects").select(
            "object_id, scene_id, bbox, metadata"
        ).is_("metadata->colors", "null").limit(limit).execute()
        
        return result.data or []
        
    except Exception as e:
        logger.error(f"Failed to get objects for color processing: {e}")
        return []

def get_scene_data(scene_id: str) -> Dict[str, Any]:
    """Get scene data including image URL"""
    try:
        if not database_service:
            return {}
        
        result = database_service.supabase.table("scenes").select(
            "scene_id, image_url, image_r2_key"
        ).eq("scene_id", scene_id).execute()
        
        return result.data[0] if result.data else {}
        
    except Exception as e:
        logger.error(f"Failed to get scene data for {scene_id}: {e}")
        return {}

def extract_object_colors(image_url: str, bbox: List[float]) -> Dict[str, Any]:
    """Extract colors from object region"""
    try:
        # Import detection service locally to avoid circular imports
        from services.detection_service import DetectionService
        
        # Create a temporary detection service instance
        # In practice, you might want to initialize this once and reuse
        detection_service = DetectionService(color_extractor=None)
        
        # Initialize color extractor if not available
        if not detection_service.color_extractor:
            try:
                from models.color_extractor import ColorExtractor
                detection_service.color_extractor = ColorExtractor()
            except ImportError:
                return {"error": "Color extractor not available"}
        
        # Extract colors using the detection service
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        color_data = loop.run_until_complete(
            detection_service.extract_colors_from_url(image_url, bbox)
        )
        loop.close()
        
        return color_data
        
    except Exception as e:
        logger.error(f"Color extraction failed: {e}")
        return {"error": str(e)}

def update_object_colors(object_id: str, color_data: Dict[str, Any]) -> bool:
    """Update object with extracted color data"""
    try:
        if not database_service:
            return False
        
        # Update metadata with color information
        current_metadata = get_object_metadata(object_id)
        current_metadata["colors"] = color_data
        
        result = database_service.supabase.table("detected_objects").update({
            "metadata": current_metadata
        }).eq("object_id", object_id).execute()
        
        return bool(result.data)
        
    except Exception as e:
        logger.error(f"Failed to update object colors for {object_id}: {e}")
        return False

def get_object_metadata(object_id: str) -> Dict[str, Any]:
    """Get current object metadata"""
    try:
        if not database_service:
            return {}
        
        result = database_service.supabase.table("detected_objects").select(
            "metadata"
        ).eq("object_id", object_id).execute()
        
        if result.data:
            return result.data[0].get("metadata", {})
        return {}
        
    except Exception as e:
        logger.error(f"Failed to get object metadata for {object_id}: {e}")
        return {}

@celery_app.task(bind=True)
def extract_single_object_colors(self, object_id: str, image_url: str, bbox: List[float]):
    """Extract colors for a single object (used for real-time processing)"""
    try:
        logger.info(f"🎨 Extracting colors for object {object_id}")
        
        # Extract colors
        color_data = extract_object_colors(image_url, bbox)
        
        if color_data and not color_data.get("error"):
            # Update object
            success = update_object_colors(object_id, color_data)
            
            if success:
                logger.info(f"✅ Successfully extracted colors for object {object_id}")
                return {"success": True, "object_id": object_id, "colors": color_data}
            else:
                logger.error(f"❌ Failed to update colors for object {object_id}")
                return {"success": False, "object_id": object_id, "error": "Database update failed"}
        else:
            error_msg = color_data.get("error", "Unknown error") if color_data else "No color data returned"
            logger.error(f"❌ Color extraction failed for object {object_id}: {error_msg}")
            return {"success": False, "object_id": object_id, "error": error_msg}
            
    except Exception as e:
        logger.error(f"❌ Single color extraction failed for object {object_id}: {e}")
        return {"success": False, "object_id": object_id, "error": str(e)}