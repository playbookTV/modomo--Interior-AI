"""
Phase 2: AI detection pipeline Celery tasks
"""
import os
import structlog
import uuid
from typing import Dict, Any, List
from celery_app import celery_app
from tasks import BaseTask, database_service

logger = structlog.get_logger(__name__)

@celery_app.task(bind=True, autoretry_for=(Exception,), retry_kwargs={'max_retries': 3, 'countdown': 120})
def run_detection_pipeline(self, job_id: str, image_url: str, scene_id: str = None):
    """Complete AI detection pipeline for a single image"""
    try:
        logger.info(f"🤖 Starting AI detection pipeline job {job_id} for {image_url}")
        BaseTask.update_job_progress(job_id, "running", 0, 5, "Initializing AI detection pipeline...")
        
        # Step 1: Initialize detection service
        detection_service = get_detection_service()
        if not detection_service:
            logger.warning("AI detection service not available - skipping detection task")
            return BaseTask.complete_job(job_id, 5, 5, {
                "message": "AI detection skipped - service not available on this worker",
                "skipped": True,
                "detections": []
            })
        
        if not detection_service.is_available():
            logger.warning("AI detection service not ready - skipping detection task")
            return BaseTask.complete_job(job_id, 5, 5, {
                "message": "AI detection skipped - service not ready",
                "skipped": True,
                "detections": []
            })
        
        BaseTask.update_job_progress(job_id, "running", 1, 5, "Running object detection...")
        BaseTask.update_celery_progress(1, 5, "Running object detection...")
        
        # Step 2: Run complete detection pipeline
        from config.taxonomy import MODOMO_TAXONOMY
        detections = None
        
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        detections = loop.run_until_complete(
            detection_service.run_detection_pipeline(image_url, job_id, MODOMO_TAXONOMY)
        )
        loop.close()
        
        if not detections:
            logger.warning(f"No objects detected in {image_url}")
            return BaseTask.complete_job(job_id, 5, 5, {"detections": [], "message": "No objects detected"})
        
        BaseTask.update_job_progress(job_id, "running", 3, 5, f"Detected {len(detections)} objects, storing results...")
        BaseTask.update_celery_progress(3, 5, f"Storing {len(detections)} detections...")
        
        # Step 3: Store detections in database
        stored_objects = []
        if scene_id:
            stored_objects = store_detections_in_database(scene_id, detections)
        
        BaseTask.update_job_progress(job_id, "running", 4, 5, "Finalizing results...")
        BaseTask.update_celery_progress(4, 5, "Finalizing results...")
        
        # Step 4: Prepare result
        result = {
            "image_url": image_url,
            "scene_id": scene_id,
            "detections": detections,
            "stored_objects": len(stored_objects),
            "total_detected": len(detections),
            "message": f"Successfully detected and processed {len(detections)} objects"
        }
        
        logger.info(f"🤖 Detection pipeline job {job_id} completed: {len(detections)} objects detected")
        return BaseTask.complete_job(job_id, 5, 5, result)
        
    except Exception as e:
        logger.error(f"❌ Detection pipeline job {job_id} failed: {e}")
        BaseTask.handle_task_error(job_id, e, 0, 5)
        raise

@celery_app.task(bind=True, autoretry_for=(Exception,), retry_kwargs={'max_retries': 3, 'countdown': 180})
def run_batch_detection(self, job_id: str, scene_ids: List[str]):
    """Run detection pipeline on multiple scenes"""
    processed = 0
    total_scenes = len(scene_ids)
    all_results = []
    
    try:
        logger.info(f"🤖 Starting batch detection job {job_id} for {total_scenes} scenes")
        BaseTask.update_job_progress(job_id, "running", 0, total_scenes, "Starting batch detection...")
        
        detection_service = get_detection_service()
        if not detection_service:
            logger.warning("AI detection service not available - skipping batch detection")
            return BaseTask.complete_job(job_id, total_scenes, total_scenes, {
                "message": "Batch AI detection skipped - service not available on this worker",
                "processed_scenes": 0,
                "total_scenes": total_scenes,
                "skipped": True,
                "scene_results": []
            })
        
        if not detection_service.is_available():
            logger.warning("AI detection service not ready - skipping batch detection")
            return BaseTask.complete_job(job_id, total_scenes, total_scenes, {
                "message": "Batch AI detection skipped - service not ready",
                "processed_scenes": 0,
                "total_scenes": total_scenes,
                "skipped": True,
                "scene_results": []
            })
        
        # Process each scene
        for i, scene_id in enumerate(scene_ids):
            try:
                # Get scene data
                scene_data = get_scene_data(scene_id)
                if not scene_data or not scene_data.get("image_url"):
                    logger.warning(f"No image URL for scene {scene_id}")
                    continue
                
                image_url = scene_data["image_url"]
                
                # Run detection pipeline
                from config.taxonomy import MODOMO_TAXONOMY
                
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                detections = loop.run_until_complete(
                    detection_service.run_detection_pipeline(image_url, f"{job_id}_{scene_id}", MODOMO_TAXONOMY)
                )
                loop.close()
                
                # Store detections
                stored_objects = []
                if detections:
                    stored_objects = store_detections_in_database(scene_id, detections)
                
                scene_result = {
                    "scene_id": scene_id,
                    "image_url": image_url,
                    "detected_objects": len(detections) if detections else 0,
                    "stored_objects": len(stored_objects)
                }
                all_results.append(scene_result)
                
                processed += 1
                
                # Update progress
                progress_message = f"Processed {processed}/{total_scenes} scenes"
                BaseTask.update_job_progress(job_id, "running", processed, total_scenes, progress_message)
                BaseTask.update_celery_progress(processed, total_scenes, progress_message)
                
                logger.info(f"Completed detection for scene {scene_id}: {len(detections) if detections else 0} objects")
                
            except Exception as scene_error:
                logger.error(f"Error processing scene {scene_id}: {scene_error}")
                all_results.append({
                    "scene_id": scene_id,
                    "error": str(scene_error),
                    "detected_objects": 0,
                    "stored_objects": 0
                })
                continue
        
        # Prepare final result
        total_detected = sum(r.get("detected_objects", 0) for r in all_results)
        total_stored = sum(r.get("stored_objects", 0) for r in all_results)
        
        result = {
            "processed_scenes": processed,
            "total_scenes": total_scenes,
            "total_objects_detected": total_detected,
            "total_objects_stored": total_stored,
            "success_rate": f"{(processed/total_scenes)*100:.1f}%" if total_scenes > 0 else "0%",
            "scene_results": all_results,
            "message": f"Batch detection completed: {processed}/{total_scenes} scenes, {total_detected} objects detected"
        }
        
        logger.info(f"🤖 Batch detection job {job_id} completed: {processed}/{total_scenes} scenes")
        return BaseTask.complete_job(job_id, processed, total_scenes, result)
        
    except Exception as e:
        logger.error(f"❌ Batch detection job {job_id} failed: {e}")
        BaseTask.handle_task_error(job_id, e, processed, total_scenes)
        raise

@celery_app.task(bind=True)
def reprocess_scene_objects(self, job_id: str, scene_id: str, force_redetection: bool = False):
    """Reprocess objects for a specific scene"""
    try:
        logger.info(f"🔄 Reprocessing objects for scene {scene_id}")
        BaseTask.update_job_progress(job_id, "running", 0, 3, "Initializing scene reprocessing...")
        
        # Get scene data
        scene_data = get_scene_data(scene_id)
        if not scene_data or not scene_data.get("image_url"):
            raise Exception(f"No image URL found for scene {scene_id}")
        
        image_url = scene_data["image_url"]
        
        BaseTask.update_job_progress(job_id, "running", 1, 3, "Running detection pipeline...")
        
        # If force redetection, delete existing objects first
        if force_redetection:
            delete_scene_objects(scene_id)
            logger.info(f"Deleted existing objects for scene {scene_id}")
        
        # Run detection pipeline
        detection_service = get_detection_service()
        if not detection_service:
            logger.warning("AI detection service not available - skipping reprocessing")
            return BaseTask.complete_job(job_id, 3, 3, {
                "message": "Scene reprocessing skipped - AI service not available on this worker",
                "scene_id": scene_id,
                "skipped": True,
                "detected_objects": 0
            })
        
        if not detection_service.is_available():
            logger.warning("AI detection service not ready - skipping reprocessing")
            return BaseTask.complete_job(job_id, 3, 3, {
                "message": "Scene reprocessing skipped - AI service not ready",
                "scene_id": scene_id,
                "skipped": True,
                "detected_objects": 0
            })
        
        from config.taxonomy import MODOMO_TAXONOMY
        
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        detections = loop.run_until_complete(
            detection_service.run_detection_pipeline(image_url, job_id, MODOMO_TAXONOMY)
        )
        loop.close()
        
        BaseTask.update_job_progress(job_id, "running", 2, 3, "Storing detection results...")
        
        # Store new detections
        stored_objects = []
        if detections:
            stored_objects = store_detections_in_database(scene_id, detections)
        
        result = {
            "scene_id": scene_id,
            "image_url": image_url,
            "force_redetection": force_redetection,
            "detected_objects": len(detections) if detections else 0,
            "stored_objects": len(stored_objects),
            "message": f"Scene reprocessing completed: {len(detections) if detections else 0} objects detected"
        }
        
        logger.info(f"🔄 Scene {scene_id} reprocessing completed: {len(detections) if detections else 0} objects")
        return BaseTask.complete_job(job_id, 3, 3, result)
        
    except Exception as e:
        logger.error(f"❌ Scene reprocessing failed for {scene_id}: {e}")
        BaseTask.handle_task_error(job_id, e, 0, 3)
        raise

def get_detection_service():
    """Get Railway-based detection service (HTTP client)"""
    try:
        # Instead of loading models locally, create HTTP client to Railway
        from services.railway_detection_client import RailwayDetectionClient
        
        # Railway service URL from environment
        railway_url = os.getenv("RAILWAY_AI_SERVICE_URL", "https://ovalay-recruitment-production.up.railway.app")
        
        return RailwayDetectionClient(base_url=railway_url)
        
    except ImportError:
        logger.warning("Railway detection client not available - creating fallback")
        # Create a simple HTTP client as fallback
        return RailwayDetectionHTTPClient()
    except Exception as e:
        logger.error(f"Failed to initialize Railway detection service: {e}")
        return None


class RailwayDetectionHTTPClient:
    """Simple HTTP client for Railway AI detection service"""
    
    def __init__(self, base_url: str = None):
        self.base_url = base_url or os.getenv("RAILWAY_AI_SERVICE_URL", "https://ovalay-recruitment-production.up.railway.app")
        
    def is_available(self) -> bool:
        """Check if Railway service is available"""
        try:
            import requests
            response = requests.get(f"{self.base_url}/health", timeout=10)
            return response.status_code == 200
        except:
            return False
    
    async def run_detection_pipeline(self, image_url: str, job_id: str, taxonomy: dict) -> list:
        """Run detection pipeline via Railway API"""
        try:
            import requests
            import asyncio
            
            # Make HTTP request to Railway detection endpoint
            payload = {
                "image_url": image_url,
                "scene_id": None,  # We'll use job_id for tracking
                "taxonomy": taxonomy
            }
            
            # Use requests in a thread to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: requests.post(
                    f"{self.base_url}/detect/process",
                    json=payload,
                    timeout=300  # 5 minute timeout for AI processing
                )
            )
            
            if response.status_code == 200:
                result = response.json()
                if "results" in result:
                    logger.info(f"✅ Railway detection successful: {len(result['results'])} objects detected")
                    return result["results"]
                elif "job_id" in result:
                    # If Railway returns a job_id, we need to poll for results
                    railway_job_id = result["job_id"]
                    return await self._poll_railway_job(railway_job_id)
                else:
                    logger.warning("Railway detection returned unexpected format")
                    return []
            else:
                logger.error(f"Railway detection failed: {response.status_code} - {response.text}")
                return []
                
        except Exception as e:
            logger.error(f"Railway detection pipeline failed: {e}")
            return []
    
    async def _poll_railway_job(self, railway_job_id: str, max_polls: int = 60) -> list:
        """Poll Railway job status until completion"""
        import requests
        import asyncio
        
        for i in range(max_polls):
            try:
                response = requests.get(f"{self.base_url}/jobs/{railway_job_id}/status", timeout=10)
                if response.status_code == 200:
                    job_data = response.json()
                    
                    if job_data.get("status") == "completed":
                        # Job completed, get results
                        if "result" in job_data and "results" in job_data["result"]:
                            return job_data["result"]["results"]
                        return []
                    elif job_data.get("status") == "failed":
                        logger.error(f"Railway job {railway_job_id} failed: {job_data.get('error', 'Unknown error')}")
                        return []
                    else:
                        # Still processing, wait and poll again
                        await asyncio.sleep(5)
                        continue
                else:
                    logger.warning(f"Could not check Railway job status: {response.status_code}")
                    await asyncio.sleep(5)
                    
            except Exception as poll_error:
                logger.warning(f"Error polling Railway job {railway_job_id}: {poll_error}")
                await asyncio.sleep(5)
        
        logger.warning(f"Railway job {railway_job_id} polling timed out after {max_polls * 5} seconds")
        return []

def get_scene_data(scene_id: str) -> Dict[str, Any]:
    """Get scene data from database"""
    try:
        if not database_service:
            return {}
        
        result = database_service.supabase.table("scenes").select(
            "scene_id, image_url, image_r2_key, houzz_id"
        ).eq("scene_id", scene_id).execute()
        
        return result.data[0] if result.data else {}
        
    except Exception as e:
        logger.error(f"Failed to get scene data for {scene_id}: {e}")
        return {}

def store_detections_in_database(scene_id: str, detections: List[Dict[str, Any]]) -> List[str]:
    """Store detection results in database"""
    stored_object_ids = []
    
    try:
        if not database_service:
            return stored_object_ids
        
        for detection in detections:
            try:
                # Prepare object data
                object_data = {
                    "scene_id": scene_id,
                    "category": detection.get("category", "unknown"),
                    "confidence": float(detection.get("confidence", 0.0)),
                    "bbox": detection.get("bbox", []),
                    "tags": detection.get("tags", []),
                    "metadata": {
                        "embedding": detection.get("embedding", []),
                        "colors": detection.get("color_data"),
                        "detection_model": "GroundingDINO+SAM2",
                        "processed_at": str(uuid.uuid4())
                    }
                }
                
                # Add mask information if available
                if detection.get("mask_path"):
                    object_data["mask_url"] = detection["mask_path"]
                
                # Insert into database
                result = database_service.supabase.table("detected_objects").insert(object_data).execute()
                
                if result.data:
                    object_id = result.data[0]["object_id"]
                    stored_object_ids.append(object_id)
                    logger.debug(f"Stored object {object_id} for scene {scene_id}")
                
            except Exception as obj_error:
                logger.error(f"Failed to store detection object: {obj_error}")
                continue
        
        logger.info(f"Stored {len(stored_object_ids)} objects for scene {scene_id}")
        return stored_object_ids
        
    except Exception as e:
        logger.error(f"Failed to store detections for scene {scene_id}: {e}")
        return stored_object_ids

def delete_scene_objects(scene_id: str) -> bool:
    """Delete all objects for a scene"""
    try:
        if not database_service:
            return False
        
        result = database_service.supabase.table("detected_objects").delete().eq("scene_id", scene_id).execute()
        logger.info(f"Deleted objects for scene {scene_id}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to delete objects for scene {scene_id}: {e}")
        return False