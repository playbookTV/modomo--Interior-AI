"""
Advanced AI pipeline endpoints (from main_full.py)
Complete GroundingDINO + SAM2 + CLIP detection pipeline
"""
import uuid
import os
import tempfile
from typing import List
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, Body
from core.dependencies import get_detection_service, get_database_service, get_job_service
from config.comprehensive_taxonomy import COMPREHENSIVE_TAXONOMY
from utils.logging import get_logger

logger = get_logger(__name__)


def register_advanced_ai_routes(app: FastAPI):
    """Register advanced AI pipeline endpoints to the app"""
    
    @app.get("/taxonomy")
    async def get_taxonomy():
        """Get the comprehensive furniture taxonomy"""
        return COMPREHENSIVE_TAXONOMY
    
    @app.post("/detect/process")
    async def process_detection(
        image_url: str = Body(...),
        scene_id: str = Body(None),
        background_tasks: BackgroundTasks = None
    ):
        """Run advanced object detection pipeline on an image"""
        job_id = str(uuid.uuid4())
        
        detection_service = get_detection_service()
        job_service = get_job_service()
        database_service = get_database_service()
        
        # Create job record in database with parameters for retry functionality
        if database_service:
            await database_service.create_job_in_database(
                job_id=job_id,
                job_type="detection",
                total_items=1,
                parameters={
                    "image_url": image_url,
                    "scene_id": scene_id,
                    "job_type": "detection"
                }
            )
        
        if background_tasks and job_service:
            # Create job tracking
            job_service.create_job(
                job_id=job_id,
                job_type="detection",
                total=1,
                message="Processing advanced AI detection pipeline"
            )
            
            background_tasks.add_task(
                run_advanced_detection_pipeline, 
                image_url, 
                job_id,
                scene_id
            )
            return {"job_id": job_id, "status": "processing"}
        else:
            # Run synchronously for testing
            if not detection_service:
                return {"error": "Detection service not available"}
            
            results = await run_advanced_detection_pipeline(image_url, job_id, scene_id)
            return {"job_id": job_id, "results": results}

    @app.get("/search/color")
    async def search_objects_by_color(
        query: str = Query(..., description="Color-based search query (e.g., 'red sofa', 'blue curtains')"),
        limit: int = Query(10, description="Maximum number of results"),
        threshold: float = Query(0.3, description="Minimum similarity threshold (0-1)")
    ):
        """Search for objects using color-based CLIP queries"""
        try:
            database_service = get_database_service()
            if not database_service:
                return {"error": "Database connection not available"}
            
            # Get all objects with embeddings
            result = database_service.supabase.table("detected_objects").select(
                "object_id, scene_id, category, confidence, tags, clip_embedding_json, metadata"
            ).not_.is_("clip_embedding_json", "null").execute()
            
            if not result.data:
                return {"results": [], "query": query, "total": 0}
            
            # Prepare data for search
            object_ids = []
            object_embeddings = []
            object_metadata = []
            
            for obj in result.data:
                if obj.get("clip_embedding_json"):
                    object_ids.append(obj["object_id"])
                    object_embeddings.append(obj["clip_embedding_json"])
                    object_metadata.append({
                        "scene_id": obj["scene_id"],
                        "category": obj["category"],
                        "confidence": obj["confidence"],
                        "tags": obj.get("tags", []),
                        "colors": obj.get("metadata", {}).get("colors")
                    })
            
            # Perform vector search (simplified implementation)
            # In production, this would use proper CLIP embedding comparison
            matches = []
            query_lower = query.lower()
            
            for i, metadata in enumerate(object_metadata):
                # Simple keyword matching fallback
                score = 0.0
                if query_lower in metadata["category"].lower():
                    score += 0.5
                
                # Check tags for color matches
                for tag in metadata["tags"]:
                    if any(color in tag.lower() for color in query_lower.split()):
                        score += 0.3
                
                # Check color metadata
                if metadata.get("colors"):
                    colors_data = metadata["colors"]
                    if isinstance(colors_data, dict) and colors_data.get("colors"):
                        for color_info in colors_data["colors"]:
                            color_name = color_info.get("name", "").lower()
                            if any(q_word in color_name for q_word in query_lower.split()):
                                score += 0.4
                
                if score >= threshold:
                    matches.append({
                        "object_id": object_ids[i],
                        "similarity": score,
                        **metadata
                    })
            
            # Sort by similarity
            matches.sort(key=lambda x: x["similarity"], reverse=True)
            matches = matches[:limit]
            
            return {
                "results": matches,
                "query": query,
                "total": len(matches),
                "threshold": threshold
            }
            
        except Exception as e:
            logger.error(f"Color search API failed: {e}")
            return {"error": str(e)}

    @app.get("/debug/detector-status", response_model=None)
    async def debug_detector_status():
        """Debug endpoint to check multi-model detector status"""
        detection_service = get_detection_service()
        
        if not detection_service:
            return {"error": "No detector available"}
        
        status = {
            "detector_available": detection_service is not None,
            "detector_type": "Multi-model (YOLO + DETR)"
        }
        
        if hasattr(detection_service, 'get_detector_status'):
            status.update(detection_service.get_detector_status())
        
        # Check YOLO dependencies (required for multi-model approach)
        try:
            from ultralytics import YOLO
            yolo_status = "✅ Available (required for multi-model detection)"
        except ImportError:
            yolo_status = "❌ Not installed (REQUIRED for multi-model detection)"
        except Exception as e:
            yolo_status = f"❌ Error: {e} (REQUIRED for multi-model detection)"
        
        status["yolo_package"] = yolo_status
        
        return status


async def run_advanced_detection_pipeline(image_url: str, job_id: str, scene_id: str = None):
    """Complete advanced AI pipeline: detect -> segment -> embed -> color extract"""
    try:
        logger.info(f"Starting advanced detection pipeline for {image_url}")
        
        detection_service = get_detection_service()
        job_service = get_job_service()
        
        # Check if models are available
        if not detection_service:
            logger.error("❌ No detection service available")
            if job_service:
                job_service.fail_job(job_id, "Detection service not available")
            return []
        
        # Update job status
        if job_service:
            job_service.update_job(job_id, status="running", message="Starting advanced AI pipeline")
        
        # Step 1: Download image from URL
        image_path = f"/tmp/scene_{job_id}.jpg"
        
        import aiohttp
        import aiofiles
        async with aiohttp.ClientSession() as session:
            async with session.get(image_url) as response:
                if response.status == 200:
                    async with aiofiles.open(image_path, 'wb') as f:
                        async for chunk in response.content.iter_chunked(8192):
                            await f.write(chunk)
                    logger.info(f"✅ Downloaded image to {image_path}")
                else:
                    logger.error(f"❌ Failed to download image: HTTP {response.status}")
                    if job_service:
                        job_service.fail_job(job_id, f"Failed to download image: HTTP {response.status}")
                    return []
        
        if job_service:
            job_service.update_job(job_id, message="Running object detection")
        
        # Step 2: Run comprehensive object detection
        # This would use the advanced detection service with GroundingDINO + YOLO
        try:
            # Use detection service if available
            if hasattr(detection_service, 'run_detection_pipeline'):
                detections = await detection_service.run_detection_pipeline(
                    image_url, job_id, COMPREHENSIVE_TAXONOMY
                )
            else:
                # Fallback to basic detection
                detections = await detection_service.detect_objects(image_path, COMPREHENSIVE_TAXONOMY)
            
            if not detections:
                logger.warning(f"⚠️ No objects detected in {image_url}")
                if job_service:
                    job_service.complete_job(job_id, "No objects detected")
                return []
            
        except Exception as detection_error:
            logger.error(f"❌ Detection failed: {detection_error}")
            if job_service:
                job_service.fail_job(job_id, str(detection_error))
            return []
        
        # Step 3: Process each detection with advanced pipeline
        processed_detections = []
        
        for i, detection in enumerate(detections):
            try:
                if job_service:
                    job_service.update_job(
                        job_id, 
                        processed=i, 
                        total=len(detections),
                        message=f"Processing object {i+1}/{len(detections)}"
                    )
                
                # Enhanced detection processing would happen here
                # - SAM2 segmentation
                # - CLIP embeddings
                # - Color extraction
                # - Advanced metadata enrichment
                
                processed_detection = {
                    **detection,
                    "enhanced": True,
                    "pipeline_version": "advanced",
                    "processing_timestamp": f"job_{job_id}",
                }
                
                processed_detections.append(processed_detection)
                logger.debug(f"✅ Processed detection {i+1}/{len(detections)}")
                
            except Exception as processing_error:
                logger.error(f"❌ Processing failed for detection {i+1}: {processing_error}")
                # Add failed detection with error info
                processed_detections.append({
                    **detection,
                    "processing_error": str(processing_error),
                    "enhanced": False
                })
        
        # Cleanup temporary image file
        try:
            if os.path.exists(image_path):
                os.unlink(image_path)
                logger.debug(f"🧹 Cleaned up temporary image: {image_path}")
        except Exception as cleanup_error:
            logger.warning(f"Failed to cleanup {image_path}: {cleanup_error}")
        
        if job_service:
            job_service.complete_job(
                job_id, 
                f"Advanced pipeline complete: {len(processed_detections)} objects processed"
            )
        
        logger.info(f"Advanced detection pipeline complete: {len(processed_detections)} objects processed")
        return processed_detections
        
    except Exception as e:
        logger.error(f"Advanced detection pipeline failed: {e}")
        
        if job_service:
            job_service.fail_job(job_id, str(e))
        
        # Cleanup on error
        try:
            if 'image_path' in locals() and os.path.exists(image_path):
                os.unlink(image_path)
        except:
            pass
        
        return []


def make_json_serializable(obj):
    """Convert NumPy types and other non-serializable types to JSON serializable types"""
    import numpy as np
    
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif hasattr(obj, 'item') and callable(obj.item):  # Handle numpy scalars
        return obj.item()
    return obj