"""
Dataset import/export endpoints extracted from main file
"""
import uuid
from fastapi import FastAPI, HTTPException, Query
from core.dependencies import get_database_service, get_job_service
from utils.logging import get_logger

logger = get_logger(__name__)


def register_dataset_routes(app: FastAPI):
    """Register dataset-related endpoints to the app"""
    
    @app.get("/export/training-dataset")
    async def export_training_dataset(
        format: str = Query("json", description="Export format (json, yaml)"),
        split_ratio: str = Query("70:20:10", description="Train:Val:Test split ratio")
    ):
        """Export dataset for training"""
        try:
            database_service = get_database_service()
            if not database_service:
                raise HTTPException(status_code=503, detail="Database service not available")
            
            # Parse split ratio
            try:
                ratios = [float(x) for x in split_ratio.split(':')]
                if len(ratios) != 3 or sum(ratios) != 100:
                    raise ValueError("Split ratios must sum to 100")
            except:
                raise HTTPException(status_code=400, detail="Invalid split ratio format. Use 'train:val:test' (e.g., '70:20:10')")
            
            # Get dataset statistics
            stats = await database_service.get_dataset_stats()
            
            return {
                "export_format": format,
                "split_ratios": {
                    "train": ratios[0],
                    "val": ratios[1], 
                    "test": ratios[2]
                },
                "dataset_stats": stats,
                "message": "Dataset export prepared. Full export would be implemented as background job.",
                "note": "This is a simplified version for the refactored architecture"
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"❌ Dataset export error: {e}")
            raise HTTPException(status_code=500, detail=f"Dataset export failed: {str(e)}")

    @app.post("/import/huggingface-dataset")
    async def import_huggingface_dataset(
        dataset: str = Query("sk2003/houzzdata", description="HuggingFace dataset ID (e.g., username/dataset-name)"),
        offset: int = Query(0, description="Starting offset in dataset"),
        limit: int = Query(50, description="Number of images to import and process"),
        include_detection: bool = Query(True, description="Run AI detection on imported images")
    ):
        """Import any HuggingFace dataset and process with AI"""
        try:
            job_id = str(uuid.uuid4())
            
            database_service = get_database_service()
            job_service = get_job_service()
            
            # Create job in database for persistent tracking
            if database_service:
                await database_service.create_job_in_database(
                    job_id=job_id,
                    job_type="import",
                    total_items=limit,
                    parameters={
                        "dataset": dataset,
                        "offset": offset,
                        "limit": limit,
                        "include_detection": include_detection
                    }
                )
            
            # Create job in Redis if available
            if job_service and job_service.is_available():
                job_service.create_job(
                    job_id=job_id,
                    job_type="import",
                    total=limit,
                    message=f"Importing {limit} images from {dataset}",
                    dataset=dataset,
                    offset=str(offset),
                    include_detection=str(include_detection)
                )
            
            # Start Celery task for background processing
            try:
                from tasks.scraping_tasks import import_huggingface_dataset as import_task
                task = import_task.delay(job_id, dataset, offset, limit, include_detection)
                
                return {
                    "job_id": job_id,
                    "task_id": task.id, 
                    "status": "running",
                    "message": f"Started importing {limit} images from HuggingFace dataset '{dataset}' (offset: {offset})",
                    "dataset": dataset,
                    "features": ["import", "object_detection", "segmentation", "embeddings"] if include_detection else ["import"],
                    "celery_task": "import_huggingface_dataset"
                }
            except ImportError:
                # Fallback if Celery not available
                return {
                    "job_id": job_id, 
                    "status": "pending",
                    "message": f"Started importing {limit} images from HuggingFace dataset '{dataset}' (offset: {offset})",
                    "dataset": dataset,
                    "features": ["import", "object_detection", "segmentation", "embeddings"] if include_detection else ["import"],
                    "note": "Celery not available - job created but not executed"
                }
            
        except Exception as e:
            logger.error(f"❌ Dataset import job creation failed: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to start dataset import: {str(e)}")

    @app.get("/classify/test")
    async def test_classification(
        image_url: str = Query(..., description="Image URL to test classification"),
        caption: str = Query(None, description="Optional caption/description")
    ):
        """Test image classification on a single image"""
        try:
            # Import classification function from tasks
            try:
                from tasks.classification_tasks import classify_image_type
                classification = await classify_image_type(image_url, caption)
                
                return {
                    "image_url": image_url,
                    "caption": caption,
                    "classification": classification,
                    "status": "success"
                }
            except ImportError:
                # Fallback simple classification if tasks module not available
                return {
                    "image_url": image_url,
                    "caption": caption,
                    "classification": {
                        "image_type": "scene",
                        "confidence": 0.8,
                        "reason": "fallback_classification",
                        "is_primary_object": False
                    },
                    "status": "success",
                    "note": "Using fallback classification - tasks module not available"
                }
        except Exception as e:
            logger.error(f"❌ Classification test error: {e}")
            return {
                "image_url": image_url,
                "error": str(e),
                "status": "failed"
            }

    @app.post("/classify/reclassify-scenes")
    async def reclassify_scenes(
        limit: int = Query(100, description="Number of scenes to reclassify"),
        force_reclassify: bool = Query(False, description="Force reclassify all scenes")
    ):
        """Start scene reclassification job"""
        try:
            job_id = str(uuid.uuid4())
            
            database_service = get_database_service()
            job_service = get_job_service()
            
            # Create job in database if available
            if database_service:
                await database_service.create_job_in_database(
                    job_id=job_id,
                    job_type="scene_reclassification",
                    total_items=limit,
                    parameters={"limit": limit, "operation": "scene_reclassification", "force_reclassify": force_reclassify}
                )
            
            # Create job in Redis if available  
            if job_service and job_service.is_available():
                job_service.create_job(
                    job_id=job_id,
                    job_type="scene_reclassification",
                    total=limit,
                    message=f"Reclassifying {limit} scenes",
                    force_reclassify=str(force_reclassify)
                )
            
            # Start Celery task for background processing
            try:
                from tasks.classification_tasks import run_scene_reclassification_job
                task = run_scene_reclassification_job.delay(job_id, limit, force_reclassify)
                
                return {
                    "job_id": job_id,
                    "task_id": task.id,
                    "message": f"Scene reclassification job started for {limit} scenes",
                    "status": "running",
                    "parameters": {
                        "limit": limit,
                        "force_reclassify": force_reclassify
                    },
                    "celery_task": "run_scene_reclassification_job"
                }
            except ImportError:
                # Fallback if Celery not available
                return {
                    "job_id": job_id,
                    "message": f"Scene reclassification job started for {limit} scenes",
                    "status": "pending",
                    "parameters": {
                        "limit": limit,
                        "force_reclassify": force_reclassify
                    },
                    "note": "Celery not available - job created but not executed"
                }
            
        except Exception as e:
            logger.error(f"❌ Scene reclassification job creation failed: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to start scene reclassification: {str(e)}")