"""
Admin utilities and cache migration endpoints (from main_full.py)
"""
import os
from fastapi import FastAPI, HTTPException
from core.dependencies import get_database_service
from utils.logging import get_logger

logger = get_logger(__name__)

# Global R2 client - will be set by app factory
r2_client = None
r2_bucket_name = None


def set_r2_client(client, bucket):
    """Set R2 client for migration utilities"""
    global r2_client, r2_bucket_name
    r2_client = client
    r2_bucket_name = bucket


def register_admin_utilities(app: FastAPI):
    """Register admin utility endpoints to the app"""
    
    @app.get("/admin/migrate-cache-to-r2")
    async def migrate_cache_to_r2():
        """Migrate all cached files (masks, maps) from local storage to R2"""
        try:
            if not r2_client:
                return {"error": "R2 client not available", "status": "failed"}
            
            migration_results = {
                "masks": {"uploaded": 0, "skipped": 0, "errors": []},
                "maps": {"uploaded": 0, "skipped": 0, "errors": []},
                "total_processed": 0,
                "status": "in_progress"
            }
            
            # Define cache directories to migrate
            cache_dirs = [
                ("/app/cache_volume/masks", "masks"),
                ("/app/cache_volume/maps", "training-data/maps"),
                ("/app/cache_volume/depth_maps", "training-data/maps"),
                ("/app/cache_volume/edge_maps", "training-data/maps"),
            ]
            
            for local_dir, r2_prefix in cache_dirs:
                if not os.path.exists(local_dir):
                    logger.warning(f"⚠️ Directory {local_dir} does not exist, skipping...")
                    continue
                
                logger.info(f"🔍 Scanning {local_dir} for files to migrate...")
                
                for root, dirs, files in os.walk(local_dir):
                    for file in files:
                        if file.startswith('.'):  # Skip hidden files
                            continue
                            
                        local_file_path = os.path.join(root, file)
                        relative_path = os.path.relpath(local_file_path, local_dir)
                        r2_key = f"{r2_prefix}/{relative_path}".replace("\\", "/")  # Ensure forward slashes
                        
                        try:
                            # Check if file already exists in R2
                            try:
                                r2_client.head_object(Bucket=r2_bucket_name, Key=r2_key)
                                # File exists, skip
                                if "masks" in r2_prefix:
                                    migration_results["masks"]["skipped"] += 1
                                else:
                                    migration_results["maps"]["skipped"] += 1
                                logger.info(f"⏭️ Skipping {r2_key} (already exists)")
                                continue
                            except r2_client.exceptions.NoSuchKey:
                                # File doesn't exist, proceed with upload
                                pass
                            
                            # Upload file to R2
                            with open(local_file_path, 'rb') as f:
                                # Determine content type
                                content_type = "application/octet-stream"
                                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                                    content_type = f"image/{file.split('.')[-1].lower()}"
                                elif file.lower().endswith('.json'):
                                    content_type = "application/json"
                                
                                r2_client.upload_fileobj(
                                    f,
                                    r2_bucket_name,
                                    r2_key,
                                    ExtraArgs={
                                        'ContentType': content_type,
                                        'ACL': 'public-read'
                                    }
                                )
                            
                            if "masks" in r2_prefix:
                                migration_results["masks"]["uploaded"] += 1
                            else:
                                migration_results["maps"]["uploaded"] += 1
                            
                            migration_results["total_processed"] += 1
                            logger.info(f"✅ Uploaded {local_file_path} → {r2_key}")
                            
                        except Exception as e:
                            error_msg = f"Failed to upload {local_file_path}: {str(e)}"
                            if "masks" in r2_prefix:
                                migration_results["masks"]["errors"].append(error_msg)
                            else:
                                migration_results["maps"]["errors"].append(error_msg)
                            logger.error(f"❌ {error_msg}")
            
            migration_results["status"] = "completed"
            migration_results["summary"] = f"Processed {migration_results['total_processed']} files"
            
            return migration_results
            
        except Exception as e:
            return {
                "error": f"Migration failed: {str(e)}",
                "status": "failed",
                "results": migration_results
            }

    @app.get("/debug/color-deps", response_model=None)
    async def debug_color_dependencies():
        """Debug endpoint to check color extraction dependencies"""
        try:
            import cv2
            cv2_version = cv2.__version__
        except ImportError as e:
            cv2_version = f"Error: {e}"
        
        try:
            import sklearn
            sklearn_version = sklearn.__version__
        except ImportError as e:
            sklearn_version = f"Error: {e}"
        
        try:
            import webcolors
            webcolors_version = webcolors.__version__
        except ImportError as e:
            webcolors_version = f"Error: {e}"
        
        try:
            # Check if color extractor service is available
            from core.dependencies import get_detection_service
            detection_service = get_detection_service()
            
            if detection_service and hasattr(detection_service, 'color_extractor'):
                color_status = "✅ Available via detection service"
            else:
                # Try direct import
                from models.color_extractor import ColorExtractor
                color_extractor_test = ColorExtractor()
                color_status = "✅ Available via direct import"
        except Exception as e:
            color_status = f"❌ Error: {e}"
        
        return {
            "dependencies": {
                "cv2": cv2_version,
                "sklearn": sklearn_version, 
                "webcolors": webcolors_version
            },
            "color_extractor": color_status,
            "color_extractor_loaded": "detection_service" in locals() and detection_service is not None
        }

    @app.get("/admin/system-info", response_model=None)
    async def get_system_info():
        """Get comprehensive system information for debugging"""
        try:
            import torch
            torch_info = {
                "version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None
            }
        except ImportError:
            torch_info = {"error": "PyTorch not available"}
        
        # Check service availability
        from core.dependencies import check_services_ready
        services_status = check_services_ready()
        
        # Environment info
        env_info = {
            "r2_configured": r2_client is not None,
            "r2_bucket": r2_bucket_name,
            "cache_dirs": {
                "masks": os.path.exists("/app/cache_volume/masks"),
                "maps": os.path.exists("/app/cache_volume/maps"),
                "depth_maps": os.path.exists("/app/cache_volume/depth_maps"),
                "edge_maps": os.path.exists("/app/cache_volume/edge_maps")
            }
        }
        
        return {
            "torch": torch_info,
            "services": services_status,
            "environment": env_info,
            "ai_features": ["GroundingDINO", "SAM2", "CLIP", "Vector Search", "Color Extraction"],
            "mode": "refactored_architecture"
        }

    @app.get("/admin/health-detailed", response_model=None)
    async def detailed_health_check():
        """Comprehensive health check with detailed AI model status"""
        from core.dependencies import get_detection_service, get_database_service, get_job_service
        from datetime import datetime
        
        detection_service = get_detection_service()
        database_service = get_database_service()
        job_service = get_job_service()
        
        # AI model status
        ai_status = {
            "detector_loaded": detection_service is not None,
            "detector_details": (
                detection_service.get_detector_status() 
                if detection_service and hasattr(detection_service, 'get_detector_status') 
                else {}
            ),
            "segmenter_loaded": (
                detection_service is not None and 
                hasattr(detection_service, 'segmenter')
            ),
            "embedder_loaded": (
                detection_service is not None and 
                hasattr(detection_service, 'embedder')
            ),
            "color_extractor_loaded": (
                detection_service is not None and 
                hasattr(detection_service, 'color_extractor')
            )
        }
        
        # Add device info
        try:
            import torch
            ai_status.update({
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "pytorch_version": torch.__version__
            })
        except ImportError:
            ai_status.update({
                "device": "unknown",
                "pytorch_version": "Not available"
            })
        
        # Database status
        db_status = {
            "database_connected": database_service is not None,
            "supabase_available": (
                database_service is not None and 
                hasattr(database_service, 'supabase') and 
                database_service.supabase is not None
            )
        }
        
        # Job service status
        job_status = {
            "job_service_available": job_service is not None,
            "redis_connected": (
                job_service is not None and 
                hasattr(job_service, 'is_available') and 
                job_service.is_available()
            )
        }
        
        # R2 storage status
        storage_status = {
            "r2_client_available": r2_client is not None,
            "r2_bucket_configured": r2_bucket_name is not None
        }
        
        overall_status = "healthy" if all([
            ai_status["detector_loaded"],
            db_status["database_connected"],
            storage_status["r2_client_available"]
        ]) else "degraded"
        
        return {
            "status": overall_status,
            "timestamp": datetime.utcnow(),
            "mode": "refactored_full_ai",
            "ai_models": ai_status,
            "database": db_status,
            "jobs": job_status,
            "storage": storage_status,
            "features": [
                "Modular Architecture",
                "Advanced AI Pipeline", 
                "Comprehensive Taxonomy",
                "R2 Storage Integration",
                "Background Job Processing"
            ]
        }