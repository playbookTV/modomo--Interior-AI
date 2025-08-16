"""
Modomo Dataset Scraping System - Refactored Main Application
Complete system with modular architecture and clean separation of concerns
"""
import os
from datetime import datetime
from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Any

# Import with graceful fallback for local development
try:
    import redis
except ImportError:
    redis = None
    
try:
    from supabase import create_client, Client
except ImportError:
    create_client = None
    Client = None

# Import our new modular components
from config.settings import settings
from config.taxonomy import MODOMO_TAXONOMY
from utils.logging import configure_logging, get_logger
from services.database_service import DatabaseService
from services.job_service import JobService
from services.detection_service import DetectionService

# Configure logging first
configure_logging()
logger = get_logger(__name__)

# Import simplified routers (no dependency injection)
try:
    from routers_simple.admin import router as admin_router
    from routers_simple.analytics import router as analytics_router
    logger.info("✅ Using simplified routers")
except ImportError:
    # Fallback to original routers if simple ones don't exist
    logger.warning("⚠️ Simplified routers not found, creating minimal routes")
    from fastapi import APIRouter
    
    # Create minimal routers
    admin_router = APIRouter(prefix="/admin", tags=["admin"])
    analytics_router = APIRouter(tags=["analytics"])
    
    @admin_router.get("/test-supabase")
    async def test_supabase():
        return {"status": "minimal", "message": "Using minimal admin router"}
    
    @analytics_router.get("/taxonomy")
    async def get_taxonomy():
        from config.taxonomy import MODOMO_TAXONOMY
        return MODOMO_TAXONOMY

# Create minimal routers for other endpoints
from fastapi import APIRouter
detection_router = APIRouter(prefix="/detect", tags=["detection"])
jobs_router = APIRouter(prefix="/jobs", tags=["jobs"])  
scraping_router = APIRouter(prefix="/scrape", tags=["scraping"])

# Add basic endpoints to prevent empty router errors
@detection_router.get("/status")
async def detection_status():
    return {"status": "available", "message": "Detection service endpoint"}

@jobs_router.get("/status")
async def jobs_status():
    return {"status": "available", "message": "Jobs service endpoint"}

@jobs_router.get("/active")
async def get_active_jobs():
    """Get currently active/running jobs"""
    try:
        if _job_service and _job_service.is_available():
            jobs = _job_service.get_active_jobs()  # Remove await - this is synchronous
            return jobs
        else:
            # Return empty list if Redis/job service unavailable
            return []
    except Exception as e:
        logger.warning(f"Failed to get active jobs: {e}")
        return []

@jobs_router.get("/history")
async def get_job_history(
    limit: int = Query(20, description="Number of jobs to return"),
    offset: int = Query(0, description="Number of jobs to skip"),
    status: str = Query("all", description="Filter by job status")
):
    """Get job history with pagination"""
    try:
        if _job_service and _job_service.is_available():
            # Since get_job_history doesn't exist, use get_active_jobs and simulate history
            jobs = _job_service.get_active_jobs()  # Remove await and use available method
            return {
                "jobs": jobs,
                "total": len(jobs),
                "limit": limit,
                "offset": offset,
                "has_more": len(jobs) == limit
            }
        else:
            return {
                "jobs": [],
                "total": 0,
                "limit": limit,
                "offset": offset,
                "has_more": False
            }
    except Exception as e:
        logger.warning(f"Failed to get job history: {e}")
        return {
            "jobs": [],
            "total": 0,
            "limit": limit,
            "offset": offset,
            "has_more": False
        }

@jobs_router.get("/errors/recent")
async def get_recent_errors(limit: int = Query(10, description="Number of error jobs to return")):
    """Get recent failed/error jobs"""
    try:
        if _job_service and _job_service.is_available():
            error_jobs = _job_service.get_recent_errors(limit=limit)  # Remove await - this is synchronous
            return {
                "errors": error_jobs,
                "total_error_jobs": len(error_jobs)
            }
        else:
            return {
                "errors": [],
                "total_error_jobs": 0
            }
    except Exception as e:
        logger.warning(f"Failed to get recent errors: {e}")
        return {
            "errors": [],
            "total_error_jobs": 0
        }

@scraping_router.get("/status")
async def scraping_status():
    return {"status": "available", "message": "Scraping service endpoint"}

# Logger already configured above

# Import AI models with graceful handling
AI_MODELS_AVAILABLE = False
try:
    import torch
    from models.grounding_dino import GroundingDINODetector
    from models.sam2_segmenter import SAM2Segmenter, SegmentationConfig
    from models.clip_embedder import CLIPEmbedder
    from models.color_extractor import ColorExtractor
    AI_MODELS_AVAILABLE = True
    logger.info("✅ AI model classes imported successfully")
except ImportError as e:
    logger.warning(f"⚠️ AI models import failed: {e}")
    logger.info("🔄 Will run in basic mode without AI features")
    # Create dummy classes for Railway compatibility
    GroundingDINODetector = None
    SAM2Segmenter = None
    SegmentationConfig = None
    CLIPEmbedder = None
    ColorExtractor = None

# Check crawler availability
try:
    from crawlers.houzz_crawler import HouzzCrawler
    CRAWLER_AVAILABLE = True
except ImportError:
    CRAWLER_AVAILABLE = False
    logger.warning("⚠️ Houzz crawler not available - detection only mode")

# Initialize FastAPI app
app = FastAPI(
    title=settings.APP_TITLE,
    description=settings.APP_DESCRIPTION,
    version=settings.APP_VERSION
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=settings.CORS_ALLOW_METHODS,
    allow_headers=settings.CORS_ALLOW_HEADERS,
)

# Create masks directory (with fallback for read-only filesystems)
try:
    os.makedirs(settings.MASKS_DIR, exist_ok=True)
    logger.info(f"✅ Created masks directory: {settings.MASKS_DIR}")
except (OSError, PermissionError) as e:
    logger.warning(f"⚠️ Cannot create masks directory {settings.MASKS_DIR}: {e}")
    logger.info("🔄 Will use temporary directory for masks")

# Global service instances
supabase_client = None
redis_client = None
database_service = None
job_service = None
detection_service = None

# AI model instances
detector = None
segmenter = None
embedder = None
color_extractor = None

# Pydantic models
class SceneMetadata(BaseModel):
    houzz_id: str
    image_url: str
    room_type: Optional[str] = None
    style_tags: List[str] = Field(default_factory=list)
    color_tags: List[str] = Field(default_factory=list)
    project_url: Optional[str] = None

class DetectedObject(BaseModel):
    bbox: List[float] = Field(..., description="[x, y, width, height]")
    mask_url: Optional[str] = None
    category: str
    confidence: float
    tags: List[str] = Field(default_factory=list)
    matched_product_id: Optional[str] = None


# Global service instances for router access
# These will be set during startup
_database_service = None
_job_service = None
_detection_service = None

def get_database_service():
    """Get database service instance"""
    return _database_service

def get_job_service():
    """Get job service instance"""
    return _job_service

def get_detection_service():
    """Get detection service instance"""
    return _detection_service


@app.on_event("startup")
async def startup():
    """Initialize all services and AI models"""
    global supabase_client, redis_client, database_service, job_service, detection_service
    global detector, segmenter, embedder, color_extractor
    global _database_service, _job_service, _detection_service
    
    try:
        logger.info("Starting Modomo Scraper (Refactored Architecture)")
        
        # Validate configuration
        config_status = settings.validate_required_settings()
        logger.info(f"🔍 Configuration validation: {config_status}")
        
        # Initialize Supabase client
        if config_status["supabase_configured"] and create_client:
            try:
                supabase_client = create_client(settings.SUPABASE_URL, settings.SUPABASE_ANON_KEY)
                logger.info("✅ Supabase client initialized successfully")
                
                # Test the connection
                test_result = supabase_client.table("scenes").select("scene_id").limit(1).execute()
                logger.info("✅ Supabase connection test passed")
                
                # Initialize database service
                database_service = DatabaseService(supabase_client)
                
            except Exception as e:
                logger.error(f"❌ Supabase client initialization failed: {e}")
                supabase_client = None
        else:
            logger.error(f"❌ Missing Supabase credentials: {config_status['missing']}")
        
        # Initialize Redis client
        try:
            if redis:
                redis_client = redis.from_url(settings.REDIS_URL, socket_timeout=10)
            else:
                raise ImportError("Redis not available")
            redis_client.ping()  # Test connection
            logger.info("✅ Connected to Redis")
            
            # Initialize job service
            job_service = JobService(redis_client)
            
        except Exception as redis_error:
            logger.warning(f"Redis connection failed: {redis_error}")
            logger.info("Will continue without Redis - job tracking disabled")
            redis_client = None
            job_service = JobService(None)  # Create service without Redis
        
        # Initialize AI models
        logger.info("🤖 Loading AI models...")
        try:
            # Initialize color extractor first (minimal dependencies)
            logger.info("🎨 Loading color extractor...")
            color_extractor = ColorExtractor()
            logger.info("✅ Color extractor loaded successfully")
            
            # Initialize detector
            detector = GroundingDINODetector()
            logger.info("✅ GroundingDINO detector initialized")
            
            # Initialize segmenter
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"🖥️ Using device: {device}")
            config = SegmentationConfig(device=device)
            segmenter = SAM2Segmenter(config=config)
            
            # Get detailed model info
            model_info = segmenter.get_model_info()
            if model_info.get("sam2_available"):
                logger.info(f"🔥 SAM2Segmenter initialized with REAL SAM2 on {device}")
                if model_info.get("checkpoint"):
                    logger.info(f"📦 Using checkpoint: {model_info['checkpoint']}")
            else:
                logger.warning(f"⚠️ SAM2Segmenter using fallback mode on {device}")
            
            # Initialize embedder
            embedder = CLIPEmbedder()
            logger.info("✅ CLIP embedder initialized")
            
            # Initialize map generation models
            logger.info("🗺️ Loading map generation models...")
            depth_estimator = None
            edge_detector = None
            
            try:
                from models.depth_estimator import DepthEstimator, DepthConfig
                depth_config = DepthConfig(
                    device=device,
                    cpu_optimization=device == "cpu",
                    reduce_precision=device == "cuda"
                )
                depth_estimator = DepthEstimator(depth_config)
                logger.info(f"✅ Depth Anything V2 estimator initialized on {device}")
                if device == "cpu":
                    logger.info("🔧 CPU optimizations enabled for depth estimation")
            except Exception as depth_error:
                logger.warning(f"⚠️ Depth estimator failed to load: {depth_error}")
                depth_estimator = None
            
            try:
                from models.edge_detector import EdgeDetector, EdgeConfig
                edge_config = EdgeConfig()
                edge_detector = EdgeDetector(edge_config)
                logger.info("✅ CV2 Canny edge detector initialized")
            except Exception as edge_error:
                logger.warning(f"⚠️ Edge detector failed to load: {edge_error}")
            
            # Initialize detection service with map generation
            detection_service = DetectionService(
                detector=detector,
                segmenter=segmenter,
                embedder=embedder,
                color_extractor=color_extractor,
                depth_estimator=depth_estimator,
                edge_detector=edge_detector
            )
            
            logger.info("✅ All AI models and services loaded successfully")
            
        except Exception as ai_error:
            logger.error(f"❌ AI model loading failed: {ai_error}")
            # Create minimal services for basic functionality
            detection_service = None
        
        # Set global service instances for router access
        _database_service = database_service
        _job_service = job_service 
        _detection_service = detection_service
        
        logger.info("✅ Service instances configured for router access")
        
    except Exception as e:
        logger.error(f"Startup error: {e}")
        raise


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown"""
    if redis_client:
        redis_client.close()
    logger.info("Modomo Scraper shutdown complete")


# Custom static file handler for masks with CORS headers
@app.get("/masks/{filename}")
async def serve_mask(filename: str):
    """Serve mask files with CORS headers"""
    file_path = os.path.join(settings.MASKS_DIR, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Mask file not found")
    
    # Return file with CORS headers
    response = FileResponse(
        file_path,
        media_type="image/png",
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "*",
            "Cache-Control": "public, max-age=3600"  # Cache for 1 hour
        }
    )
    return response


@app.options("/masks/{filename}")
async def mask_options(filename: str):
    """Handle CORS preflight requests for mask files"""
    from fastapi import Response
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        }
    )


# Map serving endpoints
@app.get("/maps/{map_type}/{filename}")
async def serve_map(map_type: str, filename: str):
    """Serve depth/edge maps with CORS headers"""
    # Validate map type
    if map_type not in ["depth", "edge"]:
        raise HTTPException(status_code=400, detail="Invalid map type. Use 'depth' or 'edge'")
    
    # For now, serve from local cache directory (later: proxy to R2)
    maps_dir = os.path.join(settings.MASKS_DIR, "../maps", map_type)
    file_path = os.path.join(maps_dir, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"{map_type.title()} map not found")
    
    # Determine media type
    media_type = "image/png" if filename.endswith('.png') else "image/jpeg"
    
    # Return file with CORS headers
    response = FileResponse(
        file_path,
        media_type=media_type,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "*",
            "Cache-Control": "public, max-age=3600"  # Cache for 1 hour
        }
    )
    return response


@app.options("/maps/{map_type}/{filename}")
async def map_options(map_type: str, filename: str):
    """Handle CORS preflight requests for map files"""
    from fastapi import Response
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        }
    )


# Main application routes
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Modomo Dataset Creation System (Refactored Architecture)",
        "docs": "/docs", 
        "health": "/health",
        "ai_features": ["GroundingDINO", "SAM2", "CLIP", "Vector Search"],
        "scraping": "/scrape/scenes",
        "note": "Complete AI pipeline with modular architecture"
    }


@app.get("/health")
async def health_check():
    """Health check with AI model status"""
    ai_status = {}
    
    if detection_service:
        ai_status = detection_service.get_status()
        ai_status.update({
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "pytorch_version": torch.__version__
        })
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "mode": "refactored_architecture",
        "ai_models": ai_status,
        "services": {
            "database": database_service is not None,
            "job_tracking": job_service is not None and job_service.is_available(),
            "detection": detection_service is not None and detection_service.is_available(),
            "crawler": CRAWLER_AVAILABLE
        },
        "note": "Refactored modular architecture with clean separation of concerns"
    }


@app.get("/scenes")
async def get_scenes(
    limit: int = 20,
    offset: int = 0,
    status: str = None,
    db_service: DatabaseService = Depends(get_database_service)
):
    """Get list of stored scenes with their images"""
    return await db_service.get_scenes(limit, offset, status)


@app.get("/objects")
async def get_detected_objects(
    limit: int = 20,
    offset: int = 0,
    category: str = None,
    scene_id: str = None,
    db_service: DatabaseService = Depends(get_database_service)
):
    """Get list of detected objects with their details"""
    return await db_service.get_detected_objects(limit, offset, category, scene_id)


# Debug endpoints
@app.get("/debug/color-deps")
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
    
    color_status = "✅ Available" if color_extractor else "❌ Not loaded"
    
    return {
        "dependencies": {
            "cv2": cv2_version,
            "sklearn": sklearn_version, 
            "webcolors": webcolors_version
        },
        "color_extractor": color_status,
        "color_extractor_loaded": color_extractor is not None
    }


@app.get("/debug/detector-status")
async def debug_detector_status():
    """Debug endpoint to check detector status"""
    if not detector:
        return {"error": "No detector available"}
    
    status = {
        "detector_available": detector is not None,
        "detector_type": "Multi-model (YOLO + DETR)"
    }
    
    if hasattr(detector, 'get_detector_status'):
        status.update(detector.get_detector_status())
    
    # Check YOLO dependencies
    try:
        from ultralytics import YOLO
        yolo_status = "✅ Available (required for multi-model detection)"
    except ImportError:
        yolo_status = "❌ Not installed (REQUIRED for multi-model detection)"
    except Exception as e:
        yolo_status = f"❌ Error: {e} (REQUIRED for multi-model detection)"
    
    status["yolo_package"] = yolo_status
    
    return status


# Performance monitoring endpoint
@app.get("/performance/status")
async def get_performance_status():
    """Get system performance status and recommendations"""
    try:
        from utils.performance_monitor import get_performance_monitor
        monitor = get_performance_monitor()
        
        recommendations = monitor.get_recommendations()
        latest_metrics = monitor.get_latest_metrics()
        
        return {
            "system_specs": {
                "cpu_count": monitor.system_specs.cpu_count,
                "memory_total_gb": round(monitor.system_specs.memory_total, 1),
                "memory_available_gb": round(monitor.system_specs.memory_available, 1),
                "gpu_available": monitor.system_specs.gpu_available,
                "gpu_name": monitor.system_specs.gpu_name,
                "gpu_memory_gb": round(monitor.system_specs.gpu_memory, 1) if monitor.system_specs.gpu_memory else None
            },
            "performance_status": recommendations["system_status"],
            "warnings": recommendations.get("warnings"),
            "suggestions": recommendations.get("suggestions", []),
            "estimated_times": recommendations.get("estimated_times", {}),
            "latest_operation": {
                "name": latest_metrics.operation_name,
                "duration": round(latest_metrics.duration, 1),
                "cpu_usage": round(latest_metrics.cpu_usage_avg, 1)
            } if latest_metrics else None
        }
    except Exception as e:
        logger.warning(f"Performance monitoring not available: {e}")
        return {
            "system_specs": {"gpu_available": torch.cuda.is_available()},
            "performance_status": "unknown",
            "note": "Performance monitoring not available"
        }


# Map generation endpoints
@app.post("/generate-maps/{scene_id}")
async def generate_maps_for_scene(
    scene_id: str,
    map_types: List[str] = Query(["depth", "edge"], description="Types of maps to generate"),
    db_service: DatabaseService = Depends(get_database_service),
    detection_service: DetectionService = Depends(get_detection_service)
):
    """Generate depth and edge maps for a specific scene"""
    if not detection_service or not detection_service.map_generator:
        raise HTTPException(status_code=503, detail="Map generation service not available")
    
    try:
        # Validate map types
        valid_types = ["depth", "edge"]
        invalid_types = [t for t in map_types if t not in valid_types]
        if invalid_types:
            raise HTTPException(status_code=400, detail=f"Invalid map types: {invalid_types}")
        
        # Get scene from database
        scenes = await db_service.get_scenes(limit=1, offset=0, status=None, scene_id=scene_id)
        if not scenes.get("scenes"):
            raise HTTPException(status_code=404, detail="Scene not found")
        
        scene = scenes["scenes"][0]
        image_url = scene.get("image_url")
        if not image_url:
            raise HTTPException(status_code=400, detail="Scene has no image URL")
        
        logger.info(f"🗺️ Generating maps for scene {scene_id}: {map_types}")
        
        # Generate maps
        results = await detection_service.generate_scene_maps(image_url, scene_id, map_types)
        
        if results["success"]:
            # Update database with R2 keys
            update_data = {}
            if "depth" in results["r2_keys"]:
                update_data["depth_map_r2_key"] = results["r2_keys"]["depth"]
            if "edge" in results["r2_keys"]:
                update_data["edge_map_r2_key"] = results["r2_keys"]["edge"]
            
            if update_data:
                update_data["maps_generated_at"] = datetime.utcnow().isoformat()
                update_data["maps_metadata"] = {
                    "generated_maps": list(results["maps_generated"].keys()),
                    "generation_time": results["generation_time"],
                    "r2_keys": results["r2_keys"]
                }
                
                # Update scene in database
                await db_service.update_scene(scene_id, update_data)
            
            return {
                "scene_id": scene_id,
                "success": True,
                "maps_generated": results["maps_generated"],
                "r2_keys": results["r2_keys"],
                "message": f"Successfully generated {len(results['maps_generated'])} maps"
            }
        else:
            return {
                "scene_id": scene_id,
                "success": False,
                "errors": results["errors"],
                "message": "Map generation failed"
            }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Map generation endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/generate-maps/batch")
async def generate_maps_batch(
    limit: int = Query(10, description="Number of scenes to process"),
    map_types: List[str] = Query(["depth", "edge"], description="Types of maps to generate"),
    force_regenerate: bool = Query(False, description="Regenerate maps even if they exist"),
    db_service: DatabaseService = Depends(get_database_service),
    detection_service: DetectionService = Depends(get_detection_service)
):
    """Generate maps for multiple scenes in batch"""
    if not detection_service or not detection_service.map_generator:
        raise HTTPException(status_code=503, detail="Map generation service not available")
    
    try:
        # Validate map types
        valid_types = ["depth", "edge"]
        invalid_types = [t for t in map_types if t not in valid_types]
        if invalid_types:
            raise HTTPException(status_code=400, detail=f"Invalid map types: {invalid_types}")
        
        logger.info(f"🗺️ Starting batch map generation: {limit} scenes, {map_types}")
        
        # Get scenes that need maps
        scenes_filter = {}
        if not force_regenerate:
            # Only get scenes without maps
            if "depth" in map_types:
                scenes_filter["depth_map_r2_key"] = None
            if "edge" in map_types:
                scenes_filter["edge_map_r2_key"] = None
        
        scenes = await db_service.get_scenes(limit=limit, offset=0, filters=scenes_filter)
        
        if not scenes.get("scenes"):
            return {
                "message": "No scenes found that need map generation",
                "processed": 0,
                "success": True
            }
        
        results = {
            "total_scenes": len(scenes["scenes"]),
            "processed": 0,
            "successful": 0,
            "failed": 0,
            "errors": []
        }
        
        # Process each scene
        for scene in scenes["scenes"]:
            try:
                scene_id = scene["scene_id"]
                image_url = scene.get("image_url")
                
                if not image_url:
                    results["failed"] += 1
                    results["errors"].append(f"Scene {scene_id}: No image URL")
                    continue
                
                # Generate maps for this scene
                map_results = await detection_service.generate_scene_maps(image_url, scene_id, map_types)
                
                if map_results["success"]:
                    # Update database
                    update_data = {}
                    if "depth" in map_results["r2_keys"]:
                        update_data["depth_map_r2_key"] = map_results["r2_keys"]["depth"]
                    if "edge" in map_results["r2_keys"]:
                        update_data["edge_map_r2_key"] = map_results["r2_keys"]["edge"]
                    
                    if update_data:
                        update_data["maps_generated_at"] = datetime.utcnow().isoformat()
                        await db_service.update_scene(scene_id, update_data)
                    
                    results["successful"] += 1
                    logger.info(f"✅ Generated maps for scene {scene_id}")
                else:
                    results["failed"] += 1
                    results["errors"].append(f"Scene {scene_id}: {map_results.get('errors', {})}")
                    logger.error(f"❌ Failed to generate maps for scene {scene_id}")
                
                results["processed"] += 1
                
            except Exception as scene_error:
                results["failed"] += 1
                results["errors"].append(f"Scene {scene.get('scene_id', 'unknown')}: {str(scene_error)}")
                logger.error(f"❌ Error processing scene: {scene_error}")
        
        return {
            "message": f"Batch map generation completed: {results['successful']}/{results['total_scenes']} successful",
            **results,
            "success": results["successful"] > 0
        }
        
    except Exception as e:
        logger.error(f"❌ Batch map generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")


@app.get("/scenes/{scene_id}/maps")
async def get_scene_maps(
    scene_id: str,
    db_service: DatabaseService = Depends(get_database_service)
):
    """Get available maps for a specific scene"""
    try:
        # Get scene from database
        scenes = await db_service.get_scenes(limit=1, offset=0, status=None, scene_id=scene_id)
        if not scenes.get("scenes"):
            raise HTTPException(status_code=404, detail="Scene not found")
        
        scene = scenes["scenes"][0]
        
        # Extract map information
        maps_info = {
            "scene_id": scene_id,
            "maps_available": {},
            "maps_generated_at": scene.get("maps_generated_at"),
            "maps_metadata": scene.get("maps_metadata", {})
        }
        
        # Check available maps
        if scene.get("depth_map_r2_key"):
            maps_info["maps_available"]["depth"] = {
                "r2_key": scene["depth_map_r2_key"],
                "url": f"/maps/depth/{scene_id}_depth.png"
            }
        
        if scene.get("edge_map_r2_key"):
            maps_info["maps_available"]["edge"] = {
                "r2_key": scene["edge_map_r2_key"],
                "url": f"/maps/edge/{scene_id}_edge.png"
            }
        
        return maps_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Get scene maps error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# Include all routers
app.include_router(admin_router)
app.include_router(analytics_router)
app.include_router(detection_router)
app.include_router(jobs_router)
app.include_router(scraping_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)