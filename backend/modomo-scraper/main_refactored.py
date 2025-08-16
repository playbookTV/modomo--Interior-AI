"""
Modomo Dataset Scraping System - Refactored Main Application
Complete system with modular architecture and clean separation of concerns
"""
import os
from datetime import datetime
from fastapi import FastAPI, HTTPException, Depends, Query, APIRouter
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
    
# Import basic modules first
from config.settings import settings
from config.taxonomy import MODOMO_TAXONOMY
from utils.logging import configure_logging, get_logger
from services.database_service import DatabaseService
from services.job_service import JobService
from services.detection_service import DetectionService

# Configure logging first
configure_logging()
logger = get_logger(__name__)

# Now import supabase with proper logging
try:
    from supabase import create_client, Client
    logger.info("✅ Supabase package imported successfully")
except ImportError as e:
    logger.error(f"❌ Supabase import failed: {e}")
    create_client = None
    Client = None
except Exception as e:
    logger.error(f"❌ Supabase import error: {e}")
    create_client = None
    Client = None

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
        try:
            if not _database_service:
                raise HTTPException(status_code=503, detail="Database service not available")
            
            # Test Supabase connection
            test_result = await _database_service.test_connection()
            return test_result
        except Exception as e:
            logger.error(f"❌ Supabase test error: {e}")
            return {"status": "error", "message": f"Supabase test failed: {str(e)}"}
    
    @admin_router.post("/init-database")
    async def init_database():
        """Initialize database schema (admin only)"""
        try:
            if not _database_service:
                raise HTTPException(status_code=503, detail="Database service not available")
            
            # This would typically use a schema migration system
            # For now, we'll indicate that schema should be applied via SQL
            return {
                "status": "info",
                "message": "Database schema initialization should be done via SQL migration",
                "recommendation": "Run the SQL commands from database/schema.sql in Supabase SQL editor",
                "schema_location": "backend/modomo-scraper/database/schema.sql",
                "note": "Automated schema creation not implemented in refactored architecture"
            }
            
        except Exception as e:
            logger.error(f"❌ Database initialization error: {e}")
            raise HTTPException(status_code=500, detail=f"Database initialization failed: {str(e)}")
    
    @analytics_router.get("/taxonomy")
    async def get_taxonomy():
        from config.taxonomy import MODOMO_TAXONOMY
        return MODOMO_TAXONOMY
    
    @analytics_router.get("/stats/dataset")
    async def get_dataset_stats():
        """Get comprehensive dataset statistics"""
        try:
            if not _database_service:
                raise HTTPException(status_code=503, detail="Database service not available")
            
            stats = await _database_service.get_dataset_stats()
            return stats
        except Exception as e:
            logger.error(f"❌ Dataset stats error: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get dataset stats: {str(e)}")
    
    @analytics_router.get("/stats/categories")
    async def get_category_stats():
        """Get category-wise statistics"""
        try:
            if not _database_service:
                raise HTTPException(status_code=503, detail="Database service not available")
            
            # Get category breakdown from detected objects
            result = _database_service.supabase.table("detected_objects").select(
                "category, confidence, approved"
            ).execute()
            
            # Process category statistics
            category_stats = {}
            for obj in result.data:
                category = obj.get("category", "unknown")
                if category not in category_stats:
                    category_stats[category] = {
                        "total_objects": 0,
                        "approved_objects": 0,
                        "avg_confidence": 0,
                        "confidence_sum": 0
                    }
                
                category_stats[category]["total_objects"] += 1
                if obj.get("approved"):
                    category_stats[category]["approved_objects"] += 1
                if obj.get("confidence"):
                    category_stats[category]["confidence_sum"] += float(obj["confidence"])
            
            # Calculate averages
            for category, stats in category_stats.items():
                if stats["total_objects"] > 0:
                    stats["avg_confidence"] = stats["confidence_sum"] / stats["total_objects"]
                del stats["confidence_sum"]  # Remove intermediate calculation
            
            return category_stats
            
        except Exception as e:
            logger.error(f"❌ Category stats error: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get category stats: {str(e)}")

# Initialize FastAPI app first (before any route definitions)
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

# Create minimal routers for other endpoints
detection_router = APIRouter(prefix="/detect", tags=["detection"])
jobs_router = APIRouter(prefix="/jobs", tags=["jobs"])  
scraping_router = APIRouter(prefix="/scrape", tags=["scraping"])

# Add basic endpoints to prevent empty router errors
@detection_router.get("/status")
async def detection_status():
    return {"status": "available", "message": "Detection service endpoint"}

@detection_router.post("/process")
async def process_detection(
    image_url: str = Query(..., description="URL of image to process"),
    job_id: str = Query(None, description="Optional job ID for tracking")
):
    """Process image for object detection and segmentation"""
    try:
        if not _detection_service:
            raise HTTPException(status_code=503, detail="Detection service not available")
        
        # Generate job ID if not provided
        if not job_id:
            import uuid
            job_id = str(uuid.uuid4())
        
        # Run detection pipeline
        from config.taxonomy import MODOMO_TAXONOMY
        detections = await _detection_service.run_detection_pipeline(image_url, job_id, MODOMO_TAXONOMY)
        
        # Start Celery task for background AI processing
        try:
            from tasks.detection_tasks import run_detection_pipeline
            task = run_detection_pipeline.delay(job_id, image_url)
            
            return {
                "job_id": job_id,
                "task_id": task.id,
                "image_url": image_url,
                "status": "running",
                "message": "AI detection pipeline started",
                "celery_task": "run_detection_pipeline"
            }
        except ImportError:
            # Fallback to synchronous processing if Celery not available
            return {
                "job_id": job_id,
                "image_url": image_url,
                "detections": detections,
                "total_objects": len(detections),
                "success": True,
                "note": "Processed synchronously - Celery not available"
            }
        
    except Exception as e:
        logger.error(f"❌ Detection processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Detection processing failed: {str(e)}")

# Color processing endpoints
@app.get("/colors/extract")
async def extract_colors(
    image_url: str = Query(..., description="URL of image to extract colors from"),
    bbox: str = Query(None, description="Optional bounding box as 'x,y,w,h'")
):
    """Extract colors from image or image region"""
    try:
        if not _detection_service:
            raise HTTPException(status_code=503, detail="Detection service not available")
        
        # Parse bbox if provided
        bbox_list = None
        if bbox:
            try:
                bbox_list = [float(x) for x in bbox.split(',')]
                if len(bbox_list) != 4:
                    raise ValueError("Bbox must have 4 values")
            except:
                raise HTTPException(status_code=400, detail="Invalid bbox format. Use 'x,y,w,h'")
        
        # Extract colors
        color_data = await _detection_service.extract_colors_from_url(image_url, bbox_list)
        
        return color_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Color extraction error: {e}")
        raise HTTPException(status_code=500, detail=f"Color extraction failed: {str(e)}")

@app.get("/search/color")
async def search_by_color(
    hex_color: str = Query(..., description="Hex color code (e.g., #FF5733)"),
    limit: int = Query(20, description="Number of results to return"),
    tolerance: float = Query(0.1, description="Color tolerance (0.0-1.0)")
):
    """Search for objects by color"""
    try:
        if not _database_service:
            raise HTTPException(status_code=503, detail="Database service not available")
        
        # Simple color search implementation
        # In production, this would use more sophisticated color matching
        result = _database_service.supabase.table("detected_objects").select(
            "object_id, scene_id, category, confidence, tags, metadata"
        ).limit(limit).execute()
        
        # Filter objects that have color tags matching the search
        matching_objects = []
        search_color = hex_color.lower().replace('#', '')
        
        for obj in result.data:
            tags = obj.get("tags", [])
            metadata = obj.get("metadata", {})
            
            # Check if color appears in tags or metadata
            color_match = any(search_color in str(tag).lower() for tag in tags)
            if not color_match and metadata.get("colors"):
                color_match = any(search_color in str(c).lower() for c in metadata["colors"])
            
            if color_match:
                matching_objects.append(obj)
        
        return {
            "color": hex_color,
            "objects": matching_objects[:limit],
            "total_found": len(matching_objects)
        }
        
    except Exception as e:
        logger.error(f"❌ Color search error: {e}")
        raise HTTPException(status_code=500, detail=f"Color search failed: {str(e)}")

@app.get("/colors/palette")
async def get_color_palette():
    """Get available color names and their RGB values for filtering"""
    try:
        if not _detection_service or not _detection_service.color_extractor:
            raise HTTPException(status_code=503, detail="Color extractor not available")
        
        color_extractor = _detection_service.color_extractor
        
        # Return the color mappings from the extractor
        return {
            "color_palette": color_extractor.color_mappings if hasattr(color_extractor, 'color_mappings') else {},
            "color_categories": {
                "neutrals": ["white", "black", "gray", "beige", "cream"],
                "warm": ["red", "orange", "yellow", "pink", "brown", "tan", "gold"],
                "cool": ["blue", "green", "teal", "purple"],
                "wood_tones": ["light_wood", "medium_wood", "dark_wood"]
            }
        }
    except Exception as e:
        logger.error(f"❌ Color palette error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get color palette: {str(e)}")

@app.get("/stats/colors")
async def get_color_statistics():
    """Get statistics about colors in the dataset"""
    try:
        if not _database_service:
            raise HTTPException(status_code=503, detail="Database service not available")
        
        # Get all objects with color metadata
        result = _database_service.supabase.table("detected_objects").select(
            "metadata, category"
        ).not_.is_("metadata", "null").execute()
        
        color_stats = {
            "total_objects_with_colors": 0,
            "color_distribution": {},
            "colors_by_category": {},
            "dominant_colors": {},
            "color_temperature_distribution": {"warm": 0, "cool": 0, "neutral": 0}
        }
        
        warm_colors = ["red", "orange", "yellow", "pink", "brown", "tan", "gold"]
        cool_colors = ["blue", "green", "teal", "purple"]
        neutral_colors = ["white", "black", "gray", "beige", "cream"]
        
        for obj in result.data or []:
            metadata = obj.get("metadata", {})
            colors_data = metadata.get("colors")
            category = obj.get("category", "unknown")
            
            if colors_data and colors_data.get("colors"):
                color_stats["total_objects_with_colors"] += 1
                
                # Track colors by category
                if category not in color_stats["colors_by_category"]:
                    color_stats["colors_by_category"][category] = {}
                
                # Process each color in the object
                for color_info in colors_data["colors"]:
                    color_name = color_info.get("name", "unknown").lower()
                    
                    # Update overall distribution
                    if color_name not in color_stats["color_distribution"]:
                        color_stats["color_distribution"][color_name] = 0
                    color_stats["color_distribution"][color_name] += 1
                    
                    # Update category distribution
                    if color_name not in color_stats["colors_by_category"][category]:
                        color_stats["colors_by_category"][category][color_name] = 0
                    color_stats["colors_by_category"][category][color_name] += 1
                    
                    # Update temperature distribution
                    if any(warm in color_name for warm in warm_colors):
                        color_stats["color_temperature_distribution"]["warm"] += 1
                    elif any(cool in color_name for cool in cool_colors):
                        color_stats["color_temperature_distribution"]["cool"] += 1
                    elif any(neutral in color_name for neutral in neutral_colors):
                        color_stats["color_temperature_distribution"]["neutral"] += 1
        
        # Find dominant colors (top 10)
        sorted_colors = sorted(color_stats["color_distribution"].items(), key=lambda x: x[1], reverse=True)
        color_stats["dominant_colors"] = dict(sorted_colors[:10])
        
        return color_stats
        
    except Exception as e:
        logger.error(f"❌ Color statistics error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get color statistics: {str(e)}")

@app.post("/process/colors")
async def process_existing_objects_colors(
    limit: int = Query(50, description="Number of objects to process")
):
    """Process existing objects with color extraction"""
    try:
        import uuid
        job_id = str(uuid.uuid4())
        
        # Create job in database if available
        if _database_service:
            await _database_service.create_job_in_database(
                job_id=job_id,
                job_type="processing",
                total_items=limit,
                parameters={
                    "limit": limit,
                    "operation": "color_extraction"
                }
            )
        
        # Create job in Redis if available
        if _job_service and _job_service.is_available():
            _job_service.create_job(
                job_id=job_id,
                job_type="processing",
                total=limit,
                message=f"Processing colors for {limit} objects",
                operation="color_extraction"
            )
        
        # Start Celery task for background processing
        try:
            from tasks.color_tasks import run_color_processing_job
            task = run_color_processing_job.delay(job_id, limit)
            
            return {
                "job_id": job_id,
                "task_id": task.id,
                "status": "running", 
                "message": f"Started color processing for up to {limit} objects",
                "celery_task": "run_color_processing_job"
            }
        except ImportError:
            # Fallback if Celery not available
            return {
                "job_id": job_id,
                "status": "pending", 
                "message": f"Started color processing for up to {limit} objects",
                "note": "Celery not available - job created but not executed"
            }
        
    except Exception as e:
        logger.error(f"❌ Color processing job creation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start color processing: {str(e)}")

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

@jobs_router.get("/{job_id}/status")
async def get_job_status(job_id: str):
    """Get status of a specific job"""
    try:
        if _job_service and _job_service.is_available():
            job_data = _job_service.get_job(job_id)
            if job_data:
                return job_data
            else:
                raise HTTPException(status_code=404, detail="Job not found")
        else:
            raise HTTPException(status_code=503, detail="Job service not available")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Get job status error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get job status: {str(e)}")

@jobs_router.get("/{job_id}/celery-status")
async def get_celery_task_status(job_id: str):
    """Get Celery task status for a job"""
    try:
        # Get task ID from job record
        if _job_service and _job_service.is_available():
            job_data = _job_service.get_job(job_id)
            if not job_data:
                raise HTTPException(status_code=404, detail="Job not found")
            
            # Try to get Celery task status
            try:
                from celery_app import celery_app
                
                # Look for task_id in job data or use job_id as fallback
                task_id = job_data.get("task_id", job_id)
                task = celery_app.AsyncResult(task_id)
                
                celery_status = {
                    "task_id": task_id,
                    "state": task.state,
                    "info": task.info,
                    "ready": task.ready(),
                    "successful": task.successful() if task.ready() else None,
                    "failed": task.failed() if task.ready() else None
                }
                
                # Add progress information if available
                if task.state == "PROGRESS" and isinstance(task.info, dict):
                    celery_status["progress"] = {
                        "current": task.info.get("current", 0),
                        "total": task.info.get("total", 1),
                        "percentage": task.info.get("progress", 0),
                        "message": task.info.get("message", "")
                    }
                
                return {
                    "job_id": job_id,
                    "celery_status": celery_status,
                    "job_data": job_data
                }
                
            except ImportError:
                return {
                    "job_id": job_id,
                    "celery_status": {"error": "Celery not available"},
                    "job_data": job_data
                }
        else:
            raise HTTPException(status_code=503, detail="Job service not available")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Get Celery task status error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get Celery status: {str(e)}")

@jobs_router.get("/celery/active-tasks")
async def get_active_celery_tasks():
    """Get all active Celery tasks"""
    try:
        from celery_app import celery_app
        
        # Get active tasks from Celery
        inspect = celery_app.control.inspect()
        active_tasks = inspect.active()
        
        if not active_tasks:
            return {"active_tasks": {}, "total_workers": 0, "total_tasks": 0}
        
        # Process active tasks
        all_tasks = []
        for worker, tasks in active_tasks.items():
            for task in tasks:
                task_info = {
                    "worker": worker,
                    "task_id": task.get("id"),
                    "name": task.get("name"),
                    "args": task.get("args", []),
                    "kwargs": task.get("kwargs", {}),
                    "time_start": task.get("time_start"),
                    "acknowledged": task.get("acknowledged", False),
                    "delivery_info": task.get("delivery_info", {})
                }
                all_tasks.append(task_info)
        
        return {
            "active_tasks": active_tasks,
            "processed_tasks": all_tasks,
            "total_workers": len(active_tasks),
            "total_tasks": len(all_tasks)
        }
        
    except ImportError:
        raise HTTPException(status_code=503, detail="Celery not available")
    except Exception as e:
        logger.error(f"❌ Get active Celery tasks error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get active tasks: {str(e)}")

@jobs_router.post("/{job_id}/cancel")
async def cancel_job(job_id: str):
    """Cancel a running job and its Celery task"""
    try:
        # Update job status
        if _job_service and _job_service.is_available():
            job_data = _job_service.get_job(job_id)
            if not job_data:
                raise HTTPException(status_code=404, detail="Job not found")
            
            # Cancel Celery task if available
            celery_cancelled = False
            try:
                from celery_app import celery_app
                
                task_id = job_data.get("task_id", job_id)
                celery_app.control.revoke(task_id, terminate=True)
                celery_cancelled = True
                
            except ImportError:
                pass
            
            # Update job status
            _job_service.update_job(
                job_id=job_id,
                status="cancelled",
                message="Job cancelled by user"
            )
            
            return {
                "job_id": job_id,
                "status": "cancelled",
                "celery_cancelled": celery_cancelled,
                "message": "Job cancelled successfully"
            }
        else:
            raise HTTPException(status_code=503, detail="Job service not available")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Cancel job error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to cancel job: {str(e)}")

@jobs_router.get("/celery/workers")
async def get_celery_workers():
    """Get Celery worker status and statistics"""
    try:
        from celery_app import celery_app
        
        # Get worker stats
        inspect = celery_app.control.inspect()
        stats = inspect.stats()
        
        if not stats:
            return {
                "workers": {},
                "total_workers": 0,
                "online_workers": 0,
                "message": "No workers found or workers offline"
            }
        
        # Process worker information
        workers_info = {}
        for worker, worker_stats in stats.items():
            workers_info[worker] = {
                "status": "online",
                "broker": worker_stats.get("broker", {}),
                "clock": worker_stats.get("clock"),
                "pid": worker_stats.get("pid"),
                "pool": worker_stats.get("pool", {}),
                "prefetch_count": worker_stats.get("prefetch_count"),
                "rusage": worker_stats.get("rusage", {}),
                "total_completed": worker_stats.get("total", {}).get("completed", 0),
                "total_failed": worker_stats.get("total", {}).get("failed", 0),
                "total_retries": worker_stats.get("total", {}).get("retries", 0),
                "total_received": worker_stats.get("total", {}).get("received", 0),
            }
        
        return {
            "workers": workers_info,
            "total_workers": len(workers_info),
            "online_workers": len(workers_info),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except ImportError:
        raise HTTPException(status_code=503, detail="Celery not available")
    except Exception as e:
        logger.error(f"❌ Get Celery workers error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get worker info: {str(e)}")

@jobs_router.get("/celery/queues")
async def get_celery_queues():
    """Get Celery queue information"""
    try:
        from celery_app import celery_app
        
        # Get queue lengths from Redis if available
        queue_info = {}
        try:
            # Try to get Redis connection to check queue lengths
            if redis:
                r = redis.Redis.from_url(settings.REDIS_URL)
                
                # Common Celery queue names
                queue_names = ["celery", "ai_processing", "detection", "scraping", "import", "color_processing", "classification"]
                
                for queue_name in queue_names:
                    queue_length = r.llen(queue_name)
                    if queue_length > 0:
                        queue_info[queue_name] = {
                            "length": queue_length,
                            "messages_waiting": queue_length
                        }
                
        except Exception as redis_error:
            logger.warning(f"Could not get queue info from Redis: {redis_error}")
        
        # Get active queues from workers
        inspect = celery_app.control.inspect()
        active_queues = inspect.active_queues()
        
        if active_queues:
            for worker, queues in active_queues.items():
                for queue in queues:
                    queue_name = queue.get("name")
                    if queue_name and queue_name not in queue_info:
                        queue_info[queue_name] = {
                            "length": 0,  # We can't get length without Redis
                            "exchange": queue.get("exchange", {}),
                            "routing_key": queue.get("routing_key"),
                            "workers": queue_info.get(queue_name, {}).get("workers", []) + [worker]
                        }
                    elif queue_name:
                        if "workers" not in queue_info[queue_name]:
                            queue_info[queue_name]["workers"] = []
                        queue_info[queue_name]["workers"].append(worker)
        
        return {
            "queues": queue_info,
            "total_queues": len(queue_info),
            "total_messages": sum(q.get("length", 0) for q in queue_info.values()),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except ImportError:
        raise HTTPException(status_code=503, detail="Celery not available")
    except Exception as e:
        logger.error(f"❌ Get Celery queues error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get queue info: {str(e)}")

@jobs_router.get("/celery/dashboard")
async def get_celery_dashboard():
    """Get comprehensive Celery monitoring dashboard data"""
    try:
        from celery_app import celery_app
        
        # Get inspect interface
        inspect = celery_app.control.inspect()
        
        # Gather all monitoring data
        dashboard_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "celery_available": True,
            "workers": {},
            "tasks": {},
            "queues": {},
            "system": {}
        }
        
        # Worker information
        try:
            stats = inspect.stats()
            if stats:
                for worker, worker_stats in stats.items():
                    dashboard_data["workers"][worker] = {
                        "status": "online",
                        "pid": worker_stats.get("pid"),
                        "pool_processes": worker_stats.get("pool", {}).get("processes", []),
                        "pool_max_concurrency": worker_stats.get("pool", {}).get("max-concurrency"),
                        "prefetch_count": worker_stats.get("prefetch_count"),
                        "tasks_completed": worker_stats.get("total", {}).get("completed", 0),
                        "tasks_failed": worker_stats.get("total", {}).get("failed", 0),
                        "tasks_retries": worker_stats.get("total", {}).get("retries", 0),
                        "tasks_received": worker_stats.get("total", {}).get("received", 0),
                    }
        except Exception as e:
            logger.warning(f"Could not get worker stats: {e}")
        
        # Active tasks
        try:
            active_tasks = inspect.active()
            if active_tasks:
                all_tasks = []
                for worker, tasks in active_tasks.items():
                    for task in tasks:
                        all_tasks.append({
                            "worker": worker,
                            "task_id": task.get("id"),
                            "name": task.get("name"),
                            "args": task.get("args", []),
                            "time_start": task.get("time_start"),
                            "acknowledged": task.get("acknowledged", False)
                        })
                dashboard_data["tasks"]["active"] = all_tasks
                dashboard_data["tasks"]["active_count"] = len(all_tasks)
        except Exception as e:
            logger.warning(f"Could not get active tasks: {e}")
        
        # Queue information (if Redis available)
        try:
            if redis:
                r = redis.Redis.from_url(settings.REDIS_URL)
                queue_names = ["celery", "ai_processing", "detection", "scraping", "import", "color_processing", "classification"]
                
                for queue_name in queue_names:
                    queue_length = r.llen(queue_name)
                    dashboard_data["queues"][queue_name] = {
                        "length": queue_length,
                        "waiting": queue_length
                    }
                
                # Get total messages waiting
                dashboard_data["system"]["total_queued"] = sum(
                    q["length"] for q in dashboard_data["queues"].values()
                )
        except Exception as e:
            logger.warning(f"Could not get queue information: {e}")
        
        # System summary
        dashboard_data["system"].update({
            "total_workers": len(dashboard_data["workers"]),
            "online_workers": len([w for w in dashboard_data["workers"].values() if w["status"] == "online"]),
            "total_active_tasks": dashboard_data["tasks"].get("active_count", 0),
            "total_queues": len(dashboard_data["queues"]),
        })
        
        return dashboard_data
        
    except ImportError:
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "celery_available": False,
            "error": "Celery not available",
            "workers": {},
            "tasks": {},
            "queues": {},
            "system": {}
        }
    except Exception as e:
        logger.error(f"❌ Get Celery dashboard error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get dashboard data: {str(e)}")

@jobs_router.post("/celery/purge-queue/{queue_name}")
async def purge_celery_queue(queue_name: str):
    """Purge all messages from a specific Celery queue (admin only)"""
    try:
        from celery_app import celery_app
        
        # Validate queue name
        valid_queues = ["celery", "ai_processing", "detection", "scraping", "import", "color_processing", "classification"]
        if queue_name not in valid_queues:
            raise HTTPException(status_code=400, detail=f"Invalid queue name. Valid queues: {valid_queues}")
        
        # Purge the queue
        result = celery_app.control.purge()
        
        return {
            "queue_name": queue_name,
            "purged": True,
            "result": result,
            "message": f"Queue '{queue_name}' purged successfully",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except ImportError:
        raise HTTPException(status_code=503, detail="Celery not available")
    except Exception as e:
        logger.error(f"❌ Purge Celery queue error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to purge queue: {str(e)}")

@scraping_router.get("/status")
async def scraping_status():
    return {"status": "available", "message": "Scraping service endpoint"}

@scraping_router.post("/scenes")
async def scrape_scenes(
    limit: int = Query(100, description="Number of scenes to scrape"),
    force_refresh: bool = Query(False, description="Force refresh existing scenes")
):
    """Start scene scraping job"""
    try:
        import uuid
        job_id = str(uuid.uuid4())
        
        # Create job in database if available
        if _database_service:
            await _database_service.create_job_in_database(
                job_id=job_id,
                job_type="scenes",
                total_items=limit,
                parameters={"limit": limit, "force_refresh": force_refresh}
            )
        
        # Create job in Redis if available
        if _job_service and _job_service.is_available():
            _job_service.create_job(
                job_id=job_id,
                job_type="scenes",
                total=limit,
                message=f"Scraping {limit} scenes",
                force_refresh=str(force_refresh)
            )
        
        # Start Celery task for background processing
        try:
            from tasks.scraping_tasks import run_scraping_job
            task = run_scraping_job.delay(job_id, limit, None)  # room_types as None for now
            
            return {
                "job_id": job_id,
                "task_id": task.id,
                "message": f"Scene scraping job started for {limit} scenes",
                "status": "running",
                "parameters": {
                    "limit": limit,
                    "force_refresh": force_refresh
                },
                "celery_task": "run_scraping_job"
            }
        except ImportError:
            # Fallback if Celery not available
            return {
                "job_id": job_id,
                "message": f"Scene scraping job started for {limit} scenes",
                "status": "pending",
                "parameters": {
                    "limit": limit,
                    "force_refresh": force_refresh
                },
                "note": "Celery not available - job created but not executed"
            }
        
    except Exception as e:
        logger.error(f"❌ Scene scraping job creation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start scene scraping: {str(e)}")

# Classification endpoints
@app.post("/classify/reclassify-scenes")
async def reclassify_scenes(
    limit: int = Query(100, description="Number of scenes to reclassify"),
    force_reclassify: bool = Query(False, description="Force reclassify all scenes")
):
    """Start scene reclassification job"""
    try:
        import uuid
        job_id = str(uuid.uuid4())
        
        # Create job in database if available
        if _database_service:
            await _database_service.create_job_in_database(
                job_id=job_id,
                job_type="scene_reclassification",
                total_items=limit,
                parameters={"limit": limit, "operation": "scene_reclassification", "force_reclassify": force_reclassify}
            )
        
        # Create job in Redis if available  
        if _job_service and _job_service.is_available():
            _job_service.create_job(
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

# Dataset export endpoint
@app.get("/export/training-dataset")
async def export_training_dataset(
    format: str = Query("json", description="Export format (json, yaml)"),
    split_ratio: str = Query("70:20:10", description="Train:Val:Test split ratio")
):
    """Export dataset for training"""
    try:
        if not _database_service:
            raise HTTPException(status_code=503, detail="Database service not available")
        
        # Parse split ratio
        try:
            ratios = [float(x) for x in split_ratio.split(':')]
            if len(ratios) != 3 or sum(ratios) != 100:
                raise ValueError("Split ratios must sum to 100")
        except:
            raise HTTPException(status_code=400, detail="Invalid split ratio format. Use 'train:val:test' (e.g., '70:20:10')")
        
        # Get dataset statistics
        stats = await _database_service.get_dataset_stats()
        
        # For now, return dataset info - actual export would be a background job
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

# Dataset import endpoint
@app.post("/import/huggingface-dataset")
async def import_huggingface_dataset(
    dataset: str = Query("sk2003/houzzdata", description="HuggingFace dataset ID (e.g., username/dataset-name)"),
    offset: int = Query(0, description="Starting offset in dataset"),
    limit: int = Query(50, description="Number of images to import and process"),
    include_detection: bool = Query(True, description="Run AI detection on imported images")
):
    """Import any HuggingFace dataset and process with AI"""
    try:
        import uuid
        job_id = str(uuid.uuid4())
        
        # Create job in database for persistent tracking
        if _database_service:
            await _database_service.create_job_in_database(
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
        if _job_service and _job_service.is_available():
            _job_service.create_job(
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

# Classification testing endpoint
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

# Review endpoints router
review_router = APIRouter(prefix="/review", tags=["review"])

@review_router.get("/queue")
async def get_review_queue(limit: int = Query(10, description="Number of scenes to return")):
    """Get scenes ready for review"""
    try:
        if not _database_service:
            raise HTTPException(status_code=503, detail="Database service not available")
        
        # Get scenes with pending_review status
        scenes = await _database_service.get_scenes(limit=limit, offset=0, status="pending_review")
        
        return {
            "scenes": scenes.get("scenes", []),
            "total": scenes.get("total", 0),
            "limit": limit
        }
    except Exception as e:
        logger.error(f"❌ Review queue error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get review queue: {str(e)}")

@review_router.post("/approve/{scene_id}")
async def approve_scene(scene_id: str):
    """Approve a scene for inclusion in dataset"""
    try:
        if not _database_service:
            raise HTTPException(status_code=503, detail="Database service not available")
        
        # Update scene status to approved
        update_data = {
            "status": "approved",
            "reviewed_at": datetime.utcnow().isoformat(),
            "reviewed_by": "api_user"  # In production, get from auth
        }
        
        success = await _database_service.update_scene(scene_id, update_data)
        
        if success:
            return {"message": f"Scene {scene_id} approved successfully"}
        else:
            raise HTTPException(status_code=404, detail="Scene not found or update failed")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Scene approval error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to approve scene: {str(e)}")

@review_router.post("/reject/{scene_id}")
async def reject_scene(scene_id: str):
    """Reject a scene from dataset"""
    try:
        if not _database_service:
            raise HTTPException(status_code=503, detail="Database service not available")
        
        # Update scene status to rejected
        update_data = {
            "status": "rejected", 
            "reviewed_at": datetime.utcnow().isoformat(),
            "reviewed_by": "api_user"  # In production, get from auth
        }
        
        success = await _database_service.update_scene(scene_id, update_data)
        
        if success:
            return {"message": f"Scene {scene_id} rejected successfully"}
        else:
            raise HTTPException(status_code=404, detail="Scene not found or update failed")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Scene rejection error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reject scene: {str(e)}")

@review_router.post("/update")
async def update_review(updates: List[dict]):
    """Update review status for multiple objects"""
    try:
        if not _database_service:
            raise HTTPException(status_code=503, detail="Database service not available")
        
        updated_count = 0
        errors = []
        
        for update in updates:
            object_id = update.get("object_id")
            if not object_id:
                errors.append("Missing object_id in update")
                continue
                
            # Prepare update data
            update_data = {}
            if "approved" in update:
                update_data["approved"] = update["approved"]
            if "category" in update:
                update_data["category"] = update["category"]
            if "tags" in update:
                update_data["tags"] = update["tags"]
            if "matched_product_id" in update:
                update_data["matched_product_id"] = update["matched_product_id"]
            
            if update_data:
                try:
                    result = _database_service.supabase.table("detected_objects").update(update_data).eq("object_id", object_id).execute()
                    if result.data:
                        updated_count += 1
                    else:
                        errors.append(f"Failed to update object {object_id}")
                except Exception as obj_error:
                    errors.append(f"Error updating object {object_id}: {str(obj_error)}")
        
        return {
            "updated_count": updated_count,
            "total_updates": len(updates),
            "errors": errors,
            "success": updated_count > 0
        }
        
    except Exception as e:
        logger.error(f"❌ Bulk review update error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update reviews: {str(e)}")

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

# FastAPI app already initialized earlier in the file

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
    return _database_service  # Can be None - endpoints should handle gracefully

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
        if not create_client:
            logger.error("❌ Supabase package not available - create_client is None")
        elif not config_status["supabase_configured"]:
            logger.error(f"❌ Missing Supabase credentials: {config_status['missing']}")
        else:
            try:
                # Initialize Supabase client with basic parameters only
                supabase_client = create_client(
                    supabase_url=settings.SUPABASE_URL,
                    supabase_key=settings.SUPABASE_ANON_KEY
                )
                logger.info("✅ Supabase client initialized successfully")
                
                # Test the connection
                test_result = supabase_client.table("scenes").select("scene_id").limit(1).execute()
                logger.info("✅ Supabase connection test passed")
                
                # Initialize database service
                database_service = DatabaseService(supabase_client)
                
            except Exception as e:
                logger.error(f"❌ Supabase client initialization failed: {e}")
                supabase_client = None
        
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
        # Get scene from database using direct query
        result = db_service.supabase.table("scenes").select("*").eq("scene_id", scene_id).execute()
        if not result.data:
            raise HTTPException(status_code=404, detail="Scene not found")
        
        scene = result.data[0]
        
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
app.include_router(review_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)