"""
Color processing endpoints extracted from main file
"""
from fastapi import FastAPI, HTTPException, Query
from core.dependencies import get_database_service, get_detection_service
from utils.logging import get_logger

logger = get_logger(__name__)


def register_color_routes(app: FastAPI):
    """Register color-related endpoints to the app"""
    
    @app.get("/colors/extract")
    async def extract_colors(
        image_url: str = Query(..., description="URL of image to extract colors from"),
        bbox: str = Query(None, description="Optional bounding box as 'x,y,w,h'")
    ):
        """Extract colors from image or image region"""
        try:
            detection_service = get_detection_service()
            if not detection_service:
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
            color_data = await detection_service.extract_colors_from_url(image_url, bbox_list)
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
            database_service = get_database_service()
            if not database_service:
                raise HTTPException(status_code=503, detail="Database service not available")
            
            # Simple color search implementation
            result = database_service.supabase.table("detected_objects").select(
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
            detection_service = get_detection_service()
            if not detection_service or not detection_service.color_extractor:
                raise HTTPException(status_code=503, detail="Color extractor not available")
            
            color_extractor = detection_service.color_extractor
            
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
            database_service = get_database_service()
            if not database_service:
                raise HTTPException(status_code=503, detail="Database service not available")
            
            # Get all objects with color metadata
            result = database_service.supabase.table("detected_objects").select(
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
            
            database_service = get_database_service()
            
            # Create job in database if available
            if database_service:
                await database_service.create_job_in_database(
                    job_id=job_id,
                    job_type="processing",
                    total_items=limit,
                    parameters={
                        "limit": limit,
                        "operation": "color_extraction"
                    }
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