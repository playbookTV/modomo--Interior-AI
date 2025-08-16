"""
Analytics and statistics API routes
"""
from fastapi import APIRouter, Query, Depends
from typing import Dict, Any, List
import structlog

from services.database_service import DatabaseService
from services.detection_service import DetectionService
from config.taxonomy import MODOMO_TAXONOMY, get_all_categories

logger = structlog.get_logger(__name__)
router = APIRouter(tags=["analytics"])


@router.get("/taxonomy")
async def get_taxonomy():
    """Get the furniture taxonomy"""
    return MODOMO_TAXONOMY


@router.get("/stats/dataset")
async def get_dataset_stats(db_service: DatabaseService = Depends()):
    """Get dataset statistics"""
    stats = await db_service.get_dataset_stats()
    
    # Add taxonomy info
    stats["unique_categories"] = len(get_all_categories())
    
    return stats


@router.get("/stats/categories")
async def get_category_stats(db_service: DatabaseService = Depends()):
    """Get category-wise statistics"""
    try:
        # Get category stats from detected objects
        result = db_service.supabase.table("detected_objects").select("category, confidence, approved").execute()
        
        # Process the data
        category_stats = {}
        for obj in result.data or []:
            category = obj.get("category", "unknown")
            confidence = obj.get("confidence", 0.0)
            approved = obj.get("approved")
            
            if category not in category_stats:
                category_stats[category] = {
                    "category": category,
                    "total_objects": 0,
                    "approved_objects": 0,
                    "confidences": [],
                    "group": None
                }
            
            category_stats[category]["total_objects"] += 1
            category_stats[category]["confidences"].append(confidence)
            
            if approved is True:
                category_stats[category]["approved_objects"] += 1
            
            # Map to taxonomy group
            for group_name, items in MODOMO_TAXONOMY.items():
                if category in items:
                    category_stats[category]["group"] = group_name
                    break
        
        # Calculate averages and format response
        categories = []
        for category, stats in category_stats.items():
            avg_confidence = sum(stats["confidences"]) / len(stats["confidences"]) if stats["confidences"] else 0.0
            categories.append({
                "category": category,
                "total_objects": stats["total_objects"],
                "approved_objects": stats["approved_objects"],
                "avg_confidence": avg_confidence,
                "group": stats["group"] or "other"
            })
        
        return sorted(categories, key=lambda x: x["total_objects"], reverse=True)
        
    except Exception as e:
        logger.error(f"Category stats query failed: {e}")
        
        # Fallback to dummy stats
        categories = []
        for category_group, items in MODOMO_TAXONOMY.items():
            for item in items:
                categories.append({
                    "category": item,
                    "total_objects": 0,
                    "approved_objects": 0,
                    "avg_confidence": 0.0,
                    "group": category_group
                })
        return categories


@router.get("/colors/extract")
async def extract_colors_from_url(
    image_url: str = Query(..., description="URL of the image to analyze"),
    bbox: str = Query(None, description="Bounding box as 'x,y,width,height' for object crop"),
    detection_service: DetectionService = Depends()
):
    """Extract dominant colors from an image or object crop"""
    # Parse bbox if provided
    bbox_coords = None
    if bbox:
        try:
            bbox_coords = [float(x) for x in bbox.split(',')]
            if len(bbox_coords) != 4:
                return {"error": "Bbox must have exactly 4 values: x,y,width,height"}
        except ValueError:
            return {"error": "Invalid bbox format. Use: x,y,width,height"}
    
    return await detection_service.extract_colors_from_url(image_url, bbox_coords)


@router.get("/colors/palette")
async def get_color_palette(detection_service: DetectionService = Depends()):
    """Get available color names and their RGB values for filtering"""
    if not detection_service.color_extractor:
        return {"error": "Color extractor not available"}
    
    # Return the color mappings from the extractor
    return {
        "color_palette": detection_service.color_extractor.color_mappings if hasattr(detection_service.color_extractor, 'color_mappings') else {},
        "color_categories": {
            "neutrals": ["white", "black", "gray", "beige", "cream"],
            "warm": ["red", "orange", "yellow", "pink", "brown", "tan", "gold"],
            "cool": ["blue", "green", "teal", "purple"],
            "wood_tones": ["light_wood", "medium_wood", "dark_wood"]
        }
    }


@router.get("/stats/colors")
async def get_color_statistics(db_service: DatabaseService = Depends()):
    """Get statistics about colors in the dataset"""
    try:
        # Get all objects with color metadata
        result = db_service.supabase.table("detected_objects").select(
            "metadata, category"
        ).not_.is_("metadata", "null").execute()
        
        color_stats = {
            "total_objects_with_colors": 0,
            "color_distribution": {},
            "colors_by_category": {},
            "dominant_colors": {},
            "color_temperature_distribution": {"warm": 0, "cool": 0, "neutral": 0}
        }
        
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
                    color_name = color_info.get("name", "unknown")
                    
                    # Overall color distribution
                    color_stats["color_distribution"][color_name] = \
                        color_stats["color_distribution"].get(color_name, 0) + 1
                    
                    # Colors by category
                    color_stats["colors_by_category"][category][color_name] = \
                        color_stats["colors_by_category"][category].get(color_name, 0) + 1
                
                # Track dominant color
                dominant = colors_data.get("dominant_color", {})
                if dominant.get("name"):
                    dom_color = dominant["name"]
                    color_stats["dominant_colors"][dom_color] = \
                        color_stats["dominant_colors"].get(dom_color, 0) + 1
                
                # Track color temperature
                props = colors_data.get("properties", {})
                temp = props.get("color_temperature", "neutral")
                if temp in color_stats["color_temperature_distribution"]:
                    color_stats["color_temperature_distribution"][temp] += 1
        
        return color_stats
        
    except Exception as e:
        logger.error(f"Color statistics API failed: {e}")
        return {"error": str(e)}