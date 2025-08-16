"""
Analytics and statistics API routes - simplified version
"""
from fastapi import APIRouter, Query
from typing import Dict, Any, List
import structlog

from config.taxonomy import MODOMO_TAXONOMY, get_all_categories

logger = structlog.get_logger(__name__)
router = APIRouter(tags=["analytics"])


@router.get("/taxonomy")
async def get_taxonomy():
    """Get the furniture taxonomy"""
    return MODOMO_TAXONOMY


@router.get("/stats/dataset")
async def get_dataset_stats():
    """Get dataset statistics"""
    try:
        from main_refactored import get_database_service
        database_service = get_database_service()
        
        if not database_service:
            # Fallback stats
            return {
                "total_scenes": 0,
                "approved_scenes": 0,
                "total_objects": 0,
                "approved_objects": 0,
                "unique_categories": len(get_all_categories()),
                "avg_confidence": 0.0,
                "objects_with_products": 0
            }
        
        stats = await database_service.get_dataset_stats()
        
        # Add taxonomy info
        stats["unique_categories"] = len(get_all_categories())
        
        return stats
    except Exception as e:
        logger.error(f"Dataset stats failed: {e}")
        return {
            "total_scenes": 0,
            "approved_scenes": 0,
            "total_objects": 0,
            "approved_objects": 0,
            "unique_categories": len(get_all_categories()),
            "avg_confidence": 0.0,
            "objects_with_products": 0
        }


@router.get("/stats/categories")
async def get_category_stats():
    """Get category-wise statistics"""
    try:
        from main_refactored import get_database_service
        database_service = get_database_service()
        
        if not database_service or not database_service.supabase:
            # Fallback to taxonomy structure
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
        
        # Get category stats from detected objects
        result = database_service.supabase.table("detected_objects").select("category, confidence, approved").execute()
        
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