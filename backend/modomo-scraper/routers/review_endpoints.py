"""
Review queue endpoints extracted from main file
"""
from fastapi import FastAPI, HTTPException, Query
from datetime import datetime
from core.dependencies import get_database_service
from utils.logging import get_logger

logger = get_logger(__name__)


def register_review_routes(app: FastAPI):
    """Register review-related endpoints to the app"""
    
    @app.get("/review/queue")
    async def get_review_queue(
        limit: int = Query(20, description="Number of scenes to return"),
        offset: int = Query(0, description="Number of scenes to skip"),
        room_type: str = Query("", description="Filter by room type"),
        category: str = Query("", description="Filter by object category"), 
        status: str = Query("", description="Filter by scene status"),
        search: str = Query("", description="Search in scene descriptions"),
        image_type: str = Query("", description="Filter by image type"),
        has_masks: bool = Query(None, description="Filter by SAM2 mask availability"),
        order_by: str = Query("created_at", description="Field to order by"),
        order_dir: str = Query("desc", description="Order direction (asc/desc)")
    ):
        """Get scenes ready for review with comprehensive filtering and pagination"""
        try:
            database_service = get_database_service()
            if not database_service:
                raise HTTPException(status_code=503, detail="Database service not available")
            
            # Build dynamic query with filters
            query = database_service.supabase.table("scenes").select(
                "scene_id, houzz_id, image_url, image_r2_key, room_type, style_tags, color_tags, status, created_at, metadata"
            )
            
            # Apply filters
            if room_type:
                query = query.eq("room_type", room_type)
            if status:
                query = query.eq("status", status) 
            if search:
                query = query.ilike("houzz_id", f"%{search}%")
            if image_type:
                query = query.contains("metadata", {"image_type": image_type})
                
            # Apply ordering
            order_field = order_by if order_by in ["created_at", "room_type", "status"] else "created_at"
            if order_dir.lower() == "asc":
                query = query.order(order_field, desc=False)
            else:
                query = query.order(order_field, desc=True)
                
            # Get total count for pagination
            count_result = database_service.supabase.table("scenes").select("scene_id", count="exact")
            if room_type:
                count_result = count_result.eq("room_type", room_type)
            if status:
                count_result = count_result.eq("status", status)
            if search:
                count_result = count_result.ilike("houzz_id", f"%{search}%")
            if image_type:
                count_result = count_result.contains("metadata", {"image_type": image_type})
                
            count_data = count_result.execute()
            total_scenes = count_data.count or 0
            
            # Apply pagination
            scenes_result = query.range(offset, offset + limit - 1).execute()
            
            # Enhance each scene with its detected objects
            enhanced_scenes = []
            for scene in scenes_result.data or []:
                # Get detected objects for this scene  
                objects_result = database_service.supabase.table("detected_objects").select(
                    "object_id, category, confidence, bbox, tags, mask_url, mask_r2_key, matched_product_id, approved, metadata"
                ).eq("scene_id", scene["scene_id"]).execute()
                
                # Include ALL scenes - even those without detected objects (they need review/processing too!)
                if category and (not objects_result.data or len(objects_result.data) == 0):
                    continue  # Skip only when filtering by specific category and no objects match
                
                scene_dict = dict(scene)
                scene_dict["objects"] = objects_result.data or []
                scene_dict["object_count"] = len(objects_result.data or [])
                
                # Add classification data if available in metadata
                if scene.get("metadata"):
                    scene_dict["image_type"] = scene["metadata"].get("image_type")
                    scene_dict["is_primary_object"] = scene["metadata"].get("is_primary_object", False)
                    scene_dict["primary_category"] = scene["metadata"].get("primary_category")
                    scene_dict["classification_confidence"] = scene["metadata"].get("classification_confidence")
                    scene_dict["classification_reason"] = scene["metadata"].get("classification_reason")
                
                # Apply has_masks filter
                if has_masks is not None:
                    objects_with_masks = [obj for obj in scene_dict["objects"] if obj.get("mask_url")]
                    scene_has_masks = len(objects_with_masks) > 0
                    if has_masks and not scene_has_masks:
                        continue
                    if not has_masks and scene_has_masks:
                        continue
                    scene_dict["masks_count"] = len(objects_with_masks)
                
                enhanced_scenes.append(scene_dict)
            
            # Calculate pagination info
            has_more = (offset + len(enhanced_scenes)) < total_scenes
            
            return {
                "scenes": enhanced_scenes,
                "pagination": {
                    "total": total_scenes,
                    "limit": limit,
                    "offset": offset,
                    "has_more": has_more,
                    "current_page": (offset // limit) + 1,
                    "total_pages": (total_scenes + limit - 1) // limit
                },
                "filters_applied": {
                    "room_type": room_type,
                    "category": category,
                    "status": status,
                    "search": search,
                    "image_type": image_type,
                    "has_masks": has_masks,
                    "order_by": f"{order_field} {order_dir}"
                },
                "debug": {
                    "total_scenes_in_db": total_scenes,
                    "scenes_after_filters": len(enhanced_scenes),
                    "query_offset": offset,
                    "query_limit": limit
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Review queue error: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get review queue: {str(e)}")

    @app.get("/review/scene/{scene_id}")
    async def get_scene_for_review(scene_id: str):
        """Get a specific scene with all its detected objects for review"""
        try:
            database_service = get_database_service()
            if not database_service:
                raise HTTPException(status_code=503, detail="Database service not available")
            
            # Get scene details
            scene_result = database_service.supabase.table("scenes").select(
                "scene_id, houzz_id, image_url, image_r2_key, room_type, style_tags, color_tags, status, created_at, metadata"
            ).eq("scene_id", scene_id).execute()
            
            if not scene_result.data:
                raise HTTPException(status_code=404, detail="Scene not found")
            
            scene = dict(scene_result.data[0])
            
            # Get detected objects for this scene
            objects_result = database_service.supabase.table("detected_objects").select(
                "object_id, category, confidence, bbox, tags, mask_url, mask_r2_key, matched_product_id, approved, metadata"
            ).eq("scene_id", scene_id).execute()
            
            scene["objects"] = objects_result.data or []
            scene["object_count"] = len(objects_result.data or [])
            
            return scene
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"❌ Get scene for review error: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get scene for review: {str(e)}")

    @app.post("/review/approve/{scene_id}")
    async def approve_scene(scene_id: str):
        """Approve a scene for inclusion in dataset"""
        try:
            database_service = get_database_service()
            if not database_service:
                raise HTTPException(status_code=503, detail="Database service not available")
            
            # Update scene status to approved
            update_data = {
                "status": "approved",
                "reviewed_at": datetime.utcnow().isoformat(),
                "reviewed_by": "api_user"  # In production, get from auth
            }
            
            success = await database_service.update_scene(scene_id, update_data)
            
            if success:
                return {"message": f"Scene {scene_id} approved successfully"}
            else:
                raise HTTPException(status_code=404, detail="Scene not found or update failed")
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"❌ Scene approval error: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to approve scene: {str(e)}")

    @app.post("/review/reject/{scene_id}")
    async def reject_scene(scene_id: str):
        """Reject a scene from dataset"""
        try:
            database_service = get_database_service()
            if not database_service:
                raise HTTPException(status_code=503, detail="Database service not available")
            
            # Update scene status to rejected
            update_data = {
                "status": "rejected",
                "reviewed_at": datetime.utcnow().isoformat(),
                "reviewed_by": "api_user"
            }
            
            success = await database_service.update_scene(scene_id, update_data)
            
            if success:
                return {"message": f"Scene {scene_id} rejected successfully"}
            else:
                raise HTTPException(status_code=404, detail="Scene not found or update failed")
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"❌ Scene rejection error: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to reject scene: {str(e)}")