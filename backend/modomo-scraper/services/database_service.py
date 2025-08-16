"""
Database service for Supabase operations
"""
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List
try:
    from supabase import Client
except ImportError:
    Client = None
import structlog

logger = structlog.get_logger(__name__)


class DatabaseService:
    """Service for database operations using Supabase"""
    
    def __init__(self, supabase_client: Client):
        self.supabase = supabase_client
    
    async def create_job_in_database(
        self,
        job_id: str,
        job_type: str,
        total_items: int,
        parameters: Dict[str, Any]
    ) -> bool:
        """Create a new job record in the database"""
        try:
            job_data = {
                "job_id": job_id,
                "job_type": job_type,
                "status": "pending",
                "total_items": total_items,
                "processed_items": 0,
                "progress": 0,
                "parameters": parameters,
                "created_at": datetime.utcnow().isoformat(),
                "started_at": datetime.utcnow().isoformat()
            }
            
            result = self.supabase.table("scraping_jobs").insert(job_data).execute()
            
            if result.data:
                logger.info(f"✅ Created job {job_id} in database")
                return True
            else:
                logger.error(f"❌ Failed to create job {job_id} in database")
                return False
                
        except Exception as e:
            logger.error(f"❌ Database job creation failed for {job_id}: {e}")
            return False
    
    async def update_job_progress(
        self,
        job_id: str,
        processed_items: int,
        total_items: int,
        status: str = "running",
        error_message: Optional[str] = None
    ) -> bool:
        """Update job progress in database"""
        try:
            progress = (processed_items / total_items * 100) if total_items > 0 else 0
            
            update_data = {
                "processed_items": processed_items,
                "progress": progress,
                "status": status,
                "updated_at": datetime.utcnow().isoformat()
            }
            
            if error_message:
                update_data["error_message"] = error_message
            
            if status == "completed":
                update_data["completed_at"] = datetime.utcnow().isoformat()
            
            result = self.supabase.table("scraping_jobs").update(update_data).eq("job_id", job_id).execute()
            return bool(result.data)
            
        except Exception as e:
            logger.error(f"❌ Failed to update job progress for {job_id}: {e}")
            return False
    
    async def get_scenes(
        self,
        limit: int = 20,
        offset: int = 0,
        status: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get scenes with pagination and filtering"""
        try:
            # Build query
            query = self.supabase.table("scenes").select(
                "scene_id, houzz_id, image_url, image_r2_key, room_type, style_tags, color_tags, status, created_at"
            )
            
            # Add status filter if provided
            if status:
                query = query.eq("status", status)
            
            # Execute query with pagination
            result = query.order("created_at", desc=True).range(offset, offset + limit - 1).execute()
            
            # Get total count
            count_query = self.supabase.table("scenes").select("scene_id", count="exact")
            if status:
                count_query = count_query.eq("status", status)
            count_result = count_query.execute()
            total = count_result.count if count_result.count else 0
            
            # Add object count for each scene
            scenes_data = []
            for scene in result.data:
                scene_dict = dict(scene)
                
                # Get object count for this scene
                try:
                    objects_count_result = self.supabase.table("detected_objects").select(
                        "object_id", count="exact"
                    ).eq("scene_id", scene["scene_id"]).execute()
                    scene_dict["object_count"] = objects_count_result.count if objects_count_result.count else 0
                except:
                    scene_dict["object_count"] = 0
                
                scenes_data.append(scene_dict)
            
            return {
                "scenes": scenes_data,
                "total": total,
                "limit": limit,
                "offset": offset
            }
        except Exception as e:
            logger.error(f"Failed to fetch scenes: {e}")
            return {"scenes": [], "total": 0, "limit": limit, "offset": offset}
    
    async def get_detected_objects(
        self,
        limit: int = 20,
        offset: int = 0,
        category: Optional[str] = None,
        scene_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get detected objects with pagination and filtering"""
        try:
            # Build query
            query = self.supabase.table("detected_objects").select(
                "object_id, scene_id, category, confidence, bbox, tags, approved, metadata, mask_r2_key, mask_url, created_at"
            )
            
            # Add filters
            if category:
                query = query.eq("category", category)
            if scene_id:
                query = query.eq("scene_id", scene_id)
            
            # Execute with pagination
            result = query.order("created_at", desc=True).range(offset, offset + limit - 1).execute()
            
            # Get total count
            count_query = self.supabase.table("detected_objects").select("object_id", count="exact")
            if category:
                count_query = count_query.eq("category", category)
            if scene_id:
                count_query = count_query.eq("scene_id", scene_id)
            count_result = count_query.execute()
            total = count_result.count if count_result.count else 0
            
            # Enrich with scene info
            objects_data = []
            for obj in result.data:
                obj_dict = dict(obj)
                
                # Get scene info
                try:
                    scene_result = self.supabase.table("scenes").select(
                        "houzz_id, image_url, room_type"
                    ).eq("scene_id", obj["scene_id"]).execute()
                    
                    if scene_result.data:
                        scene_info = scene_result.data[0]
                        obj_dict["scene_info"] = scene_info
                    else:
                        obj_dict["scene_info"] = None
                        
                except Exception as scene_error:
                    logger.warning(f"Failed to fetch scene info for object {obj['object_id']}: {scene_error}")
                    obj_dict["scene_info"] = None
                
                objects_data.append(obj_dict)
            
            return {
                "objects": objects_data,
                "total": total,
                "limit": limit,
                "offset": offset
            }
            
        except Exception as e:
            logger.error(f"Failed to fetch objects: {e}")
            return {"objects": [], "total": 0, "limit": limit, "offset": offset}
    
    async def get_dataset_stats(self) -> Dict[str, Any]:
        """Get comprehensive dataset statistics"""
        try:
            # Get scenes count
            scenes_result = self.supabase.table("scenes").select("scene_id", count="exact").execute()
            total_scenes = scenes_result.count if scenes_result.count else 0
            
            # Get approved scenes count  
            approved_scenes_result = self.supabase.table("scenes").select(
                "scene_id", count="exact"
            ).eq("status", "approved").execute()
            approved_scenes = approved_scenes_result.count if approved_scenes_result.count else 0
            
            # Get detected objects counts
            objects_result = self.supabase.table("detected_objects").select("object_id", count="exact").execute()
            total_objects = objects_result.count if objects_result.count else 0
            
            approved_objects_result = self.supabase.table("detected_objects").select(
                "object_id", count="exact"
            ).eq("approved", True).execute()
            approved_objects = approved_objects_result.count if approved_objects_result.count else 0
            
            # Get average confidence
            try:
                confidence_result = self.supabase.table("detected_objects").select("confidence").execute()
                confidences = [obj["confidence"] for obj in confidence_result.data if obj.get("confidence")]
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            except:
                avg_confidence = 0.0
            
            # Get objects with matched products
            matched_objects_result = self.supabase.table("detected_objects").select(
                "object_id", count="exact"
            ).not_.is_("matched_product_id", "null").execute()
            objects_with_products = matched_objects_result.count if matched_objects_result.count else 0
            
            return {
                "total_scenes": total_scenes,
                "approved_scenes": approved_scenes,
                "total_objects": total_objects,
                "approved_objects": approved_objects,
                "avg_confidence": avg_confidence,
                "objects_with_products": objects_with_products
            }
        except Exception as e:
            logger.error(f"Supabase query failed: {e}")
            return {
                "total_scenes": 0,
                "approved_scenes": 0,
                "total_objects": 0,
                "approved_objects": 0,
                "avg_confidence": 0.0,
                "objects_with_products": 0
            }
    
    async def test_connection(self) -> Dict[str, Any]:
        """Test Supabase connection and permissions"""
        try:
            # Test reading from scenes table
            result = self.supabase.table("scenes").select("scene_id").limit(1).execute()
            
            # Test inserting a test record
            test_scene = {
                "houzz_id": "test_connection_123",
                "image_url": "https://example.com/test.jpg",
                "room_type": "test",
                "status": "scraped"
            }
            
            insert_result = self.supabase.table("scenes").insert(test_scene).execute()
            
            # Clean up test record
            if insert_result.data:
                test_id = insert_result.data[0]["scene_id"]
                self.supabase.table("scenes").delete().eq("scene_id", test_id).execute()
            
            return {
                "status": "success", 
                "message": "Supabase connection and permissions working",
                "can_read": len(result.data) >= 0,
                "can_insert": len(insert_result.data) > 0
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Supabase test failed: {str(e)}"
            }