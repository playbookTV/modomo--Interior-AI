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
            # Map job types to valid database values
            valid_job_types = {
                "processing": "detection",
                "scene_reclassification": "detection", 
                "object_detection": "detection",
                "scene_scraping": "scenes",
                "product_scraping": "products",
                "import": "detection",  # HuggingFace dataset imports
                "color_extraction": "detection",
                "classification": "detection",
                "export": "export"
            }
            
            # Use mapped job type or fallback to 'detection'
            db_job_type = valid_job_types.get(job_type, "detection")
            
            # Handle compound job IDs - only use the first part for database
            # This handles cases like "parent_job_id_detection_child_job_id"
            db_job_id = job_id.split('_detection_')[0].split('_classification_')[0].split('_processing_')[0]
            
            # Validate that it's a proper UUID format
            import uuid
            try:
                uuid.UUID(db_job_id)
            except ValueError:
                # If it's not a valid UUID, generate a new one but log the original
                logger.warning(f"Invalid UUID format for job {job_id}, using base ID {db_job_id}")
                # Use the original job_id as a string identifier in parameters
                parameters = {**parameters, "original_job_id": job_id}
            
            job_data = {
                "job_id": db_job_id,
                "job_type": db_job_type,
                "status": "pending",
                "total_items": total_items,
                "processed_items": 0,
                "progress": 0,
                "parameters": parameters,
                "started_at": datetime.utcnow().isoformat()
            }
            
            result = self.supabase.table("scraping_jobs").insert(job_data).execute()
            
            if result.data:
                logger.info(f"✅ Created job {db_job_id} in database (original: {job_id})")
                return True
            else:
                logger.error(f"❌ Failed to create job {db_job_id} in database")
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
            # Handle compound job IDs - only use the first part for database
            db_job_id = job_id.split('_detection_')[0].split('_classification_')[0].split('_processing_')[0]
            
            # Ensure integer values are properly cast
            processed_items = int(float(processed_items)) if processed_items is not None else 0
            total_items = int(float(total_items)) if total_items is not None else 1
            progress = int((processed_items / total_items * 100)) if total_items > 0 else 0
            
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
            
            result = self.supabase.table("scraping_jobs").update(update_data).eq("job_id", db_job_id).execute()
            return bool(result.data)
            
        except Exception as e:
            logger.error(f"❌ Failed to update job progress for {job_id}: {e}")
            return False
    
    async def get_scenes(
        self,
        limit: int = 20,
        offset: int = 0,
        status: Optional[str] = None,
        image_type: Optional[str] = None,
        room_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get scenes with pagination, filtering, and classification metadata"""
        try:
            # Build query with classification fields
            query = self.supabase.table("scenes").select(
                """scene_id, houzz_id, image_url, image_r2_key, room_type, style_tags, color_tags, 
                status, created_at, image_type, is_primary_object, primary_category, metadata"""
            )
            
            # Add filters if provided
            if status:
                query = query.eq("status", status)
            if image_type:
                query = query.eq("image_type", image_type)
            if room_type:
                query = query.eq("room_type", room_type)
            
            # Execute query with pagination
            result = query.order("created_at", desc=True).range(offset, offset + limit - 1).execute()
            
            # Get total count with same filters
            count_query = self.supabase.table("scenes").select("scene_id", count="exact")
            if status:
                count_query = count_query.eq("status", status)
            if image_type:
                count_query = count_query.eq("image_type", image_type)
            if room_type:
                count_query = count_query.eq("room_type", room_type)
            count_result = count_query.execute()
            total = count_result.count if count_result.count else 0
            
            # Enhance scenes with object count and classification metadata
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
                
                # Enhance metadata with classification information
                metadata = scene_dict.get("metadata", {})
                if metadata:
                    # Add computed classification confidence if available
                    if "classification_confidence" in metadata:
                        scene_dict["classification_confidence"] = metadata["classification_confidence"]
                    if "classification_reason" in metadata:
                        scene_dict["classification_reason"] = metadata["classification_reason"]
                    if "detected_room_type" in metadata:
                        scene_dict["detected_room_type"] = metadata["detected_room_type"]
                    if "detected_styles" in metadata:
                        scene_dict["detected_styles"] = metadata["detected_styles"]
                    if "scores" in metadata:
                        scene_dict["scores"] = metadata["scores"]
                
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
    
    async def update_scene(self, scene_id: str, update_data: Dict[str, Any]) -> bool:
        """Update a scene record in the database"""
        try:
            result = self.supabase.table("scenes").update(update_data).eq("scene_id", scene_id).execute()
            
            if result.data:
                logger.info(f"✅ Updated scene {scene_id}")
                return True
            else:
                logger.warning(f"⚠️ No scene found with ID {scene_id}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to update scene {scene_id}: {e}")
            return False
    
    async def create_scene(
        self, 
        houzz_id: str, 
        image_url: str, 
        room_type: str = None,
        caption: str = None,
        image_r2_key: str = None,
        metadata: Dict[str, Any] = None
    ) -> Optional[str]:
        """Create a new scene record in the database"""
        try:
            scene_data = {
                "houzz_id": houzz_id,
                "image_url": image_url,
                "status": "scraped"
            }
            
            if room_type:
                scene_data["room_type"] = room_type
            if caption:
                scene_data["caption"] = caption
            if image_r2_key:
                scene_data["image_r2_key"] = image_r2_key
            if metadata:
                scene_data["metadata"] = metadata
            
            result = self.supabase.table("scenes").insert(scene_data).execute()
            
            if result.data and len(result.data) > 0:
                scene_id = result.data[0]["scene_id"]
                logger.info(f"✅ Created scene {scene_id} for houzz_id {houzz_id}")
                return scene_id
            else:
                logger.error(f"❌ Failed to create scene for houzz_id {houzz_id}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to create scene for houzz_id {houzz_id}: {e}")
            return None