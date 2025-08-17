"""
Phase 3: Scraping and import Celery tasks
"""
import structlog
import uuid
import asyncio
from typing import Dict, Any, List, Optional
from celery_app import celery_app
from tasks import BaseTask, database_service
from tasks.detection_tasks import run_detection_pipeline
from PIL import Image
from services.r2_uploader import create_r2_uploader

logger = structlog.get_logger(__name__)

@celery_app.task(bind=True, autoretry_for=(Exception,), retry_kwargs={'max_retries': 2, 'countdown': 300})
def run_scraping_job(self, job_id: str, limit: int, room_types: Optional[List[str]] = None):
    """Scrape scenes from Houzz with full AI processing"""
    scraped = 0
    processed = 0
    
    try:
        logger.info(f"🕷️ Starting Houzz scraping job {job_id} for {limit} scenes")
        BaseTask.update_job_progress(job_id, "running", 0, limit, "Initializing Houzz scraper...")
        
        # Check if crawler is available
        crawler = get_houzz_crawler()
        if not crawler:
            raise Exception("Houzz crawler not available")
        
        BaseTask.update_job_progress(job_id, "running", 0, limit, "Starting Houzz scraping...")
        BaseTask.update_celery_progress(0, limit, "Starting scraping...")
        
        # Start scraping
        scraped_scenes = []
        
        try:
            # Use crawler to get scenes
            scraped_scenes = scrape_houzz_scenes(crawler, limit, room_types)
            scraped = len(scraped_scenes)
            
            logger.info(f"Scraped {scraped} scenes from Houzz")
            BaseTask.update_job_progress(job_id, "running", scraped, limit, f"Scraped {scraped} scenes, starting AI processing...")
            
        except Exception as scraping_error:
            logger.error(f"Scraping failed: {scraping_error}")
            raise Exception(f"Houzz scraping failed: {scraping_error}")
        
        # Process each scraped scene with AI
        for i, scene_data in enumerate(scraped_scenes):
            try:
                # Store scene in database
                scene_id = store_scene_in_database(scene_data)
                
                if scene_id:
                    # Run AI detection pipeline
                    detection_job_id = f"{job_id}_detection_{scene_id}"
                    
                    # Run detection pipeline synchronously
                    try:
                        detection_result = run_detection_pipeline.apply(
                            args=[detection_job_id, scene_data["image_url"], scene_id]
                        ).get()
                        
                        if detection_result and detection_result.get("stored_objects", 0) > 0:
                            processed += 1
                            logger.info(f"Successfully processed scene {scene_id} with {detection_result.get('total_detected', 0)} objects")
                        else:
                            logger.warning(f"No objects detected in scene {scene_id}")
                            
                    except Exception as detection_error:
                        logger.error(f"AI detection failed for scene {scene_id}: {detection_error}")
                        # Continue with next scene even if AI processing fails
                
                # Update progress
                current_progress = i + 1
                progress_message = f"Processed {current_progress}/{scraped} scraped scenes"
                BaseTask.update_job_progress(job_id, "running", current_progress, scraped, progress_message)
                BaseTask.update_celery_progress(current_progress, scraped, progress_message)
                
            except Exception as scene_error:
                logger.error(f"Error processing scraped scene: {scene_error}")
                continue
        
        # Complete the job
        result = {
            "scraped_scenes": scraped,
            "processed_scenes": processed,
            "ai_success_rate": f"{(processed/scraped)*100:.1f}%" if scraped > 0 else "0%",
            "room_types": room_types,
            "message": f"Scraping completed: {scraped} scenes scraped, {processed} processed with AI"
        }
        
        logger.info(f"🕷️ Scraping job {job_id} completed: {scraped} scraped, {processed} AI processed")
        return BaseTask.complete_job(job_id, scraped, limit, result)
        
    except Exception as e:
        logger.error(f"❌ Scraping job {job_id} failed: {e}")
        BaseTask.handle_task_error(job_id, e, scraped, limit)
        raise

@celery_app.task(bind=True, autoretry_for=(Exception,), retry_kwargs={'max_retries': 3, 'countdown': 180})
def import_huggingface_dataset(self, job_id: str, dataset: str, offset: int, limit: int, include_detection: bool = True):
    """Import HuggingFace dataset with optional AI processing"""
    imported = 0
    processed = 0
    
    try:
        logger.info(f"📥 Starting HuggingFace import job {job_id}: {dataset} (offset: {offset}, limit: {limit})")
        BaseTask.update_job_progress(job_id, "running", 0, limit, f"Connecting to HuggingFace dataset {dataset}...")
        
        # Initialize R2 uploader (synchronous)
        from services.r2_uploader import create_r2_uploader
        r2_uploader = create_r2_uploader()
        
        # Load dataset
        dataset_data = load_huggingface_dataset(dataset, offset, limit)
        if not dataset_data:
            raise Exception(f"Failed to load HuggingFace dataset: {dataset}")
        
        total_items = len(dataset_data)
        logger.info(f"Loaded {total_items} items from dataset {dataset}")
        
        BaseTask.update_job_progress(job_id, "running", 0, total_items, f"Processing {total_items} dataset items...")
        
        # Process each item
        for i, item in enumerate(dataset_data):
            try:
                # Extract image URL and metadata (sync version with PIL handling)
                image_url = extract_image_url_from_item_sync(item, r2_uploader)
                metadata = extract_metadata_from_item(item)

async def import_huggingface_dataset_async(job_id: str, dataset: str, offset: int, limit: int, include_detection: bool = True):
    """Async implementation of HuggingFace dataset import"""
    imported = 0
    processed = 0
    
    try:
        logger.info(f"📥 Starting HuggingFace import job {job_id}: {dataset} (offset: {offset}, limit: {limit})")
        BaseTask.update_job_progress(job_id, "running", 0, limit, f"Connecting to HuggingFace dataset {dataset}...")
        
        # Initialize R2 uploader
        r2_uploader = create_r2_uploader()
        
        # Load dataset
        dataset_data = load_huggingface_dataset(dataset, offset, limit)
        if not dataset_data:
            raise Exception(f"Failed to load HuggingFace dataset: {dataset}")
        
        total_items = len(dataset_data)
        logger.info(f"Loaded {total_items} items from dataset {dataset}")
        
        BaseTask.update_job_progress(job_id, "running", 0, total_items, f"Processing {total_items} dataset items...")
        
        # Process each item
        for i, item in enumerate(dataset_data):
            try:
                # Extract image URL and metadata (now async with R2 uploader)
                image_url = await extract_image_url_from_item(item, r2_uploader)
                metadata = extract_metadata_from_item(item)
                
                if not image_url:
                    logger.warning(f"No image URL found in dataset item {i}")
                    continue
                
                # Create scene record
                scene_data = {
                    "houzz_id": f"hf_{dataset}_{offset + i}",
                    "image_url": image_url,
                    "metadata": metadata,
                    "status": "imported"
                }
                
                # Store scene
                scene_id = store_scene_in_database(scene_data)
                
                if scene_id:
                    imported += 1
                    
                    # Run AI detection if requested
                    if include_detection:
                        try:
                            detection_job_id = f"{job_id}_detection_{scene_id}"
                            detection_result = run_detection_pipeline.apply(
                                args=[detection_job_id, image_url, scene_id]
                            ).get()
                            
                            if detection_result and detection_result.get("stored_objects", 0) > 0:
                                processed += 1
                                logger.info(f"AI processed scene {scene_id}: {detection_result.get('total_detected', 0)} objects")
                                
                        except Exception as detection_error:
                            logger.error(f"AI detection failed for imported scene {scene_id}: {detection_error}")
                
                # Update progress
                progress_message = f"Imported {imported}/{total_items} items"
                if include_detection:
                    progress_message += f", AI processed {processed}"
                
                BaseTask.update_job_progress(job_id, "running", i + 1, total_items, progress_message)
                BaseTask.update_celery_progress(i + 1, total_items, progress_message)
                
            except Exception as item_error:
                logger.error(f"Error processing dataset item {i}: {item_error}")
                continue
        
        # Complete the job
        result = {
            "dataset": dataset,
            "offset": offset,
            "requested_limit": limit,
            "imported_scenes": imported,
            "ai_processed_scenes": processed if include_detection else 0,
            "include_detection": include_detection,
            "success_rate": f"{(imported/total_items)*100:.1f}%" if total_items > 0 else "0%",
            "message": f"Import completed: {imported}/{total_items} scenes imported"
        }
        
        if include_detection:
            result["ai_success_rate"] = f"{(processed/imported)*100:.1f}%" if imported > 0 else "0%"
        
        logger.info(f"📥 Import job {job_id} completed: {imported} imported, {processed} AI processed")
        return BaseTask.complete_job(job_id, imported, total_items, result)
        
    except Exception as e:
        logger.error(f"❌ Import job {job_id} failed: {e}")
        BaseTask.handle_task_error(job_id, e, imported, limit)
        raise

def get_houzz_crawler():
    """Get initialized Houzz crawler"""
    try:
        from crawlers.houzz_crawler import HouzzCrawler
        return HouzzCrawler()
    except ImportError:
        logger.error("Houzz crawler not available")
        return None

def scrape_houzz_scenes(crawler, limit: int, room_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Scrape scenes from Houzz"""
    try:
        scenes = []
        
        # Configure room types for scraping
        target_room_types = room_types or ["living-room", "bedroom", "kitchen", "bathroom"]
        
        for room_type in target_room_types:
            try:
                # Scrape scenes for this room type
                room_scenes = crawler.scrape_room_scenes(room_type, limit // len(target_room_types))
                scenes.extend(room_scenes)
                
                if len(scenes) >= limit:
                    break
                    
            except Exception as room_error:
                logger.error(f"Failed to scrape {room_type}: {room_error}")
                continue
        
        return scenes[:limit]
        
    except Exception as e:
        logger.error(f"Houzz scraping failed: {e}")
        return []

def store_scene_in_database(scene_data: Dict[str, Any]) -> Optional[str]:
    """Store scene in database and return scene_id"""
    try:
        if not database_service:
            return None
        
        # Prepare scene data for database
        db_scene_data = {
            "houzz_id": scene_data.get("houzz_id"),
            "image_url": scene_data.get("image_url"),
            "room_type": scene_data.get("room_type"),
            "style_tags": scene_data.get("style_tags", []),
            "color_tags": scene_data.get("color_tags", []),
            "project_url": scene_data.get("project_url"),
            "status": scene_data.get("status", "scraped"),
            "metadata": scene_data.get("metadata", {})
        }
        
        # Insert scene
        result = database_service.supabase.table("scenes").insert(db_scene_data).execute()
        
        if result.data:
            scene_id = result.data[0]["scene_id"]
            logger.debug(f"Stored scene {scene_id}")
            return scene_id
        
        return None
        
    except Exception as e:
        logger.error(f"Failed to store scene: {e}")
        return None

def load_huggingface_dataset(dataset: str, offset: int, limit: int) -> List[Dict[str, Any]]:
    """Load data from HuggingFace dataset"""
    try:
        # Import datasets library
        try:
            from datasets import load_dataset
        except ImportError:
            logger.error("HuggingFace datasets library not available")
            return []
        
        # Load dataset
        ds = load_dataset(dataset, split="train", streaming=True)
        
        # Skip to offset and take limit items
        dataset_items = []
        for i, item in enumerate(ds):
            if i < offset:
                continue
            if len(dataset_items) >= limit:
                break
            dataset_items.append(item)
        
        return dataset_items
        
    except Exception as e:
        logger.error(f"Failed to load HuggingFace dataset {dataset}: {e}")
        return []

def upload_pil_image_to_r2_sync(pil_image, r2_uploader) -> Optional[str]:
    """Upload PIL image to R2 storage and return public URL (sync version)"""
    try:
        import io
        import uuid
        from datetime import datetime
        import asyncio
        
        # Convert PIL image to bytes
        image_buffer = io.BytesIO()
        
        # Determine format - default to JPEG if not specified
        image_format = pil_image.format if pil_image.format else 'JPEG'
        
        # Convert RGBA to RGB for JPEG compatibility
        if image_format == 'JPEG' and pil_image.mode in ('RGBA', 'LA'):
            # Create white background
            rgb_image = Image.new('RGB', pil_image.size, (255, 255, 255))
            if pil_image.mode == 'RGBA':
                rgb_image.paste(pil_image, mask=pil_image.split()[-1])  # Use alpha channel as mask
            else:
                rgb_image.paste(pil_image)
            pil_image = rgb_image
        
        # Save image to buffer
        pil_image.save(image_buffer, format=image_format, quality=85)
        image_bytes = image_buffer.getvalue()
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        file_extension = 'jpg' if image_format == 'JPEG' else image_format.lower()
        r2_key = f"huggingface-imports/{timestamp}_{unique_id}.{file_extension}"
        
        # Set content type
        content_type = f"image/{file_extension}"
        
        # Upload to R2 (run async function in sync context)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            success = loop.run_until_complete(r2_uploader.upload_bytes(image_bytes, r2_key, content_type))
        finally:
            loop.close()
        
        if success:
            # Return R2 public URL
            public_url = f"https://pub-d1ea07ac8a9a4b7093ae9e2b17c5b6ad.r2.dev/{r2_key}"
            logger.info(f"✅ Uploaded PIL image to R2: {public_url}")
            return public_url
        else:
            logger.error("❌ Failed to upload PIL image to R2")
            return None
            
    except Exception as e:
        logger.error(f"❌ Error uploading PIL image: {e}")
        return None

async def upload_pil_image_to_r2(pil_image, r2_uploader) -> Optional[str]:
    """Upload PIL image to R2 storage and return public URL"""
    try:
        import io
        import uuid
        from datetime import datetime
        
        # Convert PIL image to bytes
        image_buffer = io.BytesIO()
        
        # Determine format - default to JPEG if not specified
        image_format = pil_image.format if pil_image.format else 'JPEG'
        
        # Convert RGBA to RGB for JPEG compatibility
        if image_format == 'JPEG' and pil_image.mode in ('RGBA', 'LA'):
            # Create white background
            rgb_image = Image.new('RGB', pil_image.size, (255, 255, 255))
            if pil_image.mode == 'RGBA':
                rgb_image.paste(pil_image, mask=pil_image.split()[-1])  # Use alpha channel as mask
            else:
                rgb_image.paste(pil_image)
            pil_image = rgb_image
        
        # Save image to buffer
        pil_image.save(image_buffer, format=image_format, quality=85)
        image_bytes = image_buffer.getvalue()
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        file_extension = 'jpg' if image_format == 'JPEG' else image_format.lower()
        r2_key = f"huggingface-imports/{timestamp}_{unique_id}.{file_extension}"
        
        # Set content type
        content_type = f"image/{file_extension}"
        
        # Upload to R2
        success = await r2_uploader.upload_bytes(image_bytes, r2_key, content_type)
        
        if success:
            # Return R2 public URL
            public_url = f"https://pub-d1ea07ac8a9a4b7093ae9e2b17c5b6ad.r2.dev/{r2_key}"
            logger.info(f"✅ Uploaded PIL image to R2: {public_url}")
            return public_url
        else:
            logger.error("❌ Failed to upload PIL image to R2")
            return None
            
    except Exception as e:
        logger.error(f"❌ Error uploading PIL image: {e}")
        return None

def extract_image_url_from_item_sync(item: Dict[str, Any], r2_uploader=None) -> Optional[str]:
    """Extract image URL from dataset item, uploading PIL images to R2 if needed (sync version)"""
    try:
        # Common field names for images in HF datasets
        image_fields = ["image", "img", "image_url", "url", "src"]
        
        for field in image_fields:
            if field in item:
                value = item[field]
                if isinstance(value, str) and (value.startswith("http") or value.startswith("https")):
                    return value
                elif hasattr(value, "save"):  # PIL Image object
                    # Upload PIL image to R2 storage (sync version)
                    if r2_uploader:
                        return upload_pil_image_to_r2_sync(value, r2_uploader)
                    else:
                        logger.warning("PIL image found but no R2 uploader available")
                        continue
        
        return None
        
    except Exception as e:
        logger.error(f"Failed to extract image URL: {e}")
        return None

async def extract_image_url_from_item(item: Dict[str, Any], r2_uploader=None) -> Optional[str]:
    """Extract image URL from dataset item, uploading PIL images to R2 if needed"""
    try:
        # Common field names for images in HF datasets
        image_fields = ["image", "img", "image_url", "url", "src"]
        
        for field in image_fields:
            if field in item:
                value = item[field]
                if isinstance(value, str) and (value.startswith("http") or value.startswith("https")):
                    return value
                elif hasattr(value, "save"):  # PIL Image object
                    # Upload PIL image to R2 storage
                    if r2_uploader:
                        return await upload_pil_image_to_r2(value, r2_uploader)
                    else:
                        logger.warning("PIL image found but no R2 uploader available")
                        continue
        
        return None
        
    except Exception as e:
        logger.error(f"Failed to extract image URL: {e}")
        return None

def extract_metadata_from_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """Extract metadata from dataset item"""
    try:
        metadata = {}
        
        # Extract common metadata fields
        metadata_fields = ["caption", "description", "tags", "category", "style", "room_type"]
        
        for field in metadata_fields:
            if field in item:
                metadata[field] = item[field]
        
        # Add dataset info
        metadata["source"] = "huggingface"
        metadata["import_timestamp"] = str(uuid.uuid4())
        
        return metadata
        
    except Exception as e:
        logger.error(f"Failed to extract metadata: {e}")
        return {}