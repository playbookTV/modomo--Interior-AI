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
# Import PIL with fallback for environments without it
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("PIL not available - PIL image processing will be disabled")
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


# Helper functions
def extract_image_url_from_item_sync(item: Dict[str, Any], r2_uploader) -> Optional[str]:
    """
    Synchronous version of extract_image_url_from_item with PIL image handling
    """
    try:
        # Check if item has a PIL image
        if "image" in item and PIL_AVAILABLE:
            pil_image = item["image"]
            if hasattr(pil_image, "save"):  # Check if it's a PIL Image
                logger.info("Found PIL Image in dataset item, uploading to R2...")
                # Upload PIL image to R2 synchronously
                image_url = upload_pil_image_to_r2_sync(pil_image, r2_uploader)
                if image_url:
                    logger.info(f"Successfully uploaded PIL image to R2: {image_url}")
                    return image_url
                else:
                    logger.warning("Failed to upload PIL image to R2")
        
        # Fallback to URL extraction (existing logic)
        if "image_url" in item:
            return item["image_url"]
        elif "url" in item:
            return item["url"]
        elif "image" in item and isinstance(item["image"], str):
            return item["image"]
        
        logger.warning(f"No image URL or PIL Image found in item: {list(item.keys())}")
        return None
        
    except Exception as e:
        logger.error(f"Error extracting image from dataset item: {e}")
        return None


def upload_pil_image_to_r2_sync(pil_image, r2_uploader) -> Optional[str]:
    """
    Synchronous PIL image upload to R2 storage
    """
    if not PIL_AVAILABLE:
        logger.warning("PIL not available - cannot upload PIL image")
        return None
        
    try:
        import io
        import uuid
        
        # Convert PIL image to bytes
        img_buffer = io.BytesIO()
        
        # Handle different image formats
        image_format = "JPEG"
        if hasattr(pil_image, 'format') and pil_image.format:
            image_format = pil_image.format
        elif pil_image.mode == "RGBA":
            image_format = "PNG"
        
        # Save to buffer
        if image_format == "JPEG" and pil_image.mode in ("RGBA", "LA", "P"):
            # Convert to RGB for JPEG
            pil_image = pil_image.convert("RGB")
        
        pil_image.save(img_buffer, format=image_format, quality=85)
        img_buffer.seek(0)
        
        # Generate unique filename
        file_extension = "jpg" if image_format == "JPEG" else image_format.lower()
        filename = f"hf_import_{uuid.uuid4().hex}.{file_extension}"
        
        # Upload to R2
        from services.r2_uploader import upload_to_r2_sync
        image_url = upload_to_r2_sync(
            img_buffer.getvalue(),
            filename,
            f"image/{file_extension}",
            r2_uploader
        )
        
        return image_url
        
    except Exception as e:
        logger.error(f"Error uploading PIL image to R2: {e}")
        return None


def extract_metadata_from_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """Extract metadata from HuggingFace dataset item"""
    metadata = {}
    
    # Common metadata fields
    for field in ["caption", "description", "title", "tags", "category", "room_type", "style"]:
        if field in item:
            metadata[field] = item[field]
    
    # Additional fields
    if "width" in item and "height" in item:
        metadata["dimensions"] = {"width": item["width"], "height": item["height"]}
    
    return metadata


def load_huggingface_dataset(dataset_name: str, offset: int, limit: int):
    """Load HuggingFace dataset with offset and limit"""
    try:
        from datasets import load_dataset
        
        logger.info(f"Loading HuggingFace dataset: {dataset_name}")
        dataset = load_dataset(dataset_name, streaming=True)
        
        # Get the train split or first available split
        if "train" in dataset:
            data = dataset["train"]
        else:
            data = dataset[list(dataset.keys())[0]]
        
        # Apply offset and limit
        items = []
        current_idx = 0
        
        for item in data:
            if current_idx < offset:
                current_idx += 1
                continue
            
            if len(items) >= limit:
                break
                
            items.append(item)
            current_idx += 1
        
        logger.info(f"Loaded {len(items)} items from dataset {dataset_name}")
        return items
        
    except Exception as e:
        logger.error(f"Error loading HuggingFace dataset {dataset_name}: {e}")
        return None


def store_scene_in_database(scene_data: Dict[str, Any]) -> Optional[str]:
    """Store scene in database"""
    try:
        # Use database service to store scene
        scene_id = database_service.create_scene(scene_data)
        return scene_id
    except Exception as e:
        logger.error(f"Error storing scene in database: {e}")
        return None


def get_houzz_crawler():
    """Get Houzz crawler instance"""
    # Placeholder - would import actual crawler
    logger.warning("Houzz crawler not implemented yet")
    return None


def scrape_houzz_scenes(crawler, limit: int, room_types: Optional[List[str]] = None):
    """Scrape scenes from Houzz"""
    # Placeholder - would use actual crawler
    logger.warning("Houzz scraping not implemented yet") 
    return []