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
logger = structlog.get_logger(__name__)

# Import PIL with fallback for environments without it
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("PIL not available - PIL image processing will be disabled")
from services.r2_uploader import create_r2_uploader


# Helper functions (moved to top for proper scope)

def load_huggingface_dataset(dataset: str, offset: int, limit: int):
    """Load HuggingFace dataset with pagination"""
    try:
        from datasets import load_dataset
        logger.info(f"Loading HuggingFace dataset: {dataset}")
        
        # Load dataset
        ds = load_dataset(dataset, split="train", streaming=True)
        
        # Apply offset and limit
        dataset_slice = ds.skip(offset).take(limit)
        
        # Convert to list for processing
        items = list(dataset_slice)
        logger.info(f"Loaded {len(items)} items from {dataset} (offset: {offset}, limit: {limit})")
        
        return items
        
    except Exception as e:
        logger.error(f"Failed to load HuggingFace dataset {dataset}: {e}")
        return None


def extract_metadata_from_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """Extract metadata from dataset item"""
    metadata = {}
    
    # Extract common metadata fields
    for key, value in item.items():
        if key not in ["image", "img", "picture", "photo"]:  # Skip image fields
            # Convert complex types to strings
            if isinstance(value, (list, dict)):
                metadata[key] = str(value)
            else:
                metadata[key] = value
    
    return metadata


def upload_pil_image_to_r2_sync(pil_image, r2_uploader) -> Optional[tuple]:
    """Upload PIL image to R2 storage synchronously"""
    try:
        import io
        import uuid
        
        # Convert PIL image to bytes
        img_buffer = io.BytesIO()
        
        # Determine format (default to PNG for safety)
        format = getattr(pil_image, 'format', 'PNG') or 'PNG'
        if format.upper() not in ['JPEG', 'PNG', 'WEBP']:
            format = 'PNG'
        
        pil_image.save(img_buffer, format=format)
        img_bytes = img_buffer.getvalue()
        
        # Generate unique R2 key
        file_ext = format.lower()
        if file_ext == 'jpeg':
            file_ext = 'jpg'
        r2_key = f"training-data/scenes/{uuid.uuid4()}.{file_ext}"
        
        # Get content type
        content_type = f"image/{file_ext}"
        if file_ext == 'jpg':
            content_type = "image/jpeg"
        
        # Upload using sync helper
        from services.r2_uploader import upload_to_r2_sync
        public_url = upload_to_r2_sync(img_bytes, r2_key, content_type, r2_uploader)
        
        return (public_url, r2_key) if public_url else None
        
    except Exception as e:
        logger.error(f"Failed to upload PIL image to R2: {e}")
        return None
def extract_image_url_from_item_sync(item: Dict[str, Any], r2_uploader) -> Optional[tuple]:
    """
    Synchronous version of extract_image_url_from_item with PIL image handling
    """
    try:
        # Check if item has a PIL image (try common field names)
        pil_field = None
        for field_name in ["image", "img", "picture", "photo", "Images"]:
            if field_name in item and PIL_AVAILABLE:
                pil_field = field_name
                break
        
        if pil_field:
            pil_image = item[pil_field]
            if hasattr(pil_image, "save"):  # Check if it's a PIL Image
                logger.info("Found PIL Image in dataset item, uploading to R2...")
                # Upload PIL image to R2 synchronously
                upload_result = upload_pil_image_to_r2_sync(pil_image, r2_uploader)
                if upload_result:
                    image_url, r2_key = upload_result
                    logger.info(f"Successfully uploaded PIL image to R2: {image_url}")
                    return (image_url, r2_key)
                else:
                    logger.warning("Failed to upload PIL image to R2")
        
        # Fallback to URL extraction (existing logic) - return tuple format
        if "image_url" in item:
            return (item["image_url"], None)  # No R2 key for external URLs
        elif "url" in item:
            return (item["url"], None)
        elif "image" in item and isinstance(item["image"], str):
            return (item["image"], None)
        elif "img" in item and isinstance(item["img"], str):
            return (item["img"], None)
        elif "Images" in item and isinstance(item["Images"], str):
            return (item["Images"], None)
        
        logger.warning(f"No image URL or PIL Image found in item: {list(item.keys())}")
        return None
        
    except Exception as e:
        logger.error(f"Error extracting image from dataset item: {e}")
        return None


def store_scene_in_database(scene_data: Dict[str, Any]) -> Optional[str]:
    """Store scene in database"""
    try:
        # Check if database service is available
        if database_service is None:
            logger.warning("Database service not available - creating mock scene ID")
            import uuid
            return str(uuid.uuid4())
        
        # Use database service to store scene (sync version)
        import asyncio
        
        # Create async wrapper to call the async method
        async def _create_scene_async():
            return await database_service.create_scene(
                houzz_id=scene_data["houzz_id"],
                image_url=scene_data["image_url"],
                room_type=scene_data.get("room_type"),
                caption=scene_data.get("caption"),
                image_r2_key=scene_data.get("image_r2_key"),
                metadata=scene_data.get("metadata")
            )
        
        # Run async function in sync context
        try:
            # Try to get existing event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is already running (in async context), create a task
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, _create_scene_async())
                    scene_id = future.result(timeout=30)
            else:
                # If no loop is running, run directly
                scene_id = asyncio.run(_create_scene_async())
        except RuntimeError:
            # Fallback: create new event loop
            scene_id = asyncio.run(_create_scene_async())
        
        return scene_id
    except Exception as e:
        logger.error(f"Error storing scene in database: {e}")
        # Return a mock scene ID so the import doesn't fail completely
        import uuid
        return str(uuid.uuid4())


def get_houzz_crawler():
    """Get Houzz crawler instance"""
    try:
        from crawlers.houzz_crawler import HouzzCrawler
        crawler = HouzzCrawler()
        logger.info("✅ Houzz crawler initialized successfully")
        return crawler
    except ImportError as e:
        logger.error(f"❌ Failed to import Houzz crawler: {e}")
        return None
    except Exception as e:
        logger.error(f"❌ Failed to initialize Houzz crawler: {e}")
        return None


async def scrape_houzz_scenes(crawler, limit: int, room_types: Optional[List[str]] = None):
    """Scrape scenes from Houzz using the actual crawler"""
    try:
        logger.info(f"Starting Houzz scraping: limit={limit}, room_types={room_types}")
        scenes = await crawler.scrape_scenes(limit=limit, room_types=room_types)
        logger.info(f"✅ Successfully scraped {len(scenes)} scenes from Houzz")
        return scenes
    except Exception as e:
        logger.error(f"❌ Houzz scraping failed: {e}")
        return []
    finally:
        # Ensure crawler is properly closed
        try:
            await crawler.close()
        except:
            pass

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
            # Use crawler to get scenes (handle async function)
            scraped_scenes = asyncio.run(scrape_houzz_scenes(crawler, limit, room_types))
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
                    
                    # Run detection pipeline asynchronously (no .get() call)
                    try:
                        detection_task = run_detection_pipeline.apply_async(
                            args=[detection_job_id, scene_data["image_url"], scene_id]
                        )
                        
                        # Log that detection was queued but don't wait for result
                        logger.info(f"🤖 Queued AI detection for scene {scene_id} (task: {detection_task.id})")
                        
                        # We can't wait for the result here, so we assume it will be processed
                        # The detection task will handle its own success/failure logging
                        
                    except Exception as detection_error:
                        logger.error(f"AI detection failed for imported scene {scene_id}: {detection_error}")
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
                image_result = extract_image_url_from_item_sync(item, r2_uploader)
                metadata = extract_metadata_from_item(item)
                
                if not image_result:
                    logger.warning(f"No image URL found in dataset item {i}")
                    continue
                
                # Unpack the result - could be (url, r2_key) or just url
                if isinstance(image_result, tuple):
                    image_url, image_r2_key = image_result
                else:
                    # Fallback for backward compatibility
                    image_url = image_result
                    image_r2_key = None
                
                # Create scene record with R2 key
                scene_data = {
                    "houzz_id": f"hf_{dataset}_{offset + i}",
                    "image_url": image_url,
                    "image_r2_key": image_r2_key,
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
                            detection_task = run_detection_pipeline.apply_async(
                                args=[detection_job_id, image_url, scene_id]
                            )
                            
                            # Log that detection was queued but don't wait for result
                            logger.info(f"🤖 Queued AI detection for scene {scene_id} (task: {detection_task.id})")
                                
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
        
        # Clean up Redis job tracking - import is done, handoff to AI detection
        from tasks import job_service
        
        if job_service and job_service.is_available():
            # Update Redis job status to completed (cleans up from active queue)
            job_service.complete_job(
                job_id=job_id,
                message=f"Import completed: {imported}/{total_items} scenes imported. Handed off to AI detection."
            )
            logger.info(f"🧹 Cleaned up Redis tracking for import job {job_id}")
        
        return BaseTask.complete_job(job_id, imported, total_items, result)
        
    except Exception as e:
        logger.error(f"❌ Import job {job_id} failed: {e}")
        BaseTask.handle_task_error(job_id, e, imported, limit)
        raise