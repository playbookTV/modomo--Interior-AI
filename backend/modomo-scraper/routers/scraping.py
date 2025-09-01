"""
Web scraping API routes for Houzz and dataset import
"""
import uuid
from fastapi import APIRouter, BackgroundTasks, Query, Depends
from typing import List, Optional
import structlog

from core.dependencies import get_database_service, get_job_service

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/scrape", tags=["scraping"])

# Check if crawler is available
try:
    from crawlers.houzz_crawler import HouzzCrawler
    CRAWLER_AVAILABLE = True
except ImportError:
    CRAWLER_AVAILABLE = False
    logger.warning("⚠️ Houzz crawler not available - scraping disabled")


@router.post("/scenes", response_model=None)
async def start_scene_scraping(
    background_tasks: BackgroundTasks,
    limit: int = Query(10, description="Number of scenes to scrape"),
    room_types: List[str] = Query(None, description="Filter by room types"),
    db_service = Depends(get_database_service),
    job_service = Depends(get_job_service)
):
    """Start scraping scenes from Houzz UK with full AI processing"""
    if not CRAWLER_AVAILABLE:
        return {"error": "Houzz crawler not available"}
    
    job_id = str(uuid.uuid4())
    
    # Create job in database for persistent tracking
    if db_service:
        await db_service.create_job_in_database(
            job_id=job_id,
            job_type="scenes",
            total_items=limit,
            parameters={
                "limit": limit,
                "room_types": room_types or []
            }
        )
    
    # Create job in Redis for real-time tracking
    if job_service:
        job_service.create_job(
            job_id=job_id,
            job_type="scraping",
            total=limit,
            message=f"Starting Houzz scraping for {limit} scenes"
        )
    
    # Hybrid Processing: Celery Scraping → Railway AI Detection
    try:
        # Step 1: Queue SCRAPING ONLY to Celery (Heroku)
        from tasks.scraping_tasks import run_scraping_job
        
        run_scraping_job.apply_async(
            args=[job_id, limit, room_types or []],
            queue="scraping"
        )
        logger.info(f"✅ Queued SCRAPING-ONLY job {job_id} to Celery")
        
        # Step 2: Schedule AI detection on Railway for scraped scenes
        try:
            from tasks.hybrid_processing import schedule_ai_detection_for_scraping
            
            background_tasks.add_task(
                schedule_ai_detection_for_scraping,
                job_id,
                limit,
                room_types or []
            )
            
            logger.info(f"✅ Scheduled AI detection on Railway for scraping job {job_id}")
            processing_mode = "hybrid_celery_railway_scraping"
            
        except ImportError:
            logger.warning("⚠️ AI detection scheduling not available for scraping")
            processing_mode = "celery_scraping_only"
        
    except Exception as celery_error:
        logger.warning(f"⚠️ Celery not available, running scraping locally: {celery_error}")
        
        # Fallback: Run scraping + AI detection on Railway
        try:
            from tasks.hybrid_processing import process_scene_scraping_hybrid
            
            background_tasks.add_task(
                process_scene_scraping_hybrid,
                job_id,
                limit,
                room_types or []
            )
            
            logger.info(f"✅ Started FULL scraping pipeline on Railway (Celery unavailable)")
            processing_mode = "railway_scraping_pipeline"
            
        except ImportError:
            logger.error("❌ Hybrid scraping processing not available")
            processing_mode = "error_no_scraping"
            raise HTTPException(status_code=503, detail="No scraping backend available")
    
    return {
        "job_id": job_id, 
        "status": "running",
        "message": f"Started {processing_mode}: {limit} scenes from Houzz UK",
        "processing_mode": processing_mode,
        "pipeline": {
            "scraping": "Celery (Heroku)" if processing_mode.startswith("hybrid") else "Railway",
            "ai_detection": "Railway",
            "expected_jobs": 2 if processing_mode.startswith("hybrid") else 1
        },
        "room_types": room_types or ["all"],
        "features": ["scraping", "object_detection", "segmentation", "embeddings"],
        "monitor_urls": {
            "scraping_job": f"/jobs/{job_id}/status",
            "detection_job": f"/jobs/{job_id}_detection/status"
        }
    }


@router.get("/scenes/{job_id}/status", response_model=None)
async def get_scraping_status(
    job_id: str,
    job_service = Depends(get_job_service),
    db_service = Depends(get_database_service)
):
    """Get the status of a scraping job"""
    
    # First try Redis (active jobs)
    if job_service and job_service.is_available():
        job_data = job_service.get_job(job_id)
        if job_data:
            return job_data
    
    # Fallback to database (historical jobs)
    if db_service and db_service.supabase:
        try:
            # Handle compound job IDs - only use the first part for database lookup
            db_job_id = job_id.split('-')[0] if '-' in job_id else job_id
            
            result = db_service.supabase.table("scraping_jobs").select("*").eq("job_id", db_job_id).execute()
            
            if result.data:
                db_job = result.data[0]
                # Convert database format to match Redis format
                return {
                    "job_id": job_id,  # Return original job_id
                    "status": db_job.get("status", "unknown"),
                    "job_type": db_job.get("job_type", "scraping"),
                    "message": db_job.get("error_message") or f"Scraping job {db_job.get('status', 'completed')}",
                    "total": str(db_job.get("total_items", 1)),
                    "processed": str(db_job.get("processed_items", 0)),
                    "progress": str(int((db_job.get("processed_items", 0) / max(db_job.get("total_items", 1), 1)) * 100)),
                    "created_at": db_job.get("created_at", ""),
                    "updated_at": db_job.get("updated_at", ""),
                    "source": "database"
                }
        except Exception as e:
            logger.error(f"Failed to fetch scraping job from database: {e}")
    
    # Job not found in either Redis or database
    from fastapi import HTTPException
    raise HTTPException(status_code=404, detail=f"Scraping job {job_id} not found")


@router.post("/import/huggingface-dataset", response_model=None)
async def import_huggingface_dataset(
    background_tasks: BackgroundTasks,
    dataset: str = Query("sk2003/houzzdata", description="HuggingFace dataset ID (e.g., username/dataset-name)"),
    offset: int = Query(0, description="Starting offset in dataset"),
    limit: int = Query(50, description="Number of images to import and process"),
    include_detection: bool = Query(True, description="Run AI detection on imported images"),
    db_service = Depends(get_database_service),
    job_service = Depends(get_job_service)
):
    """Import any HuggingFace dataset and process with AI"""
    job_id = str(uuid.uuid4())
    
    # Create job in database for persistent tracking
    if db_service:
        await db_service.create_job_in_database(
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
    
    # Create job in Redis for real-time tracking
    if job_service:
        job_service.create_job(
            job_id=job_id,
            job_type="import",
            total=limit,
            message=f"Importing from HuggingFace dataset: {dataset}"
        )
    
    # Hybrid Processing: Celery Import → Railway AI Detection
    try:
        # Step 1: Queue IMPORT ONLY to Celery (Heroku)
        from tasks.scraping_tasks import import_huggingface_dataset as import_task
        
        import_task.apply_async(
            args=[job_id, dataset, offset, limit, False],  # include_detection=False for Celery
            queue="import"
        )
        logger.info(f"✅ Queued IMPORT-ONLY job {job_id} to Celery")
        
        # Step 2: If include_detection=True, schedule AI detection on Railway
        if include_detection:
            try:
                from tasks.hybrid_processing import schedule_ai_detection_for_import
                
                # Schedule AI detection to run after import completes
                background_tasks.add_task(
                    schedule_ai_detection_for_import, 
                    job_id, 
                    dataset, 
                    offset, 
                    limit
                )
                
                logger.info(f"✅ Scheduled AI detection on Railway for job {job_id}")
                
            except ImportError:
                logger.warning("⚠️ AI detection scheduling not available")
        
        processing_mode = "hybrid_celery_railway" if include_detection else "celery_import_only"
        
    except Exception as celery_error:
        logger.warning(f"⚠️ Celery not available, running full pipeline locally: {celery_error}")
        
        # Fallback: Run full import + AI detection on Railway
        try:
            from tasks.hybrid_processing import process_import_with_ai_sync
            
            background_tasks.add_task(
                process_import_with_ai_sync,
                job_id,
                dataset, 
                offset, 
                limit, 
                include_detection
            )
            
            logger.info(f"✅ Started FULL pipeline processing on Railway (Celery unavailable)")
            processing_mode = "railway_full_pipeline"
            
        except ImportError:
            logger.error("❌ Hybrid processing module not available")
            processing_mode = "error_no_processing"
            raise HTTPException(status_code=503, detail="No processing backend available")
    
    return {
        "job_id": job_id, 
        "status": "running",
        "message": f"Started {processing_mode}: {limit} images from '{dataset}' (offset: {offset})",
        "dataset": dataset,
        "processing_mode": processing_mode,
        "pipeline": {
            "import": "Celery (Heroku)" if processing_mode.startswith("hybrid") else "Railway",
            "ai_detection": "Railway" if include_detection else "disabled",
            "expected_jobs": 2 if (include_detection and processing_mode.startswith("hybrid")) else 1
        },
        "features": ["import", "object_detection", "segmentation", "embeddings"] if include_detection else ["import"],
        "monitor_urls": {
            "import_job": f"/jobs/{job_id}/status",
            "detection_job": f"/jobs/{job_id}_detection/status" if include_detection else None
        }
    }


async def run_scraping_task(
    job_id: str, 
    limit: int, 
    room_types: Optional[List[str]],
    job_service
):
    """Background task for Houzz scraping"""
    try:
        logger.info(f"🕷️ Starting Houzz scraping job {job_id}")
        
        if job_service:
            job_service.update_job(job_id, status="running", message="Initializing Houzz crawler")
        
        # This would contain the actual scraping logic
        # For now, just simulate the process
        for i in range(limit):
            if job_service:
                job_service.update_job(
                    job_id,
                    processed=i + 1,
                    total=limit,
                    message=f"Scraped {i + 1}/{limit} scenes"
                )
            
            # Simulate work
            import asyncio
            await asyncio.sleep(0.1)
        
        logger.info(f"✅ Houzz scraping job {job_id} completed")
        
        if job_service:
            job_service.complete_job(job_id, f"Scraping complete: {limit} scenes processed")
        
    except Exception as e:
        logger.error(f"❌ Houzz scraping job {job_id} failed: {e}")
        if job_service:
            job_service.fail_job(job_id, str(e))


async def run_import_task(
    job_id: str,
    dataset: str,
    offset: int,
    limit: int,
    include_detection: bool,
    job_service
):
    """Background task for HuggingFace dataset import"""
    try:
        logger.info(f"📥 Starting dataset import job {job_id} from {dataset}")
        
        if job_service:
            job_service.update_job(job_id, status="running", message=f"Loading dataset: {dataset}")
        
        # This would contain the actual import logic
        # For now, just simulate the process
        for i in range(limit):
            if job_service:
                job_service.update_job(
                    job_id,
                    processed=i + 1,
                    total=limit,
                    message=f"Imported {i + 1}/{limit} images from {dataset}"
                )
            
            # Simulate work
            import asyncio
            await asyncio.sleep(0.1)
        
        logger.info(f"✅ Dataset import job {job_id} completed")
        
        if job_service:
            job_service.complete_job(job_id, f"Import complete: {limit} images from {dataset}")
        
    except Exception as e:
        logger.error(f"❌ Dataset import job {job_id} failed: {e}")
        if job_service:
            job_service.fail_job(job_id, str(e))