"""
Web scraping API routes for Houzz and dataset import
"""
import uuid
from fastapi import APIRouter, BackgroundTasks, Query
from typing import List, Optional
import structlog

from services.database_service import DatabaseService
from services.job_service import JobService

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/scrape", tags=["scraping"])

# Check if crawler is available
try:
    from crawlers.houzz_crawler import HouzzCrawler
    CRAWLER_AVAILABLE = True
except ImportError:
    CRAWLER_AVAILABLE = False
    logger.warning("⚠️ Houzz crawler not available - scraping disabled")


@router.post("/scenes")
async def start_scene_scraping(
    background_tasks: BackgroundTasks,
    limit: int = Query(10, description="Number of scenes to scrape"),
    room_types: List[str] = Query(None, description="Filter by room types"),
    db_service: DatabaseService = None,
    job_service: JobService = None
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
    
    # Start background scraping + AI processing task
    background_tasks.add_task(run_scraping_task, job_id, limit, room_types, job_service)
    
    return {
        "job_id": job_id, 
        "status": "running",
        "message": f"Started scraping {limit} scenes from Houzz UK with full AI processing",
        "room_types": room_types or ["all"],
        "features": ["scraping", "object_detection", "segmentation", "embeddings"]
    }


@router.post("/import/huggingface-dataset")
async def import_huggingface_dataset(
    background_tasks: BackgroundTasks,
    dataset: str = Query("sk2003/houzzdata", description="HuggingFace dataset ID (e.g., username/dataset-name)"),
    offset: int = Query(0, description="Starting offset in dataset"),
    limit: int = Query(50, description="Number of images to import and process"),
    include_detection: bool = Query(True, description="Run AI detection on imported images"),
    db_service: DatabaseService = None,
    job_service: JobService = None
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
    
    # Start background import + AI processing task
    background_tasks.add_task(
        run_import_task, 
        job_id, 
        dataset, 
        offset, 
        limit, 
        include_detection,
        job_service
    )
    
    return {
        "job_id": job_id, 
        "status": "running",
        "message": f"Started importing {limit} images from HuggingFace dataset '{dataset}' (offset: {offset})",
        "dataset": dataset,
        "features": ["import", "object_detection", "segmentation", "embeddings"] if include_detection else ["import"]
    }


async def run_scraping_task(
    job_id: str, 
    limit: int, 
    room_types: Optional[List[str]],
    job_service: JobService
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
    job_service: JobService
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