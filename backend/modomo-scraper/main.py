"""
Modomo Dataset Scraping & Tagging System
Main FastAPI application for scene scraping, object detection, and dataset creation
"""

import asyncio
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
import uuid
import json

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import structlog

# Database and storage
import asyncpg
from boto3 import client as boto3_client
import redis

# AI models
from models.grounding_dino import GroundingDINODetector
from models.sam2_segmenter import SAM2Segmenter
from models.clip_embedder import CLIPEmbedder

# Crawling
from crawlers.houzz_crawler import HouzzCrawler
from crawlers.catalog_parser import CatalogParser

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.JSONRenderer()
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Initialize FastAPI app
app = FastAPI(
    title="Modomo Scraper API",
    description="Dataset creation system for interior design AI training",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
db_pool = None
r2_client = None
redis_client = None
grounding_dino = None
sam2_segmenter = None
clip_embedder = None
houzz_crawler = None
catalog_parser = None

# Configuration
MODOMO_TAXONOMY = {
    "seating": ["sofa", "sectional", "armchair", "dining_chair", "stool", "bench"],
    "tables": ["coffee_table", "side_table", "dining_table", "console_table", "desk"],
    "storage": ["bookshelf", "cabinet", "dresser", "wardrobe"],
    "lighting": ["pendant_light", "floor_lamp", "table_lamp", "wall_sconce"],
    "soft_furnishings": ["rug", "curtains", "pillow", "blanket"],
    "decor": ["wall_art", "mirror", "plant", "decorative_object"],
    "bed_bath": ["bed_frame", "mattress", "headboard", "nightstand", "bathtub", "sink_vanity"]
}

# Pydantic models
class SceneMetadata(BaseModel):
    houzz_id: str
    image_url: str
    room_type: Optional[str] = None
    style_tags: List[str] = Field(default_factory=list)
    color_tags: List[str] = Field(default_factory=list)
    project_url: Optional[str] = None

class DetectedObject(BaseModel):
    bbox: List[float] = Field(..., description="[x, y, width, height]")
    mask_url: Optional[str] = None
    category: str
    confidence: float
    tags: List[str] = Field(default_factory=list)
    matched_product_id: Optional[str] = None
    clip_embedding: Optional[List[float]] = None

class ReviewUpdate(BaseModel):
    object_id: str
    category: Optional[str] = None
    tags: Optional[List[str]] = None
    approved: Optional[bool] = None
    matched_product_id: Optional[str] = None

# Startup/shutdown events
@app.on_event("startup")
async def startup():
    global db_pool, r2_client, redis_client, grounding_dino, sam2_segmenter, clip_embedder
    global houzz_crawler, catalog_parser
    
    # Database connection
    DATABASE_URL = os.getenv("DATABASE_URL")
    db_pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=10)
    
    # R2 storage
    r2_client = boto3_client(
        's3',
        endpoint_url=os.getenv("R2_ENDPOINT"),
        aws_access_key_id=os.getenv("R2_ACCESS_KEY"),
        aws_secret_access_key=os.getenv("R2_SECRET_KEY"),
        region_name='auto'
    )
    
    # Redis
    redis_client = redis.from_url(os.getenv("REDIS_URL"))
    
    # AI models
    grounding_dino = GroundingDINODetector()
    sam2_segmenter = SAM2Segmenter()
    clip_embedder = CLIPEmbedder()
    
    # Crawlers
    houzz_crawler = HouzzCrawler()
    catalog_parser = CatalogParser()
    
    logger.info("Modomo scraper started successfully")

@app.on_event("shutdown")
async def shutdown():
    if db_pool:
        await db_pool.close()
    if redis_client:
        redis_client.close()
    logger.info("Modomo scraper shut down")

# Health check
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow()}

# Scene scraping endpoints
@app.post("/scrape/scenes")
async def start_scene_scraping(
    background_tasks: BackgroundTasks,
    limit: int = Query(100, description="Number of scenes to scrape"),
    room_types: List[str] = Query(None, description="Filter by room types")
):
    """Start scraping scenes from Houzz"""
    job_id = str(uuid.uuid4())
    
    background_tasks.add_task(
        scrape_scenes_task,
        job_id=job_id,
        limit=limit,
        room_types=room_types
    )
    
    return {"job_id": job_id, "status": "started"}

@app.get("/scrape/scenes/{job_id}/status")
async def get_scraping_status(job_id: str):
    """Get status of a scraping job"""
    status = redis_client.get(f"job:{job_id}")
    if not status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return json.loads(status)

# Detection endpoints
@app.post("/detect/process")
async def process_detection(
    background_tasks: BackgroundTasks,
    scene_ids: List[str] = Body(..., description="Scene IDs to process")
):
    """Start object detection on scenes"""
    job_id = str(uuid.uuid4())
    
    background_tasks.add_task(
        detection_task,
        job_id=job_id,
        scene_ids=scene_ids
    )
    
    return {"job_id": job_id, "status": "started"}

# Review endpoints
@app.get("/review/queue")
async def get_review_queue(
    limit: int = Query(10, description="Number of scenes to return"),
    room_type: Optional[str] = Query(None),
    category: Optional[str] = Query(None)
):
    """Get scenes pending review"""
    async with db_pool.acquire() as conn:
        query = """
        SELECT s.scene_id, s.image_url, s.room_type, s.style_tags,
               array_agg(
                   json_build_object(
                       'object_id', o.object_id,
                       'bbox', o.bbox,
                       'mask_url', o.mask_url,
                       'category', o.category,
                       'confidence', o.confidence,
                       'tags', o.tags
                   )
               ) as objects
        FROM scenes s
        JOIN detected_objects o ON s.scene_id = o.scene_id
        WHERE s.status = 'pending_review'
        """
        
        params = []
        if room_type:
            query += " AND s.room_type = $1"
            params.append(room_type)
        if category:
            query += f" AND o.category = ${len(params) + 1}"
            params.append(category)
        
        query += " GROUP BY s.scene_id, s.image_url, s.room_type, s.style_tags"
        query += f" LIMIT ${len(params) + 1}"
        params.append(limit)
        
        rows = await conn.fetch(query, *params)
        return [dict(row) for row in rows]

@app.post("/review/update")
async def update_review(updates: List[ReviewUpdate]):
    """Update object reviews"""
    async with db_pool.acquire() as conn:
        for update in updates:
            set_clauses = []
            params = [update.object_id]
            param_idx = 2
            
            if update.category is not None:
                set_clauses.append(f"category = ${param_idx}")
                params.append(update.category)
                param_idx += 1
            
            if update.tags is not None:
                set_clauses.append(f"tags = ${param_idx}")
                params.append(update.tags)
                param_idx += 1
            
            if update.approved is not None:
                set_clauses.append(f"approved = ${param_idx}")
                params.append(update.approved)
                param_idx += 1
            
            if update.matched_product_id is not None:
                set_clauses.append(f"matched_product_id = ${param_idx}")
                params.append(update.matched_product_id)
                param_idx += 1
            
            if set_clauses:
                query = f"""
                UPDATE detected_objects 
                SET {', '.join(set_clauses)}, updated_at = NOW()
                WHERE object_id = $1
                """
                await conn.execute(query, *params)
    
    return {"status": "updated", "count": len(updates)}

@app.post("/review/approve/{scene_id}")
async def approve_scene(scene_id: str):
    """Mark scene as reviewed and approved"""
    async with db_pool.acquire() as conn:
        await conn.execute(
            "UPDATE scenes SET status = 'approved', reviewed_at = NOW() WHERE scene_id = $1",
            scene_id
        )
    return {"status": "approved", "scene_id": scene_id}

# Export endpoints
@app.post("/export/dataset")
async def export_dataset(
    background_tasks: BackgroundTasks,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1
):
    """Export approved dataset for training"""
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.01:
        raise HTTPException(status_code=400, detail="Split ratios must sum to 1.0")
    
    export_id = str(uuid.uuid4())
    
    background_tasks.add_task(
        export_dataset_task,
        export_id=export_id,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio
    )
    
    return {"export_id": export_id, "status": "started"}

# Background tasks
async def scrape_scenes_task(job_id: str, limit: int, room_types: Optional[List[str]]):
    """Background task to scrape scenes from Houzz"""
    try:
        redis_client.set(f"job:{job_id}", json.dumps({"status": "running", "progress": 0}))
        
        scenes = await houzz_crawler.scrape_scenes(limit=limit, room_types=room_types)
        
        async with db_pool.acquire() as conn:
            for i, scene in enumerate(scenes):
                scene_id = str(uuid.uuid4())
                
                # Store scene metadata
                await conn.execute("""
                    INSERT INTO scenes (scene_id, houzz_id, image_url, room_type, style_tags, color_tags, project_url, status)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, 'scraped')
                """, scene_id, scene.houzz_id, scene.image_url, scene.room_type, 
                    scene.style_tags, scene.color_tags, scene.project_url)
                
                # Update progress
                progress = int((i + 1) / len(scenes) * 100)
                redis_client.set(f"job:{job_id}", json.dumps({"status": "running", "progress": progress}))
        
        redis_client.set(f"job:{job_id}", json.dumps({"status": "completed", "scenes_count": len(scenes)}))
        
    except Exception as e:
        logger.error("Scene scraping failed", error=str(e))
        redis_client.set(f"job:{job_id}", json.dumps({"status": "failed", "error": str(e)}))

async def detection_task(job_id: str, scene_ids: List[str]):
    """Background task for object detection"""
    try:
        redis_client.set(f"job:{job_id}", json.dumps({"status": "running", "progress": 0}))
        
        async with db_pool.acquire() as conn:
            for i, scene_id in enumerate(scene_ids):
                # Get scene data
                scene = await conn.fetchrow("SELECT * FROM scenes WHERE scene_id = $1", scene_id)
                if not scene:
                    continue
                
                # Download image
                image_path = await houzz_crawler.download_image(scene['image_url'])
                
                # Run detection
                detections = await grounding_dino.detect_objects(image_path, MODOMO_TAXONOMY)
                
                for detection in detections:
                    # Generate mask with SAM2
                    mask_path = await sam2_segmenter.segment(image_path, detection['bbox'])
                    
                    # Upload mask to R2
                    mask_key = f"masks/{scene_id}/{uuid.uuid4()}.png"
                    r2_client.upload_file(mask_path, "modomo-dataset", mask_key)
                    mask_url = f"https://r2.domain.com/modomo-dataset/{mask_key}"
                    
                    # Generate CLIP embedding
                    embedding = await clip_embedder.embed_object(image_path, detection['bbox'])
                    
                    # Store detection
                    object_id = str(uuid.uuid4())
                    await conn.execute("""
                        INSERT INTO detected_objects (object_id, scene_id, bbox, mask_url, category, confidence, clip_embedding)
                        VALUES ($1, $2, $3, $4, $5, $6, $7)
                    """, object_id, scene_id, detection['bbox'], mask_url, 
                        detection['category'], detection['confidence'], embedding)
                
                # Update scene status
                await conn.execute(
                    "UPDATE scenes SET status = 'pending_review' WHERE scene_id = $1", scene_id
                )
                
                progress = int((i + 1) / len(scene_ids) * 100)
                redis_client.set(f"job:{job_id}", json.dumps({"status": "running", "progress": progress}))
        
        redis_client.set(f"job:{job_id}", json.dumps({"status": "completed"}))
        
    except Exception as e:
        logger.error("Detection task failed", error=str(e))
        redis_client.set(f"job:{job_id}", json.dumps({"status": "failed", "error": str(e)}))

async def export_dataset_task(export_id: str, train_ratio: float, val_ratio: float, test_ratio: float):
    """Background task to export dataset"""
    try:
        redis_client.set(f"export:{export_id}", json.dumps({"status": "running", "progress": 0}))
        
        async with db_pool.acquire() as conn:
            # Get all approved scenes with objects
            scenes = await conn.fetch("""
                SELECT s.*, array_agg(
                    json_build_object(
                        'object_id', o.object_id,
                        'bbox', o.bbox,
                        'mask_url', o.mask_url,
                        'category', o.category,
                        'confidence', o.confidence,
                        'tags', o.tags,
                        'matched_product_id', o.matched_product_id
                    )
                ) as objects
                FROM scenes s
                JOIN detected_objects o ON s.scene_id = o.scene_id
                WHERE s.status = 'approved' AND o.approved = true
                GROUP BY s.scene_id
            """)
            
            # Split dataset
            import random
            random.shuffle(scenes)
            
            total_scenes = len(scenes)
            train_end = int(total_scenes * train_ratio)
            val_end = train_end + int(total_scenes * val_ratio)
            
            splits = {
                'train': scenes[:train_end],
                'val': scenes[train_end:val_end],
                'test': scenes[val_end:]
            }
            
            # Export each split
            for split_name, split_scenes in splits.items():
                manifest = []
                
                for scene in split_scenes:
                    manifest.append({
                        'image_id': scene['scene_id'],
                        'image_url': scene['image_url'],
                        'objects': scene['objects'],
                        'source': 'houzz',
                        'room_type': scene['room_type'],
                        'style_tags': scene['style_tags'],
                        'license': 'per houzz ToS'
                    })
                
                # Upload manifest
                manifest_key = f"exports/{export_id}/{split_name}_manifest.json"
                r2_client.put_object(
                    Bucket="modomo-dataset",
                    Key=manifest_key,
                    Body=json.dumps(manifest, indent=2),
                    ContentType="application/json"
                )
        
        redis_client.set(f"export:{export_id}", json.dumps({
            "status": "completed", 
            "manifest_urls": {
                split: f"https://r2.domain.com/modomo-dataset/exports/{export_id}/{split}_manifest.json"
                for split in ['train', 'val', 'test']
            }
        }))
        
    except Exception as e:
        logger.error("Export task failed", error=str(e))
        redis_client.set(f"export:{export_id}", json.dumps({"status": "failed", "error": str(e)}))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)