"""
Basic Modomo Dataset Scraping System (without AI dependencies)
Start here to test the system before installing heavy ML packages
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
import asyncpg
import redis

# Import Houzz crawler
try:
    from crawlers.houzz_crawler import HouzzCrawler
    CRAWLER_AVAILABLE = True
except ImportError:
    CRAWLER_AVAILABLE = False
    print("⚠️ Houzz crawler not available - scraping will use dummy data")

# Dummy AI classes for basic testing
class DummyDetector:
    async def detect_objects(self, image_path: str, taxonomy: dict) -> List[dict]:
        # Return dummy detection for testing
        return [{
            'bbox': [100, 100, 200, 150],
            'category': 'sofa',
            'confidence': 0.85,
            'raw_label': 'sofa'
        }]

class DummySegmenter:
    async def segment(self, image_path: str, bbox: List[float]) -> str:
        # Return dummy mask path for testing
        return "/tmp/dummy_mask.png"

class DummyEmbedder:
    async def embed_object(self, image_path: str, bbox: List[float]) -> List[float]:
        # Return dummy embedding for testing
        return [0.1] * 512

# Initialize FastAPI app
app = FastAPI(
    title="Modomo Scraper API (Basic)",
    description="Basic dataset creation system - install AI deps separately",
    version="1.0.0-basic"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
db_pool = None
redis_client = None

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

# Startup/shutdown events
@app.on_event("startup")
async def startup():
    global db_pool, redis_client
    
    try:
        # Database connection - Use Supabase for cloud deployment, local for development
        DATABASE_URL = os.getenv("DATABASE_URL_CLOUD") or os.getenv("DATABASE_URL", "postgresql://reroom:reroom_dev_pass@localhost:5432/reroom_dev")
        db_pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=10)
        print("✅ Connected to database")
        
        # Redis connection
        REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
        redis_client = redis.from_url(REDIS_URL)
        print("✅ Connected to Redis")
        
    except Exception as e:
        print(f"⚠️ Startup warning: {e}")
        print("   Some features may not work without database/redis")

@app.on_event("shutdown")
async def shutdown():
    if db_pool:
        await db_pool.close()
    if redis_client:
        redis_client.close()

# Health check
@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "timestamp": datetime.utcnow(),
        "mode": "basic",
        "note": "Install AI dependencies for full functionality"
    }

# Basic endpoints
@app.get("/")
async def root():
    return {
        "message": "Modomo Dataset Creation System (Basic Mode)",
        "docs": "/docs",
        "health": "/health",
        "note": "This is the basic version. Install torch/transformers for full AI features."
    }

@app.get("/taxonomy")
async def get_taxonomy():
    """Get the furniture taxonomy"""
    return MODOMO_TAXONOMY

@app.post("/test/detection")
async def test_detection():
    """Test endpoint to verify the API is working"""
    detector = DummyDetector()
    detections = await detector.detect_objects("dummy_path", MODOMO_TAXONOMY)
    return {
        "status": "success",
        "detections": detections,
        "note": "This is a dummy detection. Install AI packages for real detection."
    }

# Scraping endpoints (real implementation)
@app.post("/scrape/scenes")
async def start_scene_scraping(
    background_tasks: BackgroundTasks,
    limit: int = Query(10, description="Number of scenes to scrape"),
    room_types: List[str] = Query(None, description="Filter by room types")
):
    """Start scraping scenes from Houzz UK"""
    job_id = str(uuid.uuid4())
    
    # Store initial job status
    if redis_client:
        try:
            redis_client.set(f"job:{job_id}", json.dumps({
                "status": "running",
                "progress": 0,
                "total_items": limit,
                "processed_items": 0,
                "scenes_count": 0,
                "started_at": datetime.utcnow().isoformat(),
                "room_types": room_types or ["all"]
            }), ex=3600)  # Expire after 1 hour
        except:
            pass
    
    # Start background scraping task
    background_tasks.add_task(run_houzz_scraping, job_id, limit, room_types)
    
    return {
        "job_id": job_id, 
        "status": "running",
        "message": f"Started scraping {limit} scenes from Houzz UK",
        "room_types": room_types or ["all"]
    }

@app.get("/scrape/scenes/{job_id}/status")
async def get_scraping_status(job_id: str):
    """Get status of a scraping job"""
    if redis_client:
        try:
            status = redis_client.get(f"job:{job_id}")
            if status:
                return json.loads(status)
        except:
            pass
    
    return {
        "status": "not_found",
        "note": "Job not found or Redis not available"
    }

# Review endpoints (basic)
@app.get("/review/queue")
async def get_review_queue(
    limit: int = Query(10),
    room_type: Optional[str] = Query(None),
    category: Optional[str] = Query(None)
):
    """Get scenes pending review (basic version)"""
    return {
        "scenes": [],
        "note": "No scenes available in basic mode. Run full scraping to populate."
    }

@app.get("/stats/dataset")
async def get_dataset_stats():
    """Get dataset statistics"""
    return {
        "total_scenes": 0,
        "approved_scenes": 0,
        "total_objects": 0,
        "approved_objects": 0,
        "unique_categories": len([item for items in MODOMO_TAXONOMY.values() for item in items]),
        "avg_confidence": 0.0,
        "objects_with_products": 0,
        "note": "Basic mode stats. Run full system for real data."
    }

@app.get("/stats/categories")
async def get_category_stats():
    """Get category-wise statistics"""
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

@app.get("/jobs/active")
async def get_active_jobs():
    """Get currently active jobs"""
    return {
        "scraping_jobs": [],
        "detection_jobs": [],
        "export_jobs": [],
        "note": "No active jobs in basic mode"
    }

# Background scraping task
async def run_houzz_scraping(job_id: str, limit: int, room_types: Optional[List[str]]):
    """Background task to scrape scenes from Houzz"""
    try:
        if not CRAWLER_AVAILABLE:
            # Update job with error
            if redis_client:
                try:
                    redis_client.set(f"job:{job_id}", json.dumps({
                        "status": "failed",
                        "progress": 0,
                        "error": "Houzz crawler dependencies not available",
                        "completed_at": datetime.utcnow().isoformat()
                    }), ex=3600)
                except:
                    pass
            return
        
        print(f"🕷️ Starting Houzz scraping job {job_id} - {limit} scenes")
        
        # Initialize crawler
        crawler = HouzzCrawler()
        
        # Update job status - starting
        if redis_client:
            try:
                redis_client.set(f"job:{job_id}", json.dumps({
                    "status": "running",
                    "progress": 10,
                    "total_items": limit,
                    "processed_items": 0,
                    "scenes_count": 0,
                    "message": "Initializing crawler...",
                    "updated_at": datetime.utcnow().isoformat()
                }), ex=3600)
            except:
                pass
        
        # Start scraping
        scenes = await crawler.scrape_scenes(limit=limit, room_types=room_types)
        
        # Update progress
        if redis_client:
            try:
                redis_client.set(f"job:{job_id}", json.dumps({
                    "status": "running",
                    "progress": 50,
                    "total_items": limit,
                    "processed_items": len(scenes),
                    "scenes_count": len(scenes),
                    "message": f"Scraped {len(scenes)} scenes, saving to database...",
                    "updated_at": datetime.utcnow().isoformat()
                }), ex=3600)
            except:
                pass
        
        # Save scenes to database
        saved_count = 0
        if db_pool and scenes:
            try:
                async with db_pool.acquire() as conn:
                    for scene in scenes:
                        try:
                            await conn.execute("""
                                INSERT INTO scenes (houzz_id, image_url, room_type, style_tags, color_tags, project_url, status)
                                VALUES ($1, $2, $3, $4, $5, $6, 'scraped')
                                ON CONFLICT (houzz_id) DO NOTHING
                            """, scene.houzz_id, scene.image_url, scene.room_type, 
                            scene.style_tags, scene.color_tags, scene.project_url)
                            saved_count += 1
                        except Exception as e:
                            print(f"Failed to save scene {scene.houzz_id}: {e}")
                            continue
            except Exception as e:
                print(f"Database error during scene saving: {e}")
        
        # Final job update - completed
        final_status = {
            "status": "completed",
            "progress": 100,
            "total_items": limit,
            "processed_items": len(scenes),
            "scenes_count": len(scenes),
            "saved_count": saved_count,
            "message": f"Successfully scraped and saved {saved_count} scenes from Houzz UK",
            "completed_at": datetime.utcnow().isoformat()
        }
        
        if redis_client:
            try:
                redis_client.set(f"job:{job_id}", json.dumps(final_status), ex=3600)
            except:
                pass
        
        # Clean up crawler
        await crawler.close()
        
        print(f"✅ Houzz scraping job {job_id} completed: {saved_count}/{len(scenes)} scenes saved")
        
    except Exception as e:
        print(f"❌ Houzz scraping job {job_id} failed: {e}")
        
        # Update job with failure status
        if redis_client:
            try:
                redis_client.set(f"job:{job_id}", json.dumps({
                    "status": "failed",
                    "progress": 0,
                    "error": str(e),
                    "completed_at": datetime.utcnow().isoformat()
                }), ex=3600)
            except:
                pass

# Add new endpoint to get scraped scenes
@app.get("/scenes")
async def get_scenes(
    limit: int = Query(20, description="Number of scenes to return"),
    room_type: Optional[str] = Query(None, description="Filter by room type"),
    status: Optional[str] = Query(None, description="Filter by status")
):
    """Get scraped scenes from database"""
    if not db_pool:
        return {"scenes": [], "note": "Database not available"}
    
    try:
        async with db_pool.acquire() as conn:
            # Build query with filters
            where_clauses = []
            params = []
            param_count = 0
            
            if room_type:
                param_count += 1
                where_clauses.append(f"room_type = ${param_count}")
                params.append(room_type)
            
            if status:
                param_count += 1
                where_clauses.append(f"status = ${param_count}")
                params.append(status)
            
            where_sql = " WHERE " + " AND ".join(where_clauses) if where_clauses else ""
            
            param_count += 1
            query = f"""
                SELECT scene_id, houzz_id, image_url, room_type, style_tags, color_tags, 
                       project_url, status, created_at, updated_at
                FROM scenes 
                {where_sql}
                ORDER BY created_at DESC 
                LIMIT ${param_count}
            """
            params.append(limit)
            
            rows = await conn.fetch(query, *params)
            
            scenes = []
            for row in rows:
                scenes.append({
                    "scene_id": str(row['scene_id']),
                    "houzz_id": row['houzz_id'],
                    "image_url": row['image_url'],
                    "room_type": row['room_type'],
                    "style_tags": row['style_tags'],
                    "color_tags": row['color_tags'],
                    "project_url": row['project_url'],
                    "status": row['status'],
                    "created_at": row['created_at'].isoformat() if row['created_at'] else None,
                    "updated_at": row['updated_at'].isoformat() if row['updated_at'] else None
                })
            
            return {
                "scenes": scenes,
                "count": len(scenes),
                "filters": {"room_type": room_type, "status": status}
            }
            
    except Exception as e:
        print(f"Failed to fetch scenes: {e}")
        return {"error": str(e), "scenes": []}

if __name__ == "__main__":
    try:
        import uvicorn
        
        # Detect if running in production (Railway sets RAILWAY_ENVIRONMENT)
        is_production = os.getenv("RAILWAY_ENVIRONMENT") == "production" or os.getenv("PORT")
        port = int(os.getenv("PORT", 8001))
        
        if not is_production:
            print("🚀 Starting Modomo Scraper (Basic Mode)")
            print("📊 Dashboard will be available at: http://localhost:3001")
            print(f"📖 API docs available at: http://localhost:{port}/docs")
            print("")
            print("💡 This is basic mode without AI dependencies.")
            print("   Install torch, transformers, etc. for full functionality.")
            print("")
        else:
            print("🚀 Starting Modomo Scraper (Production)")
            print(f"🌐 API running on port {port}")
        
        uvicorn.run("main-basic:app", host="0.0.0.0", port=port, log_level="info")
    except ImportError:
        print("❌ uvicorn not installed. Install with: pip install uvicorn[standard]")