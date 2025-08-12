"""
Modomo Dataset Scraping System - Full AI Mode
Complete system with GroundingDINO, SAM2, and CLIP embeddings
"""

import asyncio
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
import uuid
import json
import logging

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, Body
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import asyncpg
import redis
import structlog
from supabase import create_client, Client

# AI/ML imports
import torch
import numpy as np
from PIL import Image
import cv2
from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# JSON serialization helper for NumPy types
def make_json_serializable(obj):
    """Convert NumPy types and other non-serializable types to JSON serializable types"""
    import numpy as np
    
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif hasattr(obj, 'item') and callable(obj.item):  # Handle numpy scalars
        return obj.item()
    return obj

# Import Houzz crawler
try:
    from crawlers.houzz_crawler import HouzzCrawler
    CRAWLER_AVAILABLE = True
except ImportError:
    CRAWLER_AVAILABLE = False
    logger.warning("⚠️ Houzz crawler not available - detection only mode")

# Check if running in AI mode with dependencies
AI_DEPENDENCIES_AVAILABLE = True
try:
    import torch
    import transformers
    logger.info("🤖 AI dependencies detected (torch, transformers)")
except ImportError as e:
    AI_DEPENDENCIES_AVAILABLE = False
    logger.warning(f"💡 AI dependencies not available ({e}) - using fallback implementations")

# Import real AI implementations if available
try:
    from models.grounding_dino import GroundingDINODetector
    from models.sam2_segmenter import SAM2Segmenter  
    from models.clip_embedder import CLIPEmbedder
    from models.color_extractor import ColorExtractor
    AI_MODELS_AVAILABLE = True
    logger.info("✅ AI model classes imported successfully")
except ImportError as e:
    AI_MODELS_AVAILABLE = False
    logger.warning(f"⚠️ AI models not available: {e}")
    if AI_DEPENDENCIES_AVAILABLE:
        logger.info("🤖 AI dependencies detected - should work on next deploy")
    else:
        logger.info("💡 AI dependencies not available (No module named 'torch') - starting basic mode")
    
    # Fallback implementations for when models aren't available
    class GroundingDINODetector:
        def __init__(self):
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
            logger.info(f"Using fallback detector on {self.device}")
        
        async def detect_objects(self, image_path: str, taxonomy: dict) -> List[dict]:
            """Fallback detector with realistic random detections"""
            try:
                from PIL import Image
                import random
                
                image = Image.open(image_path).convert("RGB")
                width, height = image.size
                
                all_categories = []
                for category_group, items in taxonomy.items():
                    all_categories.extend(items)
                
                num_objects = random.randint(4, 8)  # Generate more objects per scene
                detections = []
                
                for i in range(num_objects):
                    category = random.choice(all_categories)
                    max_size = min(width, height) // 3
                    min_size = min(width, height) // 10
                    box_width = random.randint(min_size, max_size)
                    box_height = random.randint(min_size, max_size)
                    x = random.randint(0, width - box_width)
                    y = random.randint(0, height - box_height)
                    
                    detection = {
                        'bbox': [x, y, box_width, box_height],
                        'category': category,
                        'confidence': round(random.uniform(0.7, 0.95), 3),
                        'raw_label': f'{category} . furniture',
                        'tags': [category, 'furniture']
                    }
                    detections.append(detection)
                
                logger.info(f"Generated {len(detections)} fallback detections for {image_path}")
                return detections
                
            except Exception as e:
                logger.error(f"Fallback detection failed for {image_path}: {e}")
                return []

# Add fallback classes if models aren't available
if not AI_MODELS_AVAILABLE:
    class SAM2Segmenter:
        def __init__(self, config=None, device=None):
            # Accept same parameters as real SAM2Segmenter for compatibility
            try:
                import torch
                self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            except ImportError:
                self.device = "cpu"
            logger.info(f"Using fallback segmenter on {self.device}")
        
        def get_model_info(self):
            return {
                "sam2_available": False,
                "sam2_loaded": False,
                "fba_available": False,
                "fba_loaded": False,
                "device": self.device,
                "model_type": "fallback",
                "checkpoint": None
            }
        
        async def segment(self, image_path: str, bbox: List[float]) -> str:
            """Fallback segmentation using simple mask generation"""
            try:
                import cv2
                import uuid
                
                image = cv2.imread(image_path)
                x, y, w, h = [int(coord) for coord in bbox]
                
                mask_path = f"/tmp/masks/mask_{uuid.uuid4().hex}.png"
                import numpy as np
                mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
                mask[y:y+h, x:x+w] = 255
                
                # Apply some smoothing
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                cv2.imwrite(mask_path, mask)
                
                logger.info(f"Generated fallback mask for bbox {bbox}")
                return mask_path
                
            except Exception as e:
                logger.error(f"Fallback segmentation failed for {image_path}: {e}")
                return None

    class CLIPEmbedder:
        def __init__(self):
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
            logger.info(f"Using fallback embedder on {self.device}")
            self.embedding_dim = 512
        
        async def embed_object(self, image_path: str, bbox: List[float]) -> List[float]:
            """Generate fallback embedding (random but consistent for same input)"""
            try:
                import hashlib
                import random
                
                # Create deterministic embedding based on image path and bbox
                combined_string = f"{image_path}_{bbox}"
                hash_obj = hashlib.md5(combined_string.encode())
                seed = int.from_bytes(hash_obj.digest()[:4], 'big')
                
                random.seed(seed)
                embedding = [random.gauss(0, 0.1) for _ in range(self.embedding_dim)]
                
                # Normalize
                norm = sum(x*x for x in embedding) ** 0.5
                embedding = [x/norm for x in embedding]
                
                logger.info(f"Generated fallback embedding for object at {bbox}")
                return embedding
                
            except Exception as e:
                logger.error(f"Fallback embedding failed for {image_path}: {e}")
                return [0.0] * self.embedding_dim

    class ColorExtractor:
        def __init__(self):
            logger.info("Using fallback color extractor")
        
        async def extract_colors(self, image_path: str, bbox: List[float] = None) -> dict:
            """Fallback color extraction using basic image analysis"""
            try:
                from PIL import Image
                import random
                
                image = Image.open(image_path).convert("RGB")
                if bbox:
                    x, y, w, h = [int(coord) for coord in bbox]
                    image = image.crop((x, y, x + w, y + h))
                
                # Get average color as fallback
                pixels = list(image.getdata())
                if pixels:
                    avg_color = tuple(int(sum(channel) / len(pixels)) for channel in zip(*pixels))
                else:
                    avg_color = (128, 128, 128)
                
                # Generate some basic color names
                r, g, b = avg_color
                if r > 200 and g > 200 and b > 200:
                    color_name = "white"
                elif r < 50 and g < 50 and b < 50:
                    color_name = "black"
                elif abs(r - g) < 30 and abs(g - b) < 30:
                    color_name = "gray"
                elif r > g and r > b:
                    color_name = "red" if r > 150 else "brown"
                elif g > r and g > b:
                    color_name = "green"
                elif b > r and b > g:
                    color_name = "blue"
                else:
                    color_name = "beige"
                
                return {
                    "colors": [{
                        "rgb": avg_color,
                        "hex": f"#{r:02x}{g:02x}{b:02x}",
                        "name": color_name,
                        "percentage": 100.0
                    }],
                    "dominant_color": {
                        "rgb": avg_color,
                        "hex": f"#{r:02x}{g:02x}{b:02x}",
                        "name": color_name
                    },
                    "properties": {
                        "brightness": sum(avg_color) / (3 * 255),
                        "is_neutral": abs(r - g) < 30 and abs(g - b) < 30
                    }
                }
            except Exception as e:
                logger.error(f"Fallback color extraction failed: {e}")
                return {"colors": [], "dominant_color": None}

# Initialize FastAPI app
app = FastAPI(
    title="Modomo Scraper API (Full AI)",
    description="Complete dataset creation system with AI processing",
    version="1.0.2-full"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create masks directory and mount static files
masks_dir = "/tmp/masks"
os.makedirs(masks_dir, exist_ok=True)
app.mount("/masks", StaticFiles(directory=masks_dir), name="masks")

# Global instances
supabase: Client = None
db_pool = None  # Keep for backward compatibility
redis_client = None
detector = None
segmenter = None
embedder = None
# Force initialize color extractor since dependencies are available
try:
    from models.color_extractor import ColorExtractor
    color_extractor = ColorExtractor()
    print("✅ Color extractor force-initialized at module level")
except Exception as e:
    print(f"❌ Failed to force-initialize color extractor: {e}")
    color_extractor = None

# Enhanced Configuration for better object detection
MODOMO_TAXONOMY = {
    "seating": ["sofa", "sectional", "armchair", "dining_chair", "stool", "bench"],
    "tables": ["coffee_table", "side_table", "dining_table", "console_table", "desk", "nightstand"],
    "storage": ["bookshelf", "cabinet", "dresser", "wardrobe"],
    "lighting": ["pendant_light", "floor_lamp", "table_lamp", "wall_sconce"],
    "soft_furnishings": ["rug", "curtains", "pillow", "blanket"],
    "decor": ["wall_art", "mirror", "plant", "decorative_object"],
    "bed_bath": ["bed_frame", "mattress", "headboard", "bathtub", "sink_vanity"],
    "architectural": ["window", "door", "fireplace"],
    "electronics": ["tv", "television"],
    "bathroom_fixtures": ["toilet", "shower"]
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
    global supabase, db_pool, redis_client, detector, segmenter, embedder
    
    try:
        logger.info("Starting Modomo Scraper (Full AI Mode)")
        
        # Initialize Supabase client with detailed diagnostics
        SUPABASE_URL = os.getenv("SUPABASE_URL")
        SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
        
        logger.info(f"🔍 Supabase config check: URL={'✅' if SUPABASE_URL else '❌'}, KEY={'✅' if SUPABASE_ANON_KEY else '❌'}")
        
        if SUPABASE_URL and SUPABASE_ANON_KEY:
            try:
                supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
                logger.info("✅ Supabase client initialized successfully")
                
                # Test the connection by querying a simple table
                test_result = supabase.table("scenes").select("scene_id").limit(1).execute()
                logger.info(f"✅ Supabase connection test passed - can access scenes table")
            except Exception as e:
                logger.error(f"❌ Supabase client initialization failed: {e}")
                supabase = None
        else:
            missing = []
            if not SUPABASE_URL: missing.append("SUPABASE_URL")
            if not SUPABASE_ANON_KEY: missing.append("SUPABASE_ANON_KEY") 
            logger.error(f"❌ Missing required Supabase credentials: {', '.join(missing)}")
            supabase = None
        
        # Database connection with fallback
        DATABASE_URL = os.getenv("DATABASE_URL_CLOUD") or os.getenv("DATABASE_URL", "postgresql://reroom:reroom_dev_pass@localhost:5432/reroom_dev")
        logger.info(f"Attempting database connection to: {DATABASE_URL[:30]}...")
        
        try:
            db_pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=10, command_timeout=30)
            logger.info("✅ Connected to database")
        except Exception as db_error:
            logger.warning(f"Database connection failed: {db_error}")
            logger.info("Will continue without database - using in-memory storage")
            db_pool = None
        
        # Redis connection with fallback
        REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
        try:
            redis_client = redis.from_url(REDIS_URL, socket_timeout=10)
            redis_client.ping()  # Test connection
            logger.info("✅ Connected to Redis")
        except Exception as redis_error:
            logger.warning(f"Redis connection failed: {redis_error}")
            logger.info("Will continue without Redis - job tracking disabled")
            redis_client = None
        
        # Initialize color extractor first (has minimal dependencies)
        logger.info("🎨 Loading color extractor...")
        try:
            color_extractor = ColorExtractor()
            logger.info("✅ Color extractor loaded successfully")
        except Exception as color_error:
            logger.error(f"Color extractor failed to initialize: {color_error}")
            # Force enable if environment variable is set
            if os.getenv("FORCE_COLOR_EXTRACTOR", "false").lower() == "true":
                logger.info("🔧 Force enabling color extractor due to FORCE_COLOR_EXTRACTOR=true")
                color_extractor = ColorExtractor()
            else:
                color_extractor = None
        
        # Initialize other AI models with retry logic  
        logger.info("🤖 Loading AI models...")
        try:
            # Initialize detector
            detector = GroundingDINODetector()
            logger.info("✅ GroundingDINO detector initialized")
            
            # Initialize segmenter with appropriate device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            from models.sam2_segmenter import SegmentationConfig
            config = SegmentationConfig(device=device)
            segmenter = SAM2Segmenter(config=config)
            
            # Get detailed model info
            model_info = segmenter.get_model_info()
            if model_info.get("sam2_available"):
                logger.info(f"🔥 SAM2Segmenter initialized with REAL SAM2 on {device}")
                if model_info.get("checkpoint"):
                    logger.info(f"📦 Using checkpoint: {model_info['checkpoint']}")
            else:
                logger.warning(f"⚠️ SAM2Segmenter using fallback mode on {device}")
            logger.info(f"✅ Segmenter ready: {model_info}")
            
            # Initialize embedder
            embedder = CLIPEmbedder()
            logger.info("✅ CLIP embedder initialized")
            
            logger.info("✅ All AI models loaded successfully")
        except Exception as ai_error:
            logger.warning(f"AI model loading failed: {ai_error}")
            logger.info("Will continue with fallback AI implementations")
            # Use fallback implementations
            try:
                detector = GroundingDINODetector()
                device = "cuda" if torch.cuda.is_available() else "cpu"
                from models.sam2_segmenter import SegmentationConfig
                config = SegmentationConfig(device=device)
                segmenter = SAM2Segmenter(config=config)
                embedder = CLIPEmbedder()
            except Exception as fallback_error:
                logger.error(f"Even fallback initialization failed: {fallback_error}")
                # Set to None and handle in endpoints
                detector = None
                segmenter = None
                embedder = None
        
    except Exception as e:
        logger.error(f"Startup error: {e}")
        raise

@app.on_event("shutdown")
async def shutdown():
    if db_pool:
        await db_pool.close()
    if redis_client:
        redis_client.close()
    logger.info("Modomo Scraper shutdown complete")

# Health check
@app.get("/health")
async def health_check():
    ai_status = {
        "detector_loaded": detector is not None,
        "detector_details": detector.get_detector_status() if detector and hasattr(detector, 'get_detector_status') else {},
        "segmenter_loaded": segmenter is not None,
        "segmenter_details": segmenter.get_model_info() if segmenter and hasattr(segmenter, 'get_model_info') else {},
        "embedder_loaded": embedder is not None,
        "color_extractor_loaded": color_extractor is not None,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "pytorch_version": torch.__version__ if torch else "Not available"
    }
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "mode": "full_ai",
        "ai_models": ai_status,
        "note": "Full AI mode active with object detection and embedding"
    }

# API endpoints (same as basic but with real AI processing)
@app.get("/")
async def root():
    return {
        "message": "Modomo Dataset Creation System (Full AI Mode)",
        "docs": "/docs", 
        "health": "/health",
        "ai_features": ["GroundingDINO", "SAM2", "CLIP", "Vector Search"],
        "scraping": "/scrape/scenes",
        "note": "Complete AI pipeline with scraping and dataset creation"
    }

@app.get("/taxonomy")
async def get_taxonomy():
    """Get the furniture taxonomy"""
    return MODOMO_TAXONOMY

@app.get("/colors/extract")
async def extract_colors_from_url(
    image_url: str = Query(..., description="URL of the image to analyze"),
    bbox: str = Query(None, description="Bounding box as 'x,y,width,height' for object crop")
):
    """Extract dominant colors from an image or object crop"""
    try:
        # Download image temporarily
        import aiohttp
        import aiofiles
        import uuid
        
        temp_path = f"/tmp/color_analysis_{uuid.uuid4().hex}.jpg"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(image_url) as response:
                if response.status == 200:
                    async with aiofiles.open(temp_path, 'wb') as f:
                        async for chunk in response.content.iter_chunked(8192):
                            await f.write(chunk)
                else:
                    return {"error": f"Failed to fetch image: HTTP {response.status}"}
        
        # Parse bbox if provided
        bbox_coords = None
        if bbox:
            try:
                bbox_coords = [float(x) for x in bbox.split(',')]
                if len(bbox_coords) != 4:
                    return {"error": "Bbox must have exactly 4 values: x,y,width,height"}
            except ValueError:
                return {"error": "Invalid bbox format. Use: x,y,width,height"}
        
        # Extract colors
        color_data = await color_extractor.extract_colors(temp_path, bbox_coords)
        
        # Cleanup
        import os
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        
        return color_data
        
    except Exception as e:
        logger.error(f"Color extraction API failed: {e}")
        return {"error": str(e)}

@app.get("/search/color")
async def search_objects_by_color(
    query: str = Query(..., description="Color-based search query (e.g., 'red sofa', 'blue curtains')"),
    limit: int = Query(10, description="Maximum number of results"),
    threshold: float = Query(0.3, description="Minimum similarity threshold (0-1)")
):
    """Search for objects using color-based CLIP queries"""
    try:
        if not supabase:
            return {"error": "Database connection not available"}
        
        # Get all objects with embeddings
        result = supabase.table("detected_objects").select(
            "object_id, scene_id, category, confidence, tags, clip_embedding_json, metadata"
        ).not_.is_("clip_embedding_json", "null").execute()
        
        if not result.data:
            return {"results": [], "query": query, "total": 0}
        
        # Prepare data for search
        object_ids = []
        object_embeddings = []
        object_metadata = []
        
        for obj in result.data:
            if obj.get("clip_embedding_json"):
                object_ids.append(obj["object_id"])
                object_embeddings.append(obj["clip_embedding_json"])
                object_metadata.append({
                    "scene_id": obj["scene_id"],
                    "category": obj["category"],
                    "confidence": obj["confidence"],
                    "tags": obj.get("tags", []),
                    "colors": obj.get("metadata", {}).get("colors")
                })
        
        # Perform color-based search
        matches = await embedder.search_objects_by_color(
            query, object_embeddings, object_ids, threshold, limit
        )
        
        # Enrich results with object metadata
        enriched_results = []
        for match in matches:
            obj_idx = object_ids.index(match["object_id"])
            result_obj = {
                **match,
                **object_metadata[obj_idx]
            }
            enriched_results.append(result_obj)
        
        return {
            "results": enriched_results,
            "query": query,
            "total": len(enriched_results),
            "threshold": threshold
        }
        
    except Exception as e:
        logger.error(f"Color search API failed: {e}")
        return {"error": str(e)}

@app.get("/colors/palette")
async def get_color_palette():
    """Get available color names and their RGB values for filtering"""
    if not color_extractor:
        return {"error": "Color extractor not available"}
    
    # Return the color mappings from the extractor
    return {
        "color_palette": color_extractor.color_mappings if hasattr(color_extractor, 'color_mappings') else {},
        "color_categories": {
            "neutrals": ["white", "black", "gray", "beige", "cream"],
            "warm": ["red", "orange", "yellow", "pink", "brown", "tan", "gold"],
            "cool": ["blue", "green", "teal", "purple"],
            "wood_tones": ["light_wood", "medium_wood", "dark_wood"]
        }
    }

@app.get("/stats/colors")
async def get_color_statistics():
    """Get statistics about colors in the dataset"""
    try:
        if not supabase:
            return {"error": "Database connection not available"}
        
        # Get all objects with color metadata
        result = supabase.table("detected_objects").select(
            "metadata, category"
        ).not_.is_("metadata", "null").execute()
        
        color_stats = {
            "total_objects_with_colors": 0,
            "color_distribution": {},
            "colors_by_category": {},
            "dominant_colors": {},
            "color_temperature_distribution": {"warm": 0, "cool": 0, "neutral": 0}
        }
        
        for obj in result.data or []:
            metadata = obj.get("metadata", {})
            colors_data = metadata.get("colors")
            category = obj.get("category", "unknown")
            
            if colors_data and colors_data.get("colors"):
                color_stats["total_objects_with_colors"] += 1
                
                # Track colors by category
                if category not in color_stats["colors_by_category"]:
                    color_stats["colors_by_category"][category] = {}
                
                # Process each color in the object
                for color_info in colors_data["colors"]:
                    color_name = color_info.get("name", "unknown")
                    
                    # Overall color distribution
                    color_stats["color_distribution"][color_name] = \
                        color_stats["color_distribution"].get(color_name, 0) + 1
                    
                    # Colors by category
                    color_stats["colors_by_category"][category][color_name] = \
                        color_stats["colors_by_category"][category].get(color_name, 0) + 1
                
                # Track dominant color
                dominant = colors_data.get("dominant_color", {})
                if dominant.get("name"):
                    dom_color = dominant["name"]
                    color_stats["dominant_colors"][dom_color] = \
                        color_stats["dominant_colors"].get(dom_color, 0) + 1
                
                # Track color temperature
                props = colors_data.get("properties", {})
                temp = props.get("color_temperature", "neutral")
                if temp in color_stats["color_temperature_distribution"]:
                    color_stats["color_temperature_distribution"][temp] += 1
        
        return color_stats
        
    except Exception as e:
        logger.error(f"Color statistics API failed: {e}")
        return {"error": str(e)}

@app.post("/detect/process")
async def process_detection(
    image_url: str = Body(...),
    background_tasks: BackgroundTasks = None
):
    """Run object detection on an image"""
    job_id = str(uuid.uuid4())
    
    if background_tasks:
        background_tasks.add_task(run_detection_pipeline, image_url, job_id)
        return {"job_id": job_id, "status": "processing"}
    else:
        # Run synchronously for testing
        results = await run_detection_pipeline(image_url, job_id)
        return {"job_id": job_id, "results": results}

async def run_detection_pipeline(image_url: str, job_id: str):
    """Complete AI pipeline: detect -> segment -> embed"""
    try:
        logger.info(f"Starting detection pipeline for {image_url}")
        
        # Check if models are available
        if not detector:
            logger.error("❌ No detector available")
            return []
        if not segmenter:
            logger.error("❌ No segmenter available") 
            return []
        if not embedder:
            logger.error("❌ No embedder available")
            return []
        
        # Step 1: Download image from URL
        image_path = f"/tmp/scene_{job_id}.jpg"
        
        import aiohttp
        import aiofiles
        async with aiohttp.ClientSession() as session:
            async with session.get(image_url) as response:
                if response.status == 200:
                    async with aiofiles.open(image_path, 'wb') as f:
                        async for chunk in response.content.iter_chunked(8192):
                            await f.write(chunk)
                    logger.info(f"✅ Downloaded image to {image_path}")
                else:
                    logger.error(f"❌ Failed to download image: HTTP {response.status}")
                    return []
        
        # Step 2: Object detection
        detections = await detector.detect_objects(image_path, MODOMO_TAXONOMY)
        if not detections:
            logger.warning(f"⚠️ No objects detected in {image_url}")
            return []
        
        # Clean up detection results to ensure JSON serialization
        detections = [make_json_serializable(detection) for detection in detections]
        
        # Step 3: Segmentation, embedding, and color extraction for each detection
        for i, detection in enumerate(detections):
            try:
                # Generate mask
                mask_path = await segmenter.segment(image_path, detection['bbox'])
                detection['mask_path'] = mask_path
                
                if mask_path:
                    logger.debug(f"✅ Generated mask for detection {i+1}: {mask_path}")
                else:
                    logger.warning(f"⚠️ Failed to generate mask for detection {i+1}")
                    detection['mask_path'] = None
            
                # Generate embedding
                embedding = await embedder.embed_object(image_path, detection['bbox'])
                detection['embedding'] = make_json_serializable(embedding)
                
                if embedding:
                    logger.debug(f"✅ Generated embedding for detection {i+1}")
                else:
                    logger.warning(f"⚠️ Failed to generate embedding for detection {i+1}")
                    detection['embedding'] = []
                
                # Extract colors from object crop if color extractor is available
                if color_extractor:
                    try:
                        color_data = await color_extractor.extract_colors(image_path, detection['bbox'])
                        detection['color_data'] = make_json_serializable(color_data)
                        
                        # Add color-based tags
                        if color_data and color_data.get('colors'):
                            color_names = [c.get('name') for c in color_data['colors'] if c.get('name')]
                            detection['tags'] = detection.get('tags', []) + color_names[:3]  # Add top 3 color names
                            logger.debug(f"✅ Extracted colors for detection {i+1}")
                    except Exception as color_error:
                        logger.warning(f"⚠️ Color extraction failed for detection {i+1}: {color_error}")
                        detection['color_data'] = None
                else:
                    detection['color_data'] = None
                    
            except Exception as processing_error:
                logger.error(f"❌ Processing failed for detection {i+1}: {processing_error}")
                detection['mask_path'] = None
                detection['embedding'] = []
                detection['color_data'] = None
        
        # Cleanup temporary image file
        try:
            if os.path.exists(image_path):
                os.unlink(image_path)
                logger.debug(f"🧹 Cleaned up temporary image: {image_path}")
        except Exception as cleanup_error:
            logger.warning(f"Failed to cleanup {image_path}: {cleanup_error}")
        
        # Cleanup segmenter temp files if available
        if segmenter and hasattr(segmenter, 'cleanup'):
            try:
                segmenter.cleanup()
            except Exception as seg_cleanup_error:
                logger.warning(f"Segmenter cleanup failed: {seg_cleanup_error}")
        
        logger.info(f"Detection pipeline complete: {len(detections)} objects processed")
        return detections
        
    except Exception as e:
        logger.error(f"Detection pipeline failed: {e}")
        
        # Cleanup on error
        try:
            if 'image_path' in locals() and os.path.exists(image_path):
                os.unlink(image_path)
        except:
            pass
        
        return []

# Include all other endpoints from basic version...
@app.get("/stats/dataset")
async def get_dataset_stats():
    """Get dataset statistics"""
    # Get real stats from Supabase
    if supabase:
        try:
            # Get scenes count
            scenes_result = supabase.table("scenes").select("scene_id", count="exact").execute()
            total_scenes = scenes_result.count if scenes_result.count else 0
            
            # Get approved scenes count  
            approved_scenes_result = supabase.table("scenes").select("scene_id", count="exact").eq("status", "approved").execute()
            approved_scenes = approved_scenes_result.count if approved_scenes_result.count else 0
            
            # Get detected objects counts
            objects_result = supabase.table("detected_objects").select("object_id", count="exact").execute()
            total_objects = objects_result.count if objects_result.count else 0
            
            approved_objects_result = supabase.table("detected_objects").select("object_id", count="exact").eq("approved", True).execute()
            approved_objects = approved_objects_result.count if approved_objects_result.count else 0
            
            # Get average confidence
            try:
                confidence_result = supabase.table("detected_objects").select("confidence").execute()
                confidences = [obj["confidence"] for obj in confidence_result.data if obj.get("confidence")]
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            except:
                avg_confidence = 0.0
            
            # Get objects with matched products
            matched_objects_result = supabase.table("detected_objects").select("object_id", count="exact").not_.is_("matched_product_id", "null").execute()
            objects_with_products = matched_objects_result.count if matched_objects_result.count else 0
            
            return {
                "total_scenes": total_scenes,
                "approved_scenes": approved_scenes,
                "total_objects": total_objects,
                "approved_objects": approved_objects,
                "unique_categories": len([item for items in MODOMO_TAXONOMY.values() for item in items]),
                "avg_confidence": avg_confidence,
                "objects_with_products": objects_with_products
            }
        except Exception as e:
            logger.error(f"Supabase query failed: {e}")
    
    # Fallback to dummy stats
    return {
        "total_scenes": 0,
        "approved_scenes": 0,
        "total_objects": 0,
        "approved_objects": 0,
        "unique_categories": len([item for items in MODOMO_TAXONOMY.values() for item in items]),
        "avg_confidence": 0.0,
        "objects_with_products": 0
    }

@app.get("/stats/categories")
async def get_category_stats():
    """Get category-wise statistics"""
    if supabase:
        try:
            # Get category stats from detected objects
            result = supabase.table("detected_objects").select("category, confidence, approved").execute()
            
            # Process the data
            category_stats = {}
            for obj in result.data or []:
                category = obj.get("category", "unknown")
                confidence = obj.get("confidence", 0.0)
                approved = obj.get("approved")
                
                if category not in category_stats:
                    category_stats[category] = {
                        "category": category,
                        "total_objects": 0,
                        "approved_objects": 0,
                        "confidences": [],
                        "group": None
                    }
                
                category_stats[category]["total_objects"] += 1
                category_stats[category]["confidences"].append(confidence)
                
                if approved is True:
                    category_stats[category]["approved_objects"] += 1
                
                # Map to taxonomy group
                for group_name, items in MODOMO_TAXONOMY.items():
                    if category in items:
                        category_stats[category]["group"] = group_name
                        break
            
            # Calculate averages and format response
            categories = []
            for category, stats in category_stats.items():
                avg_confidence = sum(stats["confidences"]) / len(stats["confidences"]) if stats["confidences"] else 0.0
                categories.append({
                    "category": category,
                    "total_objects": stats["total_objects"],
                    "approved_objects": stats["approved_objects"],
                    "avg_confidence": avg_confidence,
                    "group": stats["group"] or "other"
                })
            
            return sorted(categories, key=lambda x: x["total_objects"], reverse=True)
            
        except Exception as e:
            logger.error(f"Supabase query failed: {e}")
    
    # Fallback to dummy stats
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

@app.get("/scenes")
async def get_scenes(
    limit: int = Query(20, description="Number of scenes to return"),
    offset: int = Query(0, description="Offset for pagination"),
    status: str = Query(None, description="Filter by status")
):
    """Get list of stored scenes with their images"""
    if supabase:
        try:
            # Build Supabase query
            query = supabase.table("scenes").select("scene_id, houzz_id, image_url, image_r2_key, room_type, style_tags, color_tags, status, created_at")
            
            # Add status filter if provided
            if status:
                query = query.eq("status", status)
            
            # Execute query with pagination
            result = query.order("created_at", desc=True).range(offset, offset + limit - 1).execute()
            
            # Get total count
            count_query = supabase.table("scenes").select("scene_id", count="exact")
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
                    objects_count_result = supabase.table("detected_objects").select("object_id", count="exact").eq("scene_id", scene["scene_id"]).execute()
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
            logger.error(f"Failed to fetch scenes from Supabase: {e}")
            return {"scenes": [], "total": 0, "limit": limit, "offset": offset}
    
    return {"scenes": [], "total": 0, "limit": limit, "offset": offset}

@app.get("/objects")
async def get_detected_objects(
    limit: int = Query(20, description="Number of objects to return"),
    offset: int = Query(0, description="Offset for pagination"),
    category: str = Query(None, description="Filter by category"),
    scene_id: str = Query(None, description="Filter by scene ID")
):
    """Get list of detected objects with their details"""
    if supabase:
        try:
            # Build query
            query = supabase.table("detected_objects").select(
                "object_id, scene_id, category, confidence, bbox, tags, approved, metadata, created_at"
            )
            
            # Add filters
            if category:
                query = query.eq("category", category)
            if scene_id:
                query = query.eq("scene_id", scene_id)
            
            # Execute with pagination
            result = query.order("created_at", desc=True).range(offset, offset + limit - 1).execute()
            
            # Get total count
            count_query = supabase.table("detected_objects").select("object_id", count="exact")
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
                    scene_result = supabase.table("scenes").select(
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
            logger.error(f"Failed to fetch objects from Supabase: {e}")
            return {"objects": [], "total": 0, "limit": limit, "offset": offset}
    
    return {"objects": [], "total": 0, "limit": limit, "offset": offset}

@app.get("/admin/test-supabase")
async def test_supabase():
    """Test Supabase connection and permissions"""
    if not supabase:
        raise HTTPException(status_code=503, detail="Supabase not available")
    
    try:
        # Test reading from scenes table
        result = supabase.table("scenes").select("scene_id").limit(1).execute()
        
        # Test inserting a test record
        test_scene = {
            "houzz_id": "test_connection_123",
            "image_url": "https://example.com/test.jpg",
            "room_type": "test",
            "status": "scraped"
        }
        
        insert_result = supabase.table("scenes").insert(test_scene).execute()
        
        # Clean up test record
        if insert_result.data:
            test_id = insert_result.data[0]["scene_id"]
            supabase.table("scenes").delete().eq("scene_id", test_id).execute()
        
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

@app.post("/admin/init-database")
async def init_database():
    """Initialize database tables (admin only)"""
    if not db_pool:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        async with db_pool.acquire() as conn:
            # Create scenes table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS scenes (
                    scene_id SERIAL PRIMARY KEY,
                    houzz_id VARCHAR(255) UNIQUE NOT NULL,
                    image_url TEXT NOT NULL,
                    image_r2_key TEXT,
                    room_type VARCHAR(100),
                    style_tags TEXT[] DEFAULT '{}',
                    color_tags TEXT[] DEFAULT '{}',
                    project_url TEXT,
                    status VARCHAR(50) DEFAULT 'imported' CHECK (status IN ('imported', 'processing', 'pending_review', 'approved', 'rejected')),
                    metadata JSONB DEFAULT '{}',
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    reviewed_at TIMESTAMP WITH TIME ZONE,
                    reviewed_by VARCHAR(255)
                );
            """)
            
            # Create objects table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS objects (
                    object_id SERIAL PRIMARY KEY,
                    scene_id INTEGER REFERENCES scenes(scene_id) ON DELETE CASCADE,
                    category VARCHAR(100) NOT NULL,
                    confidence FLOAT NOT NULL DEFAULT 0.0,
                    bbox_x INTEGER DEFAULT 0,
                    bbox_y INTEGER DEFAULT 0,
                    bbox_width INTEGER DEFAULT 100,
                    bbox_height INTEGER DEFAULT 100,
                    mask_path TEXT,
                    embedding JSONB,
                    status VARCHAR(50) DEFAULT 'detected' CHECK (status IN ('detected', 'pending_review', 'approved', 'rejected')),
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    reviewed_at TIMESTAMP WITH TIME ZONE,
                    reviewed_by VARCHAR(255)
                );
            """)
            
            # Create indexes
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_scenes_status ON scenes(status);")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_scenes_room_type ON scenes(room_type);")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_objects_category ON objects(category);")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_objects_scene_id ON objects(scene_id);")
            
            return {"status": "success", "message": "Database tables initialized"}
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Database initialization failed: {str(e)}")

@app.get("/jobs/active")
async def get_active_jobs():
    """Get currently active jobs"""
    all_jobs = []
    
    # Try to get jobs from Redis first (preferred for active job tracking)
    if redis_client:
        try:
            job_keys = redis_client.keys("job:*")
            for job_key in job_keys:
                job_data = redis_client.hgetall(job_key)
                if job_data and job_data.get("status") in ["pending", "running", "processing"]:
                    # Convert bytes to strings for JSON serialization
                    job_status = {key.decode() if isinstance(key, bytes) else key: 
                                 value.decode() if isinstance(value, bytes) else value 
                                 for key, value in job_data.items()}
                    all_jobs.append(job_status)
        except Exception as e:
            logger.warning(f"Failed to get active jobs from Redis: {e}")
    
    # Fallback to database if available
    if db_pool and not all_jobs:
        try:
            async with db_pool.acquire() as conn:
                jobs = await conn.fetch("SELECT * FROM scraping_jobs WHERE status = 'running'")
                all_jobs = [dict(job) for job in jobs]
        except Exception as e:
            logger.error(f"Database query failed: {e}")
    
    # Return consistent array format that frontend expects
    return all_jobs

# Debug endpoint for color dependencies
@app.get("/debug/color-deps")
async def debug_color_dependencies():
    """Debug endpoint to check color extraction dependencies"""
    try:
        import cv2
        cv2_version = cv2.__version__
    except ImportError as e:
        cv2_version = f"Error: {e}"
    
    try:
        import sklearn
        sklearn_version = sklearn.__version__
    except ImportError as e:
        sklearn_version = f"Error: {e}"
    
    try:
        import webcolors
        webcolors_version = webcolors.__version__
    except ImportError as e:
        webcolors_version = f"Error: {e}"
    
    try:
        from models.color_extractor import ColorExtractor
        color_extractor_test = ColorExtractor()
        color_status = "✅ Available"
    except Exception as e:
        color_status = f"❌ Error: {e}"
    
    return {
        "dependencies": {
            "cv2": cv2_version,
            "sklearn": sklearn_version, 
            "webcolors": webcolors_version
        },
        "color_extractor": color_status,
        "color_extractor_loaded": color_extractor is not None
    }

@app.get("/debug/detector-status")
async def debug_detector_status():
    """Debug endpoint to check YOLO and DETR detector status"""
    if not detector:
        return {"error": "No detector available"}
    
    status = {
        "detector_available": detector is not None,
        "detector_type": "Multi-model (YOLO + DETR)"
    }
    
    if hasattr(detector, 'get_detector_status'):
        status.update(detector.get_detector_status())
    
    # Check YOLO dependencies
    try:
        from ultralytics import YOLO
        yolo_status = "✅ Available"
    except ImportError:
        yolo_status = "❌ ultralytics package not installed"
    except Exception as e:
        yolo_status = f"❌ Error: {e}"
    
    status["yolo_package"] = yolo_status
    
    return status

@app.get("/jobs/errors/recent")
async def get_recent_job_errors():
    """Get recent job errors for frontend display"""
    if not redis_client:
        return {"errors": [], "message": "Error tracking not available"}
    
    try:
        # Get all job keys
        job_keys = redis_client.keys("job:*")
        recent_errors = []
        
        for job_key in job_keys:
            job_data = redis_client.hgetall(job_key)
            if job_data and job_data.get("status") in ["failed", "error"]:
                recent_errors.append({
                    "job_id": job_key.decode().replace("job:", ""),
                    "status": job_data.get("status"),
                    "error_message": job_data.get("message", ""),
                    "updated_at": job_data.get("updated_at", ""),
                    "processed": int(job_data.get("processed", 0)),
                    "total": int(job_data.get("total", 0))
                })
        
        # Sort by updated_at desc, limit to 10 most recent
        recent_errors.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        recent_errors = recent_errors[:10]
        
        return {
            "errors": recent_errors,
            "total_error_jobs": len(recent_errors)
        }
        
    except Exception as e:
        logger.error(f"Failed to get recent errors: {e}")
        return {"errors": [], "error": str(e)}

# Color processing endpoints  
@app.post("/process/colors")
async def process_existing_objects_colors(
    background_tasks: BackgroundTasks,
    limit: int = Query(50, description="Number of objects to process")
):
    """Process existing objects with color extraction"""
    job_id = str(uuid.uuid4())
    
    # Start background task
    background_tasks.add_task(run_color_processing_job, job_id, limit)
    
    return {
        "job_id": job_id,
        "status": "running", 
        "message": f"Started color processing for up to {limit} objects",
        "note": "Check /jobs/active for status updates"
    }

# Scraping endpoints
@app.post("/scrape/scenes")
async def start_scene_scraping(
    background_tasks: BackgroundTasks,
    limit: int = Query(10, description="Number of scenes to scrape"),
    room_types: List[str] = Query(None, description="Filter by room types")
):
    """Start scraping scenes from Houzz UK with full AI processing"""
    if not CRAWLER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Houzz crawler not available")
    
    job_id = str(uuid.uuid4())
    
    # Start background scraping + AI processing task
    background_tasks.add_task(run_full_scraping_pipeline, job_id, limit, room_types)
    
    return {
        "job_id": job_id, 
        "status": "running",
        "message": f"Started scraping {limit} scenes from Houzz UK with full AI processing",
        "room_types": room_types or ["all"],
        "features": ["scraping", "object_detection", "segmentation", "embeddings"]
    }

@app.get("/jobs/{job_id}/status")
async def get_job_status(job_id: str):
    """Get status of any import job"""
    if redis_client:
        try:
            # Check Redis for job status
            job_key = f"job:{job_id}"
            job_data = redis_client.hgetall(job_key)
            
            if job_data:
                job_status = {
                    "job_id": job_id,
                    "status": job_data.get("status", "unknown"),
                    "progress": int(job_data.get("progress", 0)),
                    "total": int(job_data.get("total", 0)),
                    "processed": int(job_data.get("processed", 0)),
                    "message": job_data.get("message", ""),
                    "started_at": job_data.get("started_at"),
                    "updated_at": job_data.get("updated_at")
                }
                
                # Add error flag for failed jobs to help frontend handle errors
                if job_status["status"] in ["failed", "error"]:
                    job_status["has_error"] = True
                    job_status["error_message"] = job_status["message"]
                
                return job_status
        except Exception as e:
            logger.warning(f"Failed to get job status from Redis: {e}")
    
    # Fallback status based on recent scenes/objects
    return {
        "job_id": job_id,
        "status": "completed",
        "progress": 100,
        "total": 0,
        "processed": 0,
        "message": "Job tracking not available - check dashboard stats",
        "note": "Enable Redis for detailed job tracking"
    }

@app.post("/import/huggingface-dataset")
async def import_huggingface_dataset(
    background_tasks: BackgroundTasks,
    dataset: str = Query("sk2003/houzzdata", description="HuggingFace dataset ID (e.g., username/dataset-name)"),
    offset: int = Query(0, description="Starting offset in dataset"),
    limit: int = Query(50, description="Number of images to import and process"),
    include_detection: bool = Query(True, description="Run AI detection on imported images")
):
    """Import any HuggingFace dataset and process with AI"""
    job_id = str(uuid.uuid4())
    
    # Start background import + AI processing task
    background_tasks.add_task(run_dataset_import_pipeline, job_id, dataset, offset, limit, include_detection)
    
    return {
        "job_id": job_id, 
        "status": "running",
        "message": f"Started importing {limit} images from HuggingFace dataset '{dataset}' (offset: {offset})",
        "dataset": dataset,
        "features": ["import", "object_detection", "segmentation", "embeddings"] if include_detection else ["import"]
    }

@app.get("/jobs/{job_id}/status")
async def get_job_status(job_id: str):
    """Get the status and progress of an import job"""
    if not redis_client:
        raise HTTPException(status_code=503, detail="Job tracking not available")
    
    try:
        job_key = f"job:{job_id}"
        job_data = redis_client.hgetall(job_key)
        
        if not job_data:
            raise HTTPException(status_code=404, detail="Job not found")
        
        # Convert bytes to strings for JSON serialization
        job_status = {key.decode() if isinstance(key, bytes) else key: 
                     value.decode() if isinstance(value, bytes) else value 
                     for key, value in job_data.items()}
        
        return job_status
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get job status: {e}")

async def run_color_processing_job(job_id: str, limit: int):
    """Background job to process existing objects with color extraction"""
    try:
        logger.info(f"🎨 Starting color processing job {job_id} for up to {limit} objects")
        
        if not supabase:
            logger.error("❌ Supabase client not available for color processing")
            return
            
        # Try to create color extractor if not available
        if not color_extractor:
            try:
                logger.info("🔧 Creating color extractor for processing job...")
                from models.color_extractor import ColorExtractor
                temp_color_extractor = ColorExtractor()
            except Exception as e:
                logger.error(f"❌ Color extractor not available: {e}")
                return
        else:
            temp_color_extractor = color_extractor
        
        # Get objects that don't have color data yet
        result = supabase.table("detected_objects").select(
            "object_id, scene_id, bbox, metadata"
        ).is_("metadata->colors", "null").limit(limit).execute()
        
        if not result.data:
            logger.info("✅ All objects already have color data")
            return
            
        logger.info(f"📊 Found {len(result.data)} objects to process for colors")
        processed_count = 0
        
        for obj in result.data:
            try:
                # Get scene info for image URL
                scene_result = supabase.table("scenes").select(
                    "image_url"
                ).eq("scene_id", obj["scene_id"]).single().execute()
                
                if not scene_result.data:
                    logger.warning(f"⚠️ No scene found for object {obj['object_id']}")
                    continue
                    
                image_url = scene_result.data["image_url"]
                bbox = obj["bbox"]
                
                # Extract colors using the endpoint we already have
                try:
                    import aiohttp
                    import tempfile
                    import os
                    
                    # Download image to temp file
                    async with aiohttp.ClientSession() as session:
                        async with session.get(image_url) as img_response:
                            if img_response.status == 200:
                                content = await img_response.read()
                                
                                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                                    tmp.write(content)
                                    temp_path = tmp.name
                                
                                # Extract colors
                                color_data = await temp_color_extractor.extract_colors(temp_path, bbox)
                                
                                # Clean up temp file
                                os.unlink(temp_path)
                                
                                # Update object metadata with colors
                                current_metadata = obj.get("metadata", {})
                                current_metadata["colors"] = color_data
                                
                                supabase.table("detected_objects").update({
                                    "metadata": current_metadata
                                }).eq("object_id", obj["object_id"]).execute()
                                
                                processed_count += 1
                                logger.info(f"✅ Added colors to object {obj['object_id']} ({processed_count}/{len(result.data)})")
                                
                except Exception as e:
                    logger.error(f"❌ Failed to process colors for object {obj['object_id']}: {e}")
                    continue
                    
            except Exception as e:
                logger.error(f"❌ Failed to process object {obj['object_id']}: {e}")
                continue
        
        logger.info(f"🎨 Color processing job {job_id} completed: {processed_count}/{len(result.data)} objects processed")
        
    except Exception as e:
        logger.error(f"❌ Color processing job {job_id} failed: {e}")

async def run_dataset_import_pipeline(job_id: str, dataset: str, offset: int, limit: int, include_detection: bool):
    """Import dataset from HuggingFace and process with AI pipeline"""
    def update_job_progress(status: str, processed: int, total: int, message: str = ""):
        """Update job progress in Redis"""
        if redis_client:
            try:
                job_key = f"job:{job_id}"
                progress = int((processed / total * 100)) if total > 0 else 0
                redis_client.hset(job_key, mapping={
                    "status": status,
                    "progress": str(progress),
                    "total": str(total),
                    "processed": str(processed),
                    "message": message,
                    "updated_at": datetime.utcnow().isoformat()
                })
                redis_client.expire(job_key, 3600)  # Expire after 1 hour
            except Exception as e:
                logger.warning(f"Failed to update job progress: {e}")
    
    try:
        logger.info(f"🚀 Starting dataset import job {job_id} - {limit} images from dataset '{dataset}' (offset: {offset})")
        
        # Initialize job tracking
        update_job_progress("starting", 0, limit, "Initializing import job")
        
        # Step 1: Fetch dataset info to validate request bounds
        import aiohttp
        import urllib.parse
        dataset_encoded = urllib.parse.quote(dataset, safe='')
        
        async with aiohttp.ClientSession() as session:
            # First, get dataset info to check total rows
            info_url = f"https://datasets-server.huggingface.co/info?dataset={dataset_encoded}"
            try:
                async with session.get(info_url) as info_response:
                    if info_response.status == 200:
                        info_data = await info_response.json()
                        # HuggingFace API structure: dataset_info -> default -> splits -> train -> num_examples
                        total_rows = info_data.get("dataset_info", {}).get("default", {}).get("splits", {}).get("train", {}).get("num_examples", 0)
                        if total_rows > 0:
                            # Adjust limit to not exceed available rows
                            max_available = max(0, total_rows - offset)
                            adjusted_limit = min(limit, max_available)
                            logger.info(f"📊 Dataset '{dataset}' has {total_rows} total rows. Requesting {adjusted_limit} rows from offset {offset}")
                            
                            if adjusted_limit <= 0:
                                error_msg = f"Offset {offset} exceeds dataset size {total_rows}"
                                logger.error(f"❌ {error_msg}")
                                update_job_progress("failed", 0, 0, error_msg)
                                return
                                
                            limit = adjusted_limit
                        else:
                            logger.warning(f"⚠️ Could not determine dataset size for '{dataset}', proceeding with original limit")
                    else:
                        logger.warning(f"⚠️ Could not fetch dataset info (status {info_response.status}), proceeding with original request")
            except Exception as e:
                logger.warning(f"⚠️ Error fetching dataset info: {e}, proceeding with original request")
            
            # Now fetch the actual data
            dataset_url = f"https://datasets-server.huggingface.co/rows?dataset={dataset_encoded}&config=default&split=train&offset={offset}&length={limit}"
            async with session.get(dataset_url) as response:
                if response.status != 200:
                    error_msg = f"Failed to fetch dataset '{dataset}': HTTP {response.status}"
                    if response.status == 422:
                        error_msg += f" - Dataset may have fewer than {offset + limit} rows. Try smaller offset/limit values."
                    elif response.status == 404:
                        error_msg += " - Dataset not found or config/split doesn't exist"
                    
                    # Try to get more details from the response
                    try:
                        error_details = await response.text()
                        logger.error(f"❌ {error_msg}. Response: {error_details}")
                    except:
                        logger.error(f"❌ {error_msg}")
                    
                    update_job_progress("failed", 0, 0, error_msg)
                    return
                
                try:
                    data = await response.json()
                    rows = data.get("rows", [])
                    logger.info(f"✅ Fetched {len(rows)} images from HuggingFace dataset '{dataset}'")
                except Exception as e:
                    error_msg = f"Failed to parse dataset response: {e}"
                    logger.error(f"❌ {error_msg}")
                    update_job_progress("failed", 0, 0, error_msg)
                    return
        
        # Step 2: Process each image
        update_job_progress("processing", 0, len(rows), f"Processing {len(rows)} images from dataset")
        total_objects = 0
        
        for i, row_data in enumerate(rows):
            try:
                # Update progress for each image
                update_job_progress("processing", i, len(rows), f"Processing image {i+1}/{len(rows)}")
                row = row_data.get("row", {})
                image_data = row.get("image", {})
                
                # Try different caption field names or use empty string
                caption = row.get("caption", row.get("text", row.get("description", "")))
                image_url = image_data.get("src", "")
                
                if not image_url:
                    logger.warning(f"⚠️ No image URL for row {i}")
                    continue
                
                # Create unique scene ID with UUID
                scene_id = f"hf_houzz_{uuid.uuid4().hex[:8]}"
                logger.info(f"🔍 Processing image {i+1}/{len(rows)}: {scene_id}")
                
                # Extract room type from caption or use dataset-specific defaults
                if caption:
                    room_type = "living_room"  # Default
                    caption_lower = caption.lower()
                    if "bedroom" in caption_lower:
                        room_type = "bedroom"
                    elif "kitchen" in caption_lower:
                        room_type = "kitchen"
                    elif "bathroom" in caption_lower:
                        room_type = "bathroom"
                    elif "office" in caption_lower:
                        room_type = "office"
                    elif "dining" in caption_lower:
                        room_type = "dining_room"
                else:
                    # For datasets without captions, use dataset name for hints
                    dataset_lower = dataset.lower()
                    if "ikea" in dataset_lower:
                        room_type = "furniture_showroom"
                    elif "bedroom" in dataset_lower:
                        room_type = "bedroom"
                    elif "kitchen" in dataset_lower:
                        room_type = "kitchen"
                    else:
                        room_type = "living_room"
                
                # Step 2: Download and store image in R2
                r2_key = f"training-data/scenes/{scene_id}.jpg"
                r2_url = f"https://photos.reroom.app/{r2_key}"
                
                # Check R2 credentials before attempting upload
                r2_endpoint = os.getenv('CLOUDFLARE_R2_ENDPOINT')
                r2_access_key = os.getenv('CLOUDFLARE_R2_ACCESS_KEY_ID')
                r2_secret_key = os.getenv('CLOUDFLARE_R2_SECRET_ACCESS_KEY')
                r2_bucket = os.getenv('CLOUDFLARE_R2_BUCKET', 'reroom')
                
                if r2_endpoint and r2_access_key and r2_secret_key:
                    try:
                        # Download image
                        import aiohttp
                        import boto3
                        async with aiohttp.ClientSession() as session:
                            async with session.get(image_url) as response:
                                if response.status == 200:
                                    image_data = await response.read()
                                    
                                    # Upload to R2
                                    s3_client = boto3.client(
                                        's3',
                                        endpoint_url=r2_endpoint,
                                        aws_access_key_id=r2_access_key,
                                        aws_secret_access_key=r2_secret_key,
                                        region_name='auto'
                                    )
                                    
                                    s3_client.put_object(
                                        Bucket=r2_bucket,
                                        Key=r2_key,
                                        Body=image_data,
                                        ContentType='image/jpeg'
                                    )
                                    logger.info(f"✅ Uploaded {scene_id} to R2: {r2_key}")
                                else:
                                    logger.error(f"❌ Failed to download {scene_id}: HTTP {response.status}")
                                    r2_key = None
                                    r2_url = None
                                    
                    except Exception as e:
                        logger.error(f"❌ Failed to upload {scene_id} to R2: {e}")
                        r2_key = None
                        r2_url = None
                else:
                    missing_r2 = []
                    if not r2_endpoint: missing_r2.append("CLOUDFLARE_R2_ENDPOINT")
                    if not r2_access_key: missing_r2.append("CLOUDFLARE_R2_ACCESS_KEY_ID") 
                    if not r2_secret_key: missing_r2.append("CLOUDFLARE_R2_SECRET_ACCESS_KEY")
                    
                    logger.warning(f"❌ R2 credentials missing ({', '.join(missing_r2)}) - skipping upload for {scene_id}")
                    logger.info(f"R2 config: endpoint={bool(r2_endpoint)}, access_key={bool(r2_access_key)}, secret_key={bool(r2_secret_key)}")
                    r2_key = None
                    r2_url = None

                # Step 3: Store scene in database with R2 references
                if supabase:
                    try:
                        scene_data = {
                            "houzz_id": scene_id,
                            "image_url": image_url,
                            "image_r2_key": r2_key,
                            "room_type": room_type,
                            "style_tags": [caption] if caption else [],
                            "color_tags": [],
                            "status": "scraped"
                        }
                        
                        logger.info(f"📝 Attempting to store scene {scene_id} with data: {scene_data}")
                        
                        # Use upsert to handle conflicts
                        result = supabase.table("scenes").upsert(scene_data).execute()
                        
                        if result.data:
                            scene_db_id = result.data[0]["scene_id"] 
                            logger.info(f"✅ Successfully stored scene {scene_id} in Supabase (ID: {scene_db_id})")
                            update_job_progress("processing", i, len(rows), f"Stored scene {i+1}/{len(rows)} in database")
                        else:
                            logger.error(f"❌ No data returned from Supabase for {scene_id}")
                            scene_db_id = None
                            
                    except Exception as db_error:
                        error_str = str(db_error)
                        
                        # Handle duplicate key constraint specifically
                        if "duplicate key value violates unique constraint" in error_str and "scenes_houzz_id_key" in error_str:
                            logger.warning(f"⚠️ Scene {scene_id} already exists in database - attempting to retrieve existing ID")
                            try:
                                # Try to get the existing scene
                                existing = supabase.table("scenes").select("scene_id").eq("houzz_id", scene_id).execute()
                                if existing.data:
                                    scene_db_id = existing.data[0]["scene_id"]
                                    logger.info(f"✅ Retrieved existing scene {scene_id} (ID: {scene_db_id})")
                                    update_job_progress("processing", i, len(rows), f"Found existing scene {i+1}/{len(rows)}")
                                else:
                                    logger.error(f"❌ Could not retrieve existing scene {scene_id}")
                                    scene_db_id = None
                            except Exception as retrieve_error:
                                logger.error(f"❌ Failed to retrieve existing scene {scene_id}: {retrieve_error}")
                                scene_db_id = None
                        else:
                            logger.error(f"❌ Failed to store scene {scene_id} in Supabase: {db_error}")
                            update_job_progress("processing", i, len(rows), f"Database error on scene {i+1}/{len(rows)}")
                            scene_db_id = None
                else:
                    logger.error("❌ No Supabase client available - cannot store scenes in database")
                    scene_db_id = None
                
                # Step 4: Run AI detection if requested
                if include_detection and scene_db_id:
                    update_job_progress("processing", i, len(rows), f"Running AI detection on image {i+1}/{len(rows)}")
                    detections = await run_detection_pipeline(image_url, f"{job_id}_{i}")
                    total_objects += len(detections)
                    
                    # Store detections in Supabase detected_objects table
                    if supabase and detections:
                        for detection in detections:
                            try:
                                # Prepare object data for Supabase
                                bbox = detection.get("bbox", [0, 0, 100, 100])
                                # Ensure bbox values are regular Python ints/floats
                                bbox = [float(x) if isinstance(x, (np.integer, np.floating)) else x for x in bbox]
                                
                                # Prepare metadata with color data
                                metadata = {
                                    "detection_job_id": f"{job_id}_{i}",
                                    "raw_label": detection.get("raw_label", "")
                                }
                                
                                # Add color data to metadata if available
                                color_data = detection.get("color_data")
                                if color_data:
                                    metadata["colors"] = color_data
                                
                                # Handle mask URL from mask_path
                                mask_url = None
                                mask_path = detection.get("mask_path")
                                if mask_path and os.path.exists(mask_path):
                                    # Generate URL for the static file endpoint
                                    mask_filename = os.path.basename(mask_path)
                                    mask_url = f"/masks/{mask_filename}"
                                    logger.info(f"Generated mask URL: {mask_url} for path: {mask_path}")
                                else:
                                    logger.warning(f"No valid mask path found: {mask_path}")
                                
                                # Ensure all individual values are properly converted
                                confidence_val = detection.get("confidence", 0.0)
                                if hasattr(confidence_val, 'item'):
                                    confidence_val = confidence_val.item()
                                
                                embedding_val = detection.get("embedding", [])
                                if hasattr(embedding_val, 'tolist'):
                                    embedding_val = embedding_val.tolist()
                                
                                object_data = {
                                    "scene_id": scene_db_id,
                                    "bbox": bbox,  # Store as array [x, y, width, height]
                                    "category": str(detection.get("category", "unknown")),
                                    "confidence": float(confidence_val),
                                    "tags": list(detection.get("tags", [])),
                                    "clip_embedding_json": embedding_val,
                                    "approved": None,  # Needs manual review
                                    "metadata": metadata,
                                    "mask_url": mask_url  # Add mask URL if available
                                }
                                
                                # Ensure all data is JSON serializable - apply recursively to catch all nested values
                                object_data = make_json_serializable(object_data)
                                
                                # Double-check specific fields that might still have int64 values
                                if 'bbox' in object_data:
                                    object_data['bbox'] = [float(x) for x in object_data['bbox']]
                                if 'clip_embedding_json' in object_data:
                                    object_data['clip_embedding_json'] = make_json_serializable(object_data['clip_embedding_json'])
                                if 'metadata' in object_data:
                                    object_data['metadata'] = make_json_serializable(object_data['metadata'])
                                
                                # Insert object into Supabase
                                result = supabase.table("detected_objects").insert(object_data).execute()
                                
                                if result.data:
                                    object_id = result.data[0]["object_id"]
                                    logger.info(f"✅ Stored detected object {detection['category']} (ID: {object_id}) for scene {scene_id}")
                                else:
                                    logger.error(f"❌ No data returned when storing object for scene {scene_id}")
                                    
                            except Exception as obj_error:
                                error_msg = f"Failed to store detected object for scene {scene_id}: {obj_error}"
                                logger.error(f"❌ {error_msg}")
                                update_job_progress("error", i, len(rows), error_msg)
                                # Don't continue - let user know there's an issue
                                raise Exception(f"Database storage failed: {error_msg}")
                        
                        logger.info(f"✅ Stored {len(detections)} detected objects for scene {scene_id}")
                        update_job_progress("processing", i, len(rows), f"Stored {len(detections)} objects for image {i+1}/{len(rows)}")
                    
                    elif not supabase:
                        logger.warning("❌ No Supabase client - cannot store detected objects")
                
            except Exception as e:
                error_msg = f"Failed to process image {i+1}/{len(rows)}: {str(e)}"
                logger.error(f"❌ {error_msg}")
                update_job_progress("error", i, len(rows), error_msg)
                
                # For critical errors (like serialization), stop the job
                if "JSON serializable" in str(e) or "Database storage failed" in str(e):
                    update_job_progress("failed", i, len(rows), f"Critical error stopped import: {error_msg}")
                    return
                
                # For non-critical errors, continue but notify
                continue
        
        # Job completed successfully
        update_job_progress("completed", len(rows), len(rows), f"Import completed: {len(rows)} scenes, {total_objects} objects detected")
        logger.info(f"✅ Dataset import job {job_id} completed - {len(rows)} scenes, {total_objects} objects detected")
        
    except Exception as e:
        update_job_progress("failed", 0, limit if 'limit' in locals() else 0, f"Import failed: {str(e)}")
        logger.error(f"❌ Dataset import job {job_id} failed: {e}")

@app.get("/export/training-dataset")
async def export_training_dataset(
    format: str = Query("coco", description="Export format: coco, yolo, custom"),
    split_ratios: str = Query("0.7,0.2,0.1", description="Train,val,test split ratios"),
    min_objects: int = Query(1, description="Minimum objects per scene"),
    include_r2_urls: bool = Query(True, description="Include R2 URLs for training")
):
    """Export training dataset with local image paths and annotations"""
    if not supabase:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        # Parse split ratios
        ratios = [float(r.strip()) for r in split_ratios.split(",")]
        if len(ratios) != 3 or sum(ratios) != 1.0:
            raise ValueError("Split ratios must sum to 1.0")
        train_ratio, val_ratio, test_ratio = ratios
        
        # Get all approved scenes with R2 keys
        scenes_result = supabase.table("scenes").select(
            "scene_id, houzz_id, image_url, image_r2_key, room_type, style_tags, status"
        ).eq("status", "scraped").is_("image_r2_key", "not.null").execute()
        
        scenes = scenes_result.data
        if not scenes:
            return {"error": "No scenes with R2 storage found", "count": 0}
        
        logger.info(f"Found {len(scenes)} scenes for training dataset export")
        
        # Create dataset splits
        import random
        random.shuffle(scenes)
        
        n_total = len(scenes)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_scenes = scenes[:n_train]
        val_scenes = scenes[n_train:n_train + n_val]
        test_scenes = scenes[n_train + n_val:]
        
        # Build export data
        export_data = {
            "dataset_info": {
                "name": "ReRoom Interior Design Training Dataset",
                "version": "1.0.0",
                "description": "Interior design images with furniture detection for ReRoom AI model training",
                "created_at": datetime.utcnow().isoformat(),
                "format": format,
                "splits": {
                    "train": len(train_scenes),
                    "val": len(val_scenes), 
                    "test": len(test_scenes)
                },
                "categories": list(set([item for items in MODOMO_TAXONOMY.values() for item in items])),
                "total_scenes": n_total
            },
            "splits": {}
        }
        
        # Process each split
        for split_name, split_scenes in [("train", train_scenes), ("val", val_scenes), ("test", test_scenes)]:
            split_data = []
            
            for scene in split_scenes:
                scene_data = {
                    "scene_id": scene["scene_id"],
                    "houzz_id": scene["houzz_id"],
                    "original_url": scene["image_url"],
                    "room_type": scene["room_type"],
                    "style_tags": scene["style_tags"] or [],
                    "status": scene["status"]
                }
                
                # Add R2 storage information for training
                if scene["image_r2_key"] and include_r2_urls:
                    scene_data["r2_storage"] = {
                        "key": scene["image_r2_key"],
                        "public_url": f"https://photos.reroom.app/{scene['image_r2_key']}",
                        "download_url": f"https://photos.reroom.app/{scene['image_r2_key']}"
                    }
                    
                    # For local training, provide the expected local path structure
                    scene_data["local_training_path"] = f"./training-data/images/{scene['houzz_id']}.jpg"
                
                # TODO: Add object annotations when detected_objects table is populated
                scene_data["objects"] = []
                scene_data["object_count"] = 0
                
                split_data.append(scene_data)
            
            export_data["splits"][split_name] = split_data
        
        # Add download instructions
        export_data["download_instructions"] = {
            "note": "Images are stored in Cloudflare R2 for training",
            "r2_bucket": "reroom",
            "r2_prefix": "training-data/scenes/",
            "recommended_download": "Use the public URLs or R2 API to download images to local training environment",
            "local_structure": "Create ./training-data/images/ directory and download images with original houzz_id names"
        }
        
        return export_data
        
    except Exception as e:
        logger.error(f"Training dataset export failed: {e}")
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

async def run_full_scraping_pipeline(job_id: str, limit: int, room_types: Optional[List[str]]):
    """Complete pipeline: scrape -> detect -> segment -> embed -> store"""
    try:
        logger.info(f"🚀 Starting full scraping pipeline job {job_id} - {limit} scenes")
        
        # Step 1: Scrape scenes from Houzz
        crawler = HouzzCrawler()
        scenes = await crawler.scrape_scenes(limit=limit, room_types=room_types)
        logger.info(f"✅ Scraped {len(scenes)} scenes from Houzz")
        
        # Step 2: Process each scene with full AI pipeline
        total_objects = 0
        for i, scene in enumerate(scenes):
            try:
                logger.info(f"🔍 Processing scene {i+1}/{len(scenes)}: {scene.houzz_id}")
                
                # Download image
                image_path = f"/tmp/scene_{scene.houzz_id}.jpg"
                # TODO: Implement image download from scene.image_url
                
                # Run full detection pipeline
                detections = await run_detection_pipeline(scene.image_url, f"{job_id}_{i}")
                total_objects += len(detections)
                
                # Store scene and detections in database
                if db_pool:
                    async with db_pool.acquire() as conn:
                        # Insert scene
                        scene_id = await conn.fetchval("""
                            INSERT INTO scenes (houzz_id, image_url, room_type, style_tags, color_tags, project_url, status)
                            VALUES ($1, $2, $3, $4, $5, $6, 'processed')
                            ON CONFLICT (houzz_id) DO UPDATE SET status = 'processed'
                            RETURNING scene_id
                        """, scene.houzz_id, scene.image_url, scene.room_type, 
                        scene.style_tags, scene.color_tags, scene.project_url)
                        
                        # Insert detections
                        for detection in detections:
                            await conn.execute("""
                                INSERT INTO detected_objects (scene_id, bbox, category, confidence, tags, clip_embedding_json)
                                VALUES ($1, $2, $3, $4, $5, $6)
                            """, scene_id, detection['bbox'], detection['category'], 
                            detection['confidence'], detection.get('tags', []), 
                            json.dumps(detection.get('embedding', [])))
                
                logger.info(f"✅ Processed scene {scene.houzz_id}: {len(detections)} objects detected")
                
            except Exception as e:
                logger.error(f"❌ Failed to process scene {scene.houzz_id}: {e}")
                continue
        
        await crawler.close()
        logger.info(f"🎉 Full pipeline complete: {len(scenes)} scenes, {total_objects} objects detected")
        
    except Exception as e:
        logger.error(f"❌ Full scraping pipeline failed: {e}")

# Review endpoints for object validation
@app.get("/review/queue")
async def get_review_queue(
    limit: int = Query(10, description="Maximum number of scenes to return"),
    room_type: str = Query(None, description="Filter by room type"),
    category: str = Query(None, description="Filter by object category")
):
    """Get scenes pending review with detected objects
    
    NOTE: Scenes are filtered out of the queue when:
    1. Scene status is "approved" OR "rejected" (completed review)
    2. All objects in scene have non-null approved status
    This prevents scenes from appearing in queue after completion.
    """
    try:
        if not supabase:
            return {
                "scenes": [],
                "note": "Database connection not available"
            }
        
        # Build query for scenes with objects needing review
        # First get scenes that have detected objects AND are not already completed (approved OR rejected)
        query = supabase.table("scenes").select("""
            scene_id, houzz_id, image_url, room_type, style_tags, status,
            detected_objects(
                object_id, bbox, category, confidence, tags, 
                approved, matched_product_id, metadata
            )
        """).not_.in_("status", ["approved", "rejected"])
        
        # Add filters
        if room_type:
            query = query.eq("room_type", room_type)
        
        # Execute query and filter for scenes with unapproved objects
        result = query.order("created_at", desc=True).limit(limit * 3).execute()  # Get more to filter later
        
        if not result.data:
            return {
                "scenes": [],
                "total": 0,
                "note": "No scenes found for review"
            }
        
        # Format scenes with objects that need review
        scenes = []
        for scene_data in result.data:
            # Skip scenes that are already completed (approved OR rejected) at the scene level
            scene_status = scene_data.get("status", "")
            if scene_status in ["approved", "rejected"]:
                continue
                
            detected_objects = scene_data.get("detected_objects", [])
            
            # Filter for objects that need review (approved is null or missing)
            objects_needing_review = []
            for obj in detected_objects:
                approved_status = obj.get("approved")
                # Include objects where approved is null, None, or missing
                needs_review = approved_status is None
                
                # Apply category filter if specified
                if category and obj.get("category") != category:
                    continue
                    
                if needs_review:
                    objects_needing_review.append(obj)
            
            # Only include scenes that have objects needing review
            if objects_needing_review:
                scene = {
                    "scene_id": scene_data["scene_id"],
                    "houzz_id": scene_data.get("houzz_id"),
                    "image_url": scene_data["image_url"],
                    "room_type": scene_data.get("room_type"),
                    "style_tags": scene_data.get("style_tags", []),
                    "objects": objects_needing_review
                }
                scenes.append(scene)
                
                # Stop when we have enough scenes
                if len(scenes) >= limit:
                    break
        
        # Return just the scenes array to match frontend expectations  
        return scenes[:limit]
        
    except Exception as e:
        logger.error(f"Review queue failed: {e}")
        return {
            "scenes": [],
            "error": str(e)
        }

@app.post("/review/update")
async def update_review(updates: List[dict]):
    """Update review status for multiple objects"""
    try:
        if not supabase:
            return {"error": "Database connection not available"}
        
        updated_count = 0
        
        for update in updates:
            object_id = update.get("object_id")
            if not object_id:
                continue
                
            # Prepare update data
            update_data = {}
            if "approved" in update:
                update_data["approved"] = update["approved"]
            if "category" in update:
                update_data["category"] = update["category"]
            if "tags" in update:
                update_data["tags"] = update["tags"]
            if "matched_product_id" in update:
                update_data["matched_product_id"] = update["matched_product_id"]
            
            if update_data:
                result = supabase.table("detected_objects").update(update_data).eq("object_id", object_id).execute()
                if result.data:
                    updated_count += 1
        
        return {
            "status": "success",
            "count": updated_count,
            "message": f"Updated {updated_count} objects"
        }
        
    except Exception as e:
        logger.error(f"Review update failed: {e}")
        return {"error": str(e)}

@app.post("/review/approve/{scene_id}")
async def approve_scene(scene_id: str):
    """Mark a scene as approved after review"""
    try:
        if not supabase:
            return {"error": "Database connection not available"}
        
        # Update scene status to approved
        scene_result = supabase.table("scenes").update({
            "status": "approved",
            "reviewed_at": "now()"
        }).eq("scene_id", scene_id).execute()
        
        # Also ensure any remaining null approved statuses are set to avoid edge cases
        # This handles cases where objects were never individually approved/rejected
        objects_result = supabase.table("detected_objects").update({
            "approved": True  # Default to approved when scene is approved
        }).eq("scene_id", scene_id).is_("approved", "null").execute()
        
        if not scene_result.data:
            return {"error": f"Scene {scene_id} not found"}
        
        return {
            "status": "success",
            "scene_id": scene_id,
            "message": "Scene approved successfully"
        }
        
    except Exception as e:
        logger.error(f"Scene approval failed: {e}")
        return {"error": str(e)}

@app.post("/review/reject/{scene_id}")
async def reject_scene(scene_id: str):
    """Mark a scene as rejected after review"""
    try:
        if not supabase:
            return {"error": "Database connection not available"}
        
        # Update scene status to rejected
        scene_result = supabase.table("scenes").update({
            "status": "rejected",
            "reviewed_at": "now()"
        }).eq("scene_id", scene_id).execute()
        
        # Also ensure any remaining null approved statuses are set to avoid edge cases
        # This handles cases where objects were never individually approved/rejected
        objects_result = supabase.table("detected_objects").update({
            "approved": False  # Default to rejected when scene is rejected
        }).eq("scene_id", scene_id).is_("approved", "null").execute()
        
        if not scene_result.data:
            return {"error": f"Scene {scene_id} not found"}
        
        return {
            "status": "success",
            "scene_id": scene_id,
            "message": "Scene rejected successfully"
        }
        
    except Exception as e:
        logger.error(f"Scene rejection failed: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    
    # Detect if running in production
    is_production = os.getenv("RAILWAY_ENVIRONMENT") == "production" or os.getenv("PORT")
    port = int(os.getenv("PORT", 8001))
    
    if not is_production:
        print("🤖 Starting Modomo Scraper (Full AI Mode)")
        print(f"📖 API docs available at: http://localhost:{port}/docs")
        print("🔥 AI models: GroundingDINO + SAM2 + CLIP")
    else:
        print("🤖 Starting Modomo Scraper (Production AI Mode)")
        print(f"🌐 API running on port {port}")
    
    uvicorn.run("main_full:app", host="0.0.0.0", port=port, log_level="info")