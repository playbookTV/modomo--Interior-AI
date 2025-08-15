"""
Modomo Dataset Scraping System - Full AI Mode
Complete system with GroundingDINO, SAM2, and CLIP embeddings
"""

import asyncio
import os
import shutil
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

# Ensure models directory is in Python path
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
    
# Debug logging for Railway deployment
logger.info(f"🔍 Current directory: {current_dir}")
logger.info(f"🔍 Models directory exists: {os.path.exists(os.path.join(current_dir, 'models'))}")
logger.info(f"🔍 __init__.py exists: {os.path.exists(os.path.join(current_dir, 'models', '__init__.py'))}")
logger.info(f"🔍 Python path includes current dir: {current_dir in sys.path}")

# Force proper AI model imports - no fallbacks allowed!
AI_MODELS_AVAILABLE = False
try:
    # Run the models import fix first
    try:
        from fix_models_import import fix_models_import
        logger.info("🔧 Running models import fix in main_full.py...")
        fix_models_import()
    except Exception as fix_e:
        logger.warning(f"⚠️ Models import fix failed: {fix_e}")
    
    # Now attempt the real imports
    from models.grounding_dino import GroundingDINODetector
    from models.sam2_segmenter import SAM2Segmenter  
    from models.clip_embedder import CLIPEmbedder
    from models.color_extractor import ColorExtractor
    AI_MODELS_AVAILABLE = True
    logger.info("✅ AI model classes imported successfully - NO FALLBACKS NEEDED!")
    
except ImportError as e:
    logger.error(f"❌ CRITICAL: AI models import failed: {e}")
    logger.error("❌ This should not happen - models should be available!")
    # Don't create fallbacks - let it fail properly so we can fix the root issue
    raise ImportError(f"Required AI models not available: {e}")
except Exception as e:
    logger.error(f"❌ CRITICAL: Unexpected error during model import: {e}")
    raise
# NO MORE FALLBACKS - REAL MODELS ONLY!

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
masks_dir = "/app/cache_volume/masks"
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
    # Primary Furniture Categories
    "seating": ["sofa", "sectional", "armchair", "dining_chair", "stool", "bench", "loveseat", "recliner", "chaise_lounge", "bar_stool", "office_chair", "accent_chair", "ottoman", "pouffe"],
    
    "tables": ["coffee_table", "side_table", "dining_table", "console_table", "desk", "nightstand", "end_table", "accent_table", "writing_desk", "computer_desk", "bar_table", "bistro_table", "nesting_tables", "dressing_table"],
    
    "storage": ["bookshelf", "cabinet", "dresser", "wardrobe", "armoire", "chest_of_drawers", "credenza", "sideboard", "buffet", "china_cabinet", "display_cabinet", "tv_stand", "media_console", "shoe_cabinet", "pantry_cabinet"],
    
    "bedroom": ["bed_frame", "mattress", "headboard", "footboard", "bed_base", "platform_bed", "bunk_bed", "daybed", "murphy_bed", "crib", "bassinet", "changing_table"],
    
    # Lighting & Electrical
    "lighting": ["pendant_light", "floor_lamp", "table_lamp", "wall_sconce", "chandelier", "ceiling_light", "track_lighting", "recessed_light", "under_cabinet_light", "desk_lamp", "reading_light", "accent_lighting", "string_lights"],
    
    "ceiling_fixtures": ["ceiling_fan", "smoke_detector", "air_vent", "skylight", "beam", "molding", "medallion"],
    
    # Kitchen & Appliances
    "kitchen_cabinets": ["upper_cabinet", "lower_cabinet", "kitchen_island", "breakfast_bar", "pantry", "spice_rack", "wine_rack"],
    
    "kitchen_appliances": ["refrigerator", "stove", "oven", "microwave", "dishwasher", "range_hood", "garbage_disposal", "coffee_maker", "toaster", "blender"],
    
    "kitchen_fixtures": ["kitchen_sink", "faucet", "backsplash", "countertop", "kitchen_island_top"],
    
    # Bathroom & Fixtures
    "bathroom_fixtures": ["toilet", "shower", "bathtub", "sink_vanity", "bathroom_sink", "shower_door", "shower_curtain", "medicine_cabinet", "towel_rack", "toilet_paper_holder"],
    
    "bathroom_storage": ["linen_closet", "bathroom_cabinet", "vanity_cabinet", "over_toilet_storage"],
    
    # Textiles & Soft Furnishings
    "window_treatments": ["curtains", "drapes", "blinds", "shades", "shutters", "valance", "cornice", "window_film"],
    
    "soft_furnishings": ["rug", "carpet", "pillow", "cushion", "throw_pillow", "blanket", "throw", "bedding", "duvet", "comforter", "sheets", "pillowcase"],
    
    "upholstery": ["sofa_cushions", "chair_cushions", "seat_cushions", "back_cushions"],
    
    # Decor & Accessories
    "wall_decor": ["wall_art", "painting", "photograph", "poster", "wall_sculpture", "wall_clock", "decorative_plate", "wall_shelf", "floating_shelf"],
    
    "decor_accessories": ["mirror", "plant", "vase", "candle", "sculpture", "decorative_bowl", "picture_frame", "clock", "lamp_shade", "decorative_object"],
    
    "plants_planters": ["potted_plant", "hanging_plant", "planter", "flower_pot", "garden_planter", "herb_garden"],
    
    # Architectural Elements
    "doors_windows": ["door", "window", "french_doors", "sliding_door", "bifold_door", "pocket_door", "window_frame", "door_frame"],
    
    "architectural_features": ["fireplace", "mantle", "column", "pillar", "archway", "niche", "built_in_shelf", "wainscoting", "chair_rail"],
    
    "flooring": ["hardwood_floor", "tile_floor", "carpet_floor", "laminate_floor", "vinyl_floor", "stone_floor"],
    
    "wall_features": ["accent_wall", "brick_wall", "stone_wall", "wood_paneling", "wallpaper"],
    
    # Electronics & Technology
    "entertainment": ["tv", "television", "stereo", "speakers", "gaming_console", "dvd_player", "sound_bar"],
    
    "home_office": ["computer", "monitor", "printer", "desk_accessories", "filing_cabinet", "desk_organizer"],
    
    "smart_home": ["smart_speaker", "security_camera", "thermostat", "smart_switch", "home_hub"],
    
    # Outdoor & Patio
    "outdoor_furniture": ["patio_chair", "outdoor_table", "patio_umbrella", "outdoor_sofa", "deck_chair", "garden_bench", "outdoor_dining_set"],
    
    "outdoor_decor": ["outdoor_plant", "garden_sculpture", "outdoor_lighting", "wind_chime", "bird_feeder"],
    
    # Specialty Items
    "exercise_equipment": ["treadmill", "exercise_bike", "weights", "yoga_mat", "exercise_ball"],
    
    "children_furniture": ["toy_chest", "kids_table", "kids_chair", "high_chair", "play_table", "toy_storage"],
    
    "office_furniture": ["conference_table", "office_desk", "executive_chair", "meeting_chair", "whiteboard", "bulletin_board"],
    
    # Miscellaneous
    "room_dividers": ["screen", "room_divider", "partition", "bookcase_divider"],
    
    "seasonal_decor": ["christmas_tree", "holiday_decoration", "seasonal_pillow", "seasonal_wreath"],
    
    "hardware_fixtures": ["door_handle", "cabinet_hardware", "light_switch", "outlet", "vent_cover"]
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
    global supabase, db_pool, redis_client, detector, segmenter, embedder, color_extractor
    
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
        
        # Use Supabase as primary database (already connected above)
        logger.info("✅ Using Supabase as primary database")
        db_pool = None  # Not needed - using Supabase client instead
        
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
            # Ensure YOLO is available for multi-model approach
            logger.info("🔍 Verifying YOLO availability for multi-model detection...")
            try:
                from ultralytics import YOLO
                # Test YOLO initialization
                test_yolo = YOLO('yolov8n.pt')  # Use nano model for quick test
                logger.info("✅ YOLO verified and ready for multi-model detection")
            except ImportError as yolo_error:
                logger.error(f"❌ YOLO not available: {yolo_error}")
                raise RuntimeError("YOLO is required for multi-model (YOLO + DETR) approach")
            except Exception as yolo_error:
                logger.error(f"❌ YOLO initialization failed: {yolo_error}")
                raise RuntimeError(f"YOLO initialization failed: {yolo_error}")
            
            # Initialize detector (GroundingDINO as DETR component)
            detector = GroundingDINODetector()
            logger.info("✅ GroundingDINO (DETR) detector initialized")
            logger.info("✅ Multi-model (YOLO + DETR) detection system ready")
            
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
            logger.error(f"❌ CRITICAL: AI model loading failed: {ai_error}")
            logger.error("❌ NO MORE FALLBACKS - REAL MODELS ONLY!")
            logger.error("❌ This should not happen in production - models should be available!")
            # Set to None to fail fast and surface the real issue
            detector = None
            segmenter = None
            embedder = None
            raise RuntimeError(f"AI model initialization failed: {ai_error}")
        
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
        if color_extractor is None:
            return {"error": "Color extraction service is not available"}
        
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
        
        # Perform vector search
        matches = await perform_vector_search(
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
    """Debug endpoint to check multi-model detector status"""
    if not detector:
        return {"error": "No detector available"}
    
    status = {
        "detector_available": detector is not None,
        "detector_type": "Multi-model (YOLO + DETR)"
    }
    
    if hasattr(detector, 'get_detector_status'):
        status.update(detector.get_detector_status())
    
    # Check YOLO dependencies (required for multi-model approach)
    try:
        from ultralytics import YOLO
        yolo_status = "✅ Available (required for multi-model detection)"
    except ImportError:
        yolo_status = "❌ Not installed (REQUIRED for multi-model detection)"
    except Exception as e:
        yolo_status = f"❌ Error: {e} (REQUIRED for multi-model detection)"
    
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
            if job_data:
                # Convert bytes to strings for proper comparison and access
                job_status = {key.decode() if isinstance(key, bytes) else key: 
                             value.decode() if isinstance(value, bytes) else value 
                             for key, value in job_data.items()}
                
                if job_status.get("status") in ["failed", "error"]:
                    recent_errors.append({
                        "job_id": job_key.decode().replace("job:", ""),
                        "status": job_status.get("status"),
                        "error_message": job_status.get("message", ""),
                        "updated_at": job_status.get("updated_at", ""),
                        "processed": int(job_status.get("processed", 0)),
                        "total": int(job_status.get("total", 0))
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

# Removed duplicate endpoint - using the proper one below

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

@app.post("/classify/reclassify-scenes")
async def reclassify_existing_scenes(
    background_tasks: BackgroundTasks,
    limit: int = Query(100, description="Number of scenes to reclassify"),
    force_redetection: bool = Query(False, description="Re-run object detection for better classification")
):
    """
    Reclassify existing scenes using enhanced scene vs object detection.
    Useful for improving dataset quality after classification improvements.
    """
    job_id = str(uuid.uuid4())
    
    background_tasks.add_task(run_scene_reclassification_job, job_id, limit, force_redetection)
    
    return {
        "job_id": job_id,
        "status": "running", 
        "message": f"Started reclassifying {limit} scenes with enhanced classification",
        "features": ["scene_classification", "object_detection"] if force_redetection else ["scene_classification"]
    }

@app.get("/classify/test")
async def test_classification(
    image_url: str = Query(..., description="Image URL to test classification"),
    caption: str = Query(None, description="Optional caption/description")
):
    """Test image classification on a single image"""
    try:
        classification = await classify_image_type(image_url, caption)
        return {
            "image_url": image_url,
            "caption": caption,
            "classification": classification,
            "status": "success"
        }
    except Exception as e:
        return {
            "image_url": image_url,
            "error": str(e),
            "status": "failed"
        }

async def run_scene_reclassification_job(job_id: str, limit: int, force_redetection: bool):
    """Background job to reclassify existing scenes"""
    try:
        logger.info(f"🔄 Starting scene reclassification job {job_id} for {limit} scenes")
        
        if not supabase:
            logger.error("❌ Supabase client not available")
            return
            
        # Get scenes that need reclassification (prioritize those without classification)
        scenes_query = supabase.table("scenes").select(
            "scene_id, houzz_id, image_url, image_type, is_primary_object, primary_category, metadata"
        ).order("created_at", desc=True).limit(limit)
        
        scenes_result = scenes_query.execute()
        scenes = scenes_result.data or []
        
        if not scenes:
            logger.warning("No scenes found for reclassification")
            return
            
        logger.info(f"📊 Found {len(scenes)} scenes to reclassify")
        
        for i, scene in enumerate(scenes):
            try:
                scene_id = scene["scene_id"]
                image_url = scene["image_url"]
                houzz_id = scene["houzz_id"]
                
                logger.info(f"🔍 Reclassifying scene {i+1}/{len(scenes)}: {houzz_id}")
                
                # Step 1: Get enhanced classification
                # Extract caption from existing metadata or use houzz_id
                existing_metadata = scene.get("metadata", {})
                caption = existing_metadata.get("caption") or houzz_id
                
                new_classification = await classify_image_type(image_url, caption)
                
                # Step 2: Update database with new classification
                update_data = {
                    "image_type": new_classification["image_type"],
                    "is_primary_object": new_classification["is_primary_object"], 
                    "primary_category": new_classification["primary_category"],
                    "metadata": {
                        **existing_metadata,
                        "reclassification": {
                            "job_id": job_id,
                            "timestamp": datetime.utcnow().isoformat(),
                            "confidence": new_classification["confidence"],
                            "reason": new_classification["reason"],
                            "previous_type": scene.get("image_type", "unknown")
                        }
                    },
                    "updated_at": datetime.utcnow().isoformat()
                }
                
                result = supabase.table("scenes").update(update_data).eq("scene_id", scene_id).execute()
                
                if result.data:
                    old_type = scene.get("image_type", "unknown")
                    new_type = new_classification["image_type"]
                    confidence = new_classification["confidence"]
                    
                    if old_type != new_type:
                        logger.info(f"✅ Reclassified {houzz_id}: {old_type} → {new_type} (confidence: {confidence:.2f})")
                    else:
                        logger.info(f"✅ Confirmed {houzz_id}: {new_type} (confidence: {confidence:.2f})")
                else:
                    logger.error(f"❌ Failed to update scene {houzz_id}")
                    
                # Step 3: Re-run object detection if requested and classification changed significantly
                if force_redetection and new_classification["confidence"] > 0.7:
                    try:
                        # Run enhanced detection for better object understanding
                        detections = await run_detection_pipeline(image_url, f"{job_id}_redetect_{i}")
                        if detections:
                            logger.info(f"🔍 Re-detected {len(detections)} objects for {houzz_id}")
                    except Exception as detection_error:
                        logger.warning(f"Re-detection failed for {houzz_id}: {detection_error}")
                
            except Exception as scene_error:
                logger.error(f"❌ Failed to reclassify scene {i+1}: {scene_error}")
                continue
                
        logger.info(f"✅ Scene reclassification job {job_id} completed - processed {len(scenes)} scenes")
        
    except Exception as e:
        logger.error(f"❌ Scene reclassification job {job_id} failed: {e}")

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

def get_comprehensive_keywords() -> Dict[str, List[str]]:
    """
    Comprehensive keyword system for robust image classification.
    Covers furniture, decor, room types, styles, and contextual indicators.
    """
    return {
        # OBJECT-ONLY INDICATORS (Single furniture/decor pieces)
        "object": [
            # === SEATING ===
            "sofa", "couch", "sectional", "loveseat", "settee", "chesterfield", "daybed",
            "chair", "armchair", "accent chair", "lounge chair", "dining chair", "desk chair",
            "office chair", "swivel chair", "recliner", "wingback", "bergere", "club chair",
            "stool", "bar stool", "counter stool", "ottoman", "pouf", "footstool",
            "bench", "storage bench", "entryway bench", "window bench", "piano bench",
            
            # === TABLES ===
            "table", "coffee table", "cocktail table", "end table", "side table", "accent table",
            "dining table", "kitchen table", "breakfast table", "console table", "entry table",
            "desk", "writing desk", "computer desk", "standing desk", "secretary desk",
            "nightstand", "bedside table", "night table", "bedstand",
            "nesting tables", "tv stand", "media console", "entertainment center",
            
            # === STORAGE ===
            "bookshelf", "bookcase", "shelf", "shelving unit", "etagere", "ladder shelf",
            "cabinet", "storage cabinet", "display cabinet", "china cabinet", "curio cabinet",
            "dresser", "chest of drawers", "tall dresser", "low dresser", "bachelor chest",
            "wardrobe", "armoire", "closet", "credenza", "sideboard", "buffet", "hutch",
            "filing cabinet", "storage unit", "modular storage", "cube organizer",
            
            # === LIGHTING ===
            "lamp", "table lamp", "desk lamp", "task lamp", "reading lamp", "accent lamp",
            "floor lamp", "torchiere", "arc lamp", "tripod lamp", "tree lamp",
            "pendant light", "hanging light", "chandelier", "ceiling light", "flush mount",
            "wall sconce", "wall light", "vanity light", "picture light", "under cabinet light",
            "track lighting", "recessed light", "can light", "spotlight", "downlight",
            
            # === BEDROOM ===
            "bed", "bed frame", "platform bed", "sleigh bed", "canopy bed", "four poster",
            "headboard", "footboard", "mattress", "box spring", "bedding", "pillows",
            "comforter", "duvet", "blanket", "throw", "bed skirt", "mattress pad",
            
            # === BATHROOM ===
            "bathtub", "tub", "freestanding tub", "clawfoot tub", "soaking tub", "jacuzzi",
            "shower", "walk-in shower", "shower stall", "shower door", "shower curtain",
            "vanity", "bathroom vanity", "sink vanity", "double vanity", "floating vanity",
            "toilet", "water closet", "bidet", "pedestal sink", "vessel sink",
            
            # === DECOR & ACCESSORIES ===
            "mirror", "wall mirror", "floor mirror", "vanity mirror", "decorative mirror",
            "artwork", "wall art", "painting", "print", "poster", "canvas", "framed art",
            "sculpture", "statue", "figurine", "decorative object", "vase", "bowl",
            "plant", "houseplant", "planter", "pot", "artificial plant", "tree", "fern",
            "rug", "area rug", "runner", "carpet", "mat", "doormat", "bath mat",
            "curtains", "drapes", "blinds", "shades", "window treatments", "valance",
            "pillow", "throw pillow", "accent pillow", "cushion", "bolster",
            "clock", "wall clock", "mantel clock", "desk clock", "alarm clock",
            
            # === PRODUCT/CATALOG INDICATORS ===
            "product", "item", "piece", "furniture piece", "accent piece",
            "single", "individual", "standalone", "isolated", "solo",
            "catalog", "listing", "for sale", "available", "buy now", "purchase",
            "studio", "white background", "neutral background", "clean background",
            "product photo", "catalog image", "stock photo", "commercial photo",
            "furniture store", "showroom", "retail", "brand new", "unused",
            
            # === SIZE/QUANTITY INDICATORS ===
            "one", "single piece", "individual item", "standalone piece",
            "compact", "small", "mini", "accent size", "apartment size",
            "full size", "standard size", "oversized", "large", "extra large",
            
            # === CONDITION/STYLE INDICATORS ===
            "new", "brand new", "unused", "mint condition", "like new",
            "vintage", "antique", "mid-century", "retro", "classic",
            "modern", "contemporary", "minimalist", "sleek", "clean lines",
        ],
        
        # SCENE INDICATORS (Full room contexts)
        "scene": [
            # === ROOM TYPES ===
            "room", "living room", "family room", "great room", "sitting room", "lounge",
            "bedroom", "master bedroom", "guest bedroom", "kids bedroom", "nursery", "boudoir",
            "kitchen", "galley kitchen", "eat-in kitchen", "chef's kitchen", "kitchenette",
            "dining room", "formal dining", "breakfast nook", "dinette", "banquette",
            "bathroom", "master bath", "guest bath", "powder room", "half bath", "spa bath",
            "office", "home office", "study", "den", "library", "workspace", "studio",
            "entryway", "foyer", "entry hall", "mudroom", "vestibule", "reception area",
            "hallway", "corridor", "passage", "stairway", "landing", "stair hall",
            "basement", "cellar", "finished basement", "recreation room", "game room",
            "attic", "loft", "converted space", "bonus room", "flex space",
            "closet", "walk-in closet", "master closet", "dressing room", "wardrobe room",
            "laundry room", "utility room", "mudroom", "pantry", "storage room",
            "garage", "workshop", "craft room", "hobby room", "sewing room",
            
            # === OUTDOOR SPACES ===
            "patio", "deck", "balcony", "terrace", "veranda", "porch", "sunroom",
            "garden", "backyard", "courtyard", "outdoor living", "alfresco",
            "pool area", "hot tub", "spa", "outdoor kitchen", "fire pit area",
            
            # === INTERIOR DESIGN CONCEPTS ===
            "interior", "interior design", "home decor", "decoration", "styling",
            "design", "room design", "space design", "layout", "floor plan",
            "makeover", "renovation", "remodel", "refresh", "redesign", "transformation",
            "home", "house", "residence", "dwelling", "apartment", "condo", "flat",
            "space", "living space", "personal space", "functional space", "open space",
            "vignette", "room setting", "lifestyle", "cozy", "inviting", "welcoming",
            
            # === DESIGN PROCESSES ===
            "before and after", "room reveal", "design reveal", "styled", "staged",
            "decorated", "furnished", "designed", "curated", "arranged", "organized",
            "coordinated", "matched", "complemented", "harmonized", "balanced",
            
            # === ARCHITECTURAL ELEMENTS ===
            "architecture", "architectural", "built-in", "millwork", "molding", "wainscoting",
            "ceiling", "coffered ceiling", "tray ceiling", "vaulted ceiling", "exposed beams",
            "wall", "accent wall", "feature wall", "gallery wall", "shiplap", "paneling",
            "floor", "flooring", "hardwood", "tile", "carpet", "area rug", "runner",
            "window", "windows", "natural light", "bay window", "french doors", "skylight",
            "fireplace", "mantel", "hearth", "fireplace surround", "built-in shelves",
            
            # === LIFESTYLE CONTEXTS ===
            "family home", "family living", "everyday living", "real life", "lived-in",
            "comfortable", "functional", "practical", "user-friendly", "livable",
            "entertaining", "hosting", "gathering", "socializing", "relaxing",
            "reading nook", "conversation area", "media room", "tv room", "game night",
        ],
        
        # HYBRID INDICATORS (Scenes with dominant focal pieces)
        "hybrid": [
            "showcase", "featured", "spotlight", "highlight", "centerpiece", "focal point",
            "statement piece", "accent piece", "hero piece", "show-stopper", "eye-catching",
            "dramatic", "bold", "striking", "standout", "conversation starter",
            "anchored by", "centered around", "built around", "designed around",
            "room features", "room highlights", "main attraction", "key element",
            "draws attention", "commands attention", "steals the show", "takes center stage",
        ],
        
        # STYLE KEYWORDS (For enhanced classification)
        "style": [
            # === TRADITIONAL STYLES ===
            "traditional", "classic", "timeless", "formal", "elegant", "refined",
            "victorian", "georgian", "colonial", "english country", "french country",
            "tuscan", "mediterranean", "spanish", "moroccan", "persian",
            
            # === MODERN STYLES ===
            "modern", "contemporary", "minimalist", "clean", "sleek", "streamlined",
            "mid-century modern", "mcm", "danish modern", "eames", "bauhaus",
            "scandinavian", "nordic", "hygge", "lagom", "scandi", "swedish", "danish",
            "industrial", "urban", "loft", "warehouse", "exposed brick", "concrete",
            
            # === ECLECTIC STYLES ===
            "eclectic", "bohemian", "boho", "boho chic", "maximalist", "collected",
            "vintage", "retro", "antique", "shabby chic", "farmhouse", "rustic",
            "coastal", "beach", "nautical", "tropical", "resort", "vacation",
            "art deco", "hollywood regency", "glam", "luxe", "opulent", "dramatic",
            
            # === EMERGING STYLES ===
            "transitional", "modern farmhouse", "contemporary farmhouse", "urban farmhouse",
            "industrial chic", "modern rustic", "casual luxury", "approachable luxury",
            "grandmillennial", "new traditional", "maximalist", "dark academia",
        ],
        
        # ROOM TYPE KEYWORDS (For room classification)
        "room_type": {
            "living_room": ["living", "family room", "great room", "sitting", "lounge", "parlor"],
            "bedroom": ["bedroom", "master bedroom", "guest bedroom", "nursery", "kids room"],
            "kitchen": ["kitchen", "galley", "eat-in kitchen", "chef's kitchen", "kitchenette"],
            "dining_room": ["dining", "breakfast nook", "dinette", "formal dining"],
            "bathroom": ["bathroom", "master bath", "powder room", "half bath", "spa bath"],
            "office": ["office", "home office", "study", "den", "library", "workspace"],
            "outdoor": ["patio", "deck", "balcony", "terrace", "garden", "outdoor living"],
        }
    }

def calculate_keyword_score(text: str, keywords: List[str]) -> float:
    """
    Calculate weighted keyword score with phrase matching and fuzzy matching.
    Gives higher scores for exact matches and multi-word phrases.
    """
    score = 0.0
    text_lower = text.lower()
    
    for keyword in keywords:
        keyword_lower = keyword.lower()
        
        # Exact phrase match (highest score)
        if keyword_lower in text_lower:
            # Multi-word phrases get higher scores
            word_count = len(keyword_lower.split())
            if word_count >= 3:
                score += 3.0  # "walk-in closet", "master bedroom"
            elif word_count == 2:
                score += 2.0  # "living room", "coffee table"
            else:
                score += 1.0  # "sofa", "chair"
        
        # Partial word matching for compound keywords
        elif "_" in keyword_lower or "-" in keyword_lower:
            # Check individual parts of compound words
            parts = keyword_lower.replace("_", " ").replace("-", " ").split()
            partial_matches = sum(1 for part in parts if part in text_lower)
            if partial_matches >= len(parts) * 0.6:  # At least 60% of parts match
                score += 0.5
        
        # Fuzzy matching for plurals and variations
        else:
            # Check for plural/singular variations
            if keyword_lower.endswith('s') and keyword_lower[:-1] in text_lower:
                score += 0.8
            elif not keyword_lower.endswith('s') and f"{keyword_lower}s" in text_lower:
                score += 0.8
            
            # Check for common variations
            variations = get_keyword_variations(keyword_lower)
            for variation in variations:
                if variation in text_lower:
                    score += 0.6
                    break
    
    return score

def get_keyword_variations(keyword: str) -> List[str]:
    """Generate common variations of keywords for fuzzy matching"""
    variations = []
    
    # Common furniture variations
    furniture_variations = {
        "sofa": ["couch", "sectional", "loveseat"],
        "chair": ["seating", "seat"],
        "table": ["desk", "surface"],
        "lamp": ["light", "lighting"],
        "cabinet": ["storage", "cupboard"],
        "mirror": ["looking glass", "reflective"],
        "rug": ["carpet", "mat"],
        "bed": ["sleeping", "mattress"],
        "shelf": ["shelving", "bookcase"]
    }
    
    if keyword in furniture_variations:
        variations.extend(furniture_variations[keyword])
    
    # Common room variations
    room_variations = {
        "living room": ["lounge", "sitting room", "family room"],
        "bedroom": ["sleeping room", "bed room"],
        "kitchen": ["cooking area", "galley"],
        "bathroom": ["bath", "washroom", "powder room"],
        "office": ["study", "workspace", "den"]
    }
    
    if keyword in room_variations:
        variations.extend(room_variations[keyword])
    
    return variations

def detect_room_type(text: str, room_type_keywords: Dict[str, List[str]]) -> Optional[str]:
    """Detect the primary room type from text analysis"""
    text_lower = text.lower()
    room_scores = {}
    
    for room_type, keywords in room_type_keywords.items():
        score = 0
        for keyword in keywords:
            if keyword.lower() in text_lower:
                # Longer phrases get higher scores
                score += len(keyword.split())
        
        if score > 0:
            room_scores[room_type] = score
    
    if room_scores:
        # Return room type with highest score
        return max(room_scores, key=room_scores.get)
    
    return None

def detect_primary_category_from_text(text: str) -> Optional[str]:
    """
    Enhanced primary category detection using comprehensive keyword matching.
    Returns the most likely furniture category based on text analysis.
    """
    text_lower = text.lower()
    category_scores = {}
    
    # Score each category in MODOMO_TAXONOMY
    for category_group, items in MODOMO_TAXONOMY.items():
        for item in items:
            item_variations = [
                item,
                item.replace("_", " "),
                item.replace("_", "-"),
                item.replace("_", "")
            ]
            
            # Add common variations
            variations = get_keyword_variations(item)
            item_variations.extend(variations)
            
            # Calculate score for this item
            item_score = 0
            for variation in item_variations:
                variation_lower = variation.lower()
                if variation_lower in text_lower:
                    # Exact match gets higher score
                    if variation_lower == item.replace("_", " "):
                        item_score += 3
                    else:
                        item_score += 1
                    
                    # Boost score for longer phrases
                    word_count = len(variation_lower.split())
                    if word_count >= 2:
                        item_score += word_count - 1
            
            if item_score > 0:
                category_scores[item] = item_score
    
    # Return category with highest score if above threshold
    if category_scores:
        best_category = max(category_scores, key=category_scores.get)
        best_score = category_scores[best_category]
        
        # Only return if confidence is reasonable
        if best_score >= 2:
            return best_category
    
    return None

def detect_styles_from_text(text: str, style_keywords: List[str]) -> List[str]:
    """Detect interior design styles mentioned in the text"""
    text_lower = text.lower()
    detected_styles = []
    
    for style in style_keywords:
        style_lower = style.lower()
        if style_lower in text_lower:
            detected_styles.append(style)
            
        # Check for style variations
        style_variations = {
            "mid-century modern": ["mcm", "mid century", "mid-century", "eames era"],
            "scandinavian": ["scandi", "nordic", "danish", "swedish", "hygge"],
            "industrial": ["urban loft", "warehouse", "exposed brick", "concrete"],
            "bohemian": ["boho", "boho chic", "eclectic", "maximalist"],
            "farmhouse": ["rustic", "country", "barn", "shiplap"],
            "contemporary": ["modern", "current", "up-to-date", "present-day"]
        }
        
        if style_lower in style_variations:
            for variation in style_variations[style_lower]:
                if variation in text_lower and style not in detected_styles:
                    detected_styles.append(style)
                    break
    
    return detected_styles[:3]  # Limit to top 3 styles

async def classify_image_type(image_url: str, caption: str = None) -> Dict[str, Any]:
    """
    Classify whether an image is a scene, object, product, or hybrid.
    
    Uses multiple heuristics for robust classification:
    1. Caption/filename analysis for keywords
    2. Object detection count and distribution
    3. Image composition analysis (if available)
    
    Returns:
        Dict with image_type, is_primary_object, primary_category, confidence, reason
    """
    try:
        # Default classification
        classification = {
            "image_type": "scene",
            "is_primary_object": False,
            "primary_category": None,
            "confidence": 0.5,
            "reason": "default_classification"
        }
        
        # Step 1: Caption/URL-based classification
        text_indicators = []
        if caption:
            text_indicators.append(caption.lower())
        if image_url:
            text_indicators.append(image_url.lower())
        
        combined_text = " ".join(text_indicators)
        
        # Comprehensive keyword classification system
        classification_keywords = get_comprehensive_keywords()
        
        object_keywords = classification_keywords["object"]
        scene_keywords = classification_keywords["scene"]
        hybrid_keywords = classification_keywords["hybrid"]
        style_keywords = classification_keywords["style"]
        room_type_keywords = classification_keywords["room_type"]
        
        # Enhanced scoring with weighted keywords and phrase matching
        object_score = calculate_keyword_score(combined_text, object_keywords)
        scene_score = calculate_keyword_score(combined_text, scene_keywords)
        hybrid_score = calculate_keyword_score(combined_text, hybrid_keywords)
        style_score = calculate_keyword_score(combined_text, style_keywords)
        
        # Detect room type for better context
        detected_room_type = detect_room_type(combined_text, room_type_keywords)
        
        # Adjust scene score if room type detected
        if detected_room_type:
            scene_score += 2  # Boost scene score if room type is clearly identified
        
        # Step 2: AI detection-based classification (if detector available)
        detection_classification = None
        if detector and image_url:
            try:
                # Download image temporarily for detection
                import aiohttp
                import tempfile
                import uuid
                
                temp_path = f"/tmp/classify_{uuid.uuid4().hex}.jpg"
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(image_url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                        if response.status == 200:
                            with open(temp_path, 'wb') as f:
                                async for chunk in response.content.iter_chunked(8192):
                                    f.write(chunk)
                            
                            # Run object detection
                            detections = await detector.detect_objects(temp_path, MODOMO_TAXONOMY)
                            
                            # Clean up
                            if os.path.exists(temp_path):
                                os.remove(temp_path)
                            
                            if detections:
                                detection_classification = analyze_detections_for_classification(detections)
                            
            except Exception as e:
                logger.warning(f"Detection-based classification failed: {e}")
        
        # Step 3: Combine heuristics for final classification
        if detection_classification:
            # Use AI detection results as primary signal
            classification.update(detection_classification)
            classification["confidence"] = min(0.9, classification["confidence"] + 0.2)
            classification["reason"] = f"ai_detection_{detection_classification.get('reason', 'unknown')}"
            
        elif object_score > scene_score and object_score >= 2.0:
            # Strong object indicators detected
            classification.update({
                "image_type": "object",
                "is_primary_object": True,
                "confidence": min(0.9, 0.6 + (object_score * 0.05)),
                "reason": f"text_analysis_object_score_{object_score:.1f}"
            })
            
            # Enhanced primary category detection
            primary_category = detect_primary_category_from_text(combined_text)
            if primary_category:
                classification["primary_category"] = primary_category
                classification["confidence"] += 0.1
                    
        elif scene_score >= 1.5 or detected_room_type:
            # Strong scene indicators or room type detected
            image_type = "scene"
            
            # Check for hybrid classification
            if hybrid_score >= 1.0 and object_score >= 1.0:
                image_type = "hybrid"
            
            classification.update({
                "image_type": image_type,
                "is_primary_object": image_type == "hybrid",
                "confidence": min(0.9, 0.6 + (scene_score * 0.03)),
                "reason": f"text_analysis_scene_score_{scene_score:.1f}_room_{detected_room_type or 'unknown'}"
            })
            
            # For hybrid images, try to detect primary category
            if image_type == "hybrid":
                primary_category = detect_primary_category_from_text(combined_text)
                if primary_category:
                    classification["primary_category"] = primary_category
                    
        elif style_score >= 2.0:
            # Strong style indicators suggest interior design content
            classification.update({
                "image_type": "scene",
                "confidence": min(0.8, 0.5 + (style_score * 0.02)),
                "reason": f"style_analysis_score_{style_score:.1f}"
            })
            
        # Add enhanced metadata to classification result
        classification["metadata"] = {
            "scores": {
                "object": round(object_score, 1),
                "scene": round(scene_score, 1), 
                "hybrid": round(hybrid_score, 1),
                "style": round(style_score, 1)
            },
            "detected_room_type": detected_room_type,
            "detected_styles": detect_styles_from_text(combined_text, style_keywords),
            "keyword_matches": {
                "object_matches": [kw for kw in object_keywords if kw.lower() in combined_text],
                "scene_matches": [kw for kw in scene_keywords if kw.lower() in combined_text]
            }
        }
        
        logger.info(f"Image classification: {classification['image_type']} (confidence: {classification['confidence']:.2f}, reason: {classification['reason']}, room: {detected_room_type})")
        return classification
        
    except Exception as e:
        logger.error(f"Image classification failed: {e}")
        return {
            "image_type": "scene",
            "is_primary_object": False, 
            "primary_category": None,
            "confidence": 0.3,
            "reason": f"classification_error_{str(e)[:50]}"
        }

def analyze_detections_for_classification(detections: List[Dict]) -> Dict[str, Any]:
    """
    Analyze object detections to classify image type.
    
    Classification logic:
    - object: 1-2 high-confidence objects, similar category
    - scene: 3+ objects, diverse categories
    - hybrid: 1 dominant object + background elements
    """
    if not detections:
        return {"image_type": "scene", "confidence": 0.4, "reason": "no_detections"}
    
    # Filter high-confidence detections
    high_conf_detections = [d for d in detections if d.get("confidence", 0) > 0.7]
    
    num_detections = len(detections)
    num_high_conf = len(high_conf_detections)
    
    # Get categories
    categories = [d.get("category", "unknown") for d in detections]
    unique_categories = set(categories)
    
    # Find dominant category
    from collections import Counter
    category_counts = Counter(categories)
    dominant_category, dominant_count = category_counts.most_common(1)[0] if category_counts else ("unknown", 0)
    
    # Classification logic
    if num_detections == 1:
        return {
            "image_type": "object",
            "is_primary_object": True,
            "primary_category": dominant_category,
            "confidence": 0.85,
            "reason": "single_detection"
        }
    elif num_detections == 2 and len(unique_categories) <= 2:
        return {
            "image_type": "object",
            "is_primary_object": True,
            "primary_category": dominant_category,
            "confidence": 0.75,
            "reason": "two_similar_objects"
        }
    elif num_high_conf >= 3 and len(unique_categories) >= 3:
        return {
            "image_type": "scene",
            "is_primary_object": False,
            "primary_category": None,
            "confidence": 0.8,
            "reason": "diverse_objects_scene"
        }
    elif dominant_count >= num_detections * 0.6:
        # One category dominates
        return {
            "image_type": "hybrid",
            "is_primary_object": True,
            "primary_category": dominant_category,
            "confidence": 0.7,
            "reason": "dominant_category_hybrid"
        }
    else:
        return {
            "image_type": "scene",
            "is_primary_object": False,
            "primary_category": None,
            "confidence": 0.6,
            "reason": "mixed_detection_scene"
        }

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
                
                # Check R2 credentials before attempting upload
                r2_endpoint = os.getenv('CLOUDFLARE_R2_ENDPOINT')
                r2_access_key = os.getenv('CLOUDFLARE_R2_ACCESS_KEY_ID')
                r2_secret_key = os.getenv('CLOUDFLARE_R2_SECRET_ACCESS_KEY')
                r2_bucket = os.getenv('CLOUDFLARE_R2_BUCKET', 'reroom')
                
                # Construct proper R2 public URL
                # Use custom domain if configured, otherwise use R2 native URL
                r2_custom_domain = os.getenv('CLOUDFLARE_R2_PUBLIC_DOMAIN')  # e.g., photos.reroom.app
                if r2_custom_domain:
                    r2_url = f"https://{r2_custom_domain}/{r2_key}"
                elif r2_endpoint:
                    # Extract account ID from endpoint to build public URL
                    # R2 endpoint format: https://{account_id}.r2.cloudflarestorage.com
                    if 'r2.cloudflarestorage.com' in r2_endpoint:
                        account_id = r2_endpoint.split('//')[1].split('.')[0]
                        r2_url = f"https://{account_id}.r2.cloudflarestorage.com/{r2_bucket}/{r2_key}"
                    else:
                        # Fallback for custom endpoints
                        r2_url = f"{r2_endpoint.rstrip('/')}/{r2_bucket}/{r2_key}"
                else:
                    r2_url = f"https://storage.googleapis.com/{r2_bucket}/{r2_key}"  # Generic fallback
                
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

                # Step 3: Classify image type and store scene in database with R2 references
                if supabase:
                    try:
                        # Pre-classify image type based on available data
                        image_classification = await classify_image_type(image_url, caption)
                        
                        # Build scene data with only columns that exist in Supabase
                        scene_data = {
                            "houzz_id": scene_id,
                            "image_url": image_url,
                            "image_r2_key": r2_key,
                            # TODO: Add these columns back when Supabase schema is updated:
                            # "image_type": image_classification["image_type"],
                            # "is_primary_object": image_classification["is_primary_object"], 
                            # "primary_category": image_classification["primary_category"],
                            "room_type": room_type,
                            "style_tags": [caption] if caption else [],
                            "color_tags": [],
                            "status": "scraped",
                            "metadata": {
                                "classification_confidence": image_classification["confidence"],
                                "classification_reason": image_classification["reason"],
                                # Store the classification data in metadata for now
                                "image_type": image_classification["image_type"],
                                "is_primary_object": image_classification["is_primary_object"],
                                "primary_category": image_classification["primary_category"]
                            }
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
                                
                                # Handle mask URLs and R2 keys from mask_path
                                mask_url = None
                                mask_r2_key = None
                                mask_path = detection.get("mask_path")
                                if mask_path and os.path.exists(mask_path):
                                    # Generate both URL for frontend and R2 key for storage
                                    mask_filename = os.path.basename(mask_path)
                                    mask_url = f"/masks/{mask_filename}"  # Public URL for frontend
                                    mask_r2_key = f"masks/{mask_filename}"  # Storage key for R2
                                    logger.info(f"Generated mask URL: {mask_url} and R2 key: {mask_r2_key} for path: {mask_path}")
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
                                    "mask_url": mask_url,  # Public URL for frontend access
                                    "mask_r2_key": mask_r2_key  # Storage key for R2
                                }
                                
                                # Ensure all data is JSON serializable - apply recursively to catch all nested values
                                object_data = make_json_serializable(object_data)
                                
                                # Keep mask_url for frontend access and mask_r2_key for storage
                                # Both fields are needed for proper frontend functionality
                                
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
                    # Use the same URL construction logic as above
                    r2_custom_domain = os.getenv('CLOUDFLARE_R2_PUBLIC_DOMAIN')
                    r2_endpoint = os.getenv('CLOUDFLARE_R2_ENDPOINT')
                    r2_bucket = os.getenv('CLOUDFLARE_R2_BUCKET', 'reroom')
                    
                    if r2_custom_domain:
                        public_url = f"https://{r2_custom_domain}/{scene['image_r2_key']}"
                    elif r2_endpoint and 'r2.cloudflarestorage.com' in r2_endpoint:
                        account_id = r2_endpoint.split('//')[1].split('.')[0]
                        public_url = f"https://{account_id}.r2.cloudflarestorage.com/{r2_bucket}/{scene['image_r2_key']}"
                    else:
                        public_url = f"https://storage.googleapis.com/{r2_bucket}/{scene['image_r2_key']}"
                        
                    scene_data["r2_storage"] = {
                        "key": scene["image_r2_key"],
                        "public_url": public_url,
                        "download_url": public_url
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
                approved, matched_product_id, metadata, mask_r2_key, mask_url
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