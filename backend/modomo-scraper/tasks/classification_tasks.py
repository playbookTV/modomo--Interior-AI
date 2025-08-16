"""
Background tasks for image classification and scene analysis
"""
from datetime import datetime
from typing import Dict, List, Optional, Any
import structlog

logger = structlog.get_logger(__name__)


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
            
            # === INTERIOR DESIGN CONCEPTS ===
            "interior", "interior design", "home decor", "decoration", "styling",
            "design", "room design", "space design", "layout", "floor plan",
            "makeover", "renovation", "remodel", "refresh", "redesign", "transformation",
            "home", "house", "residence", "dwelling", "apartment", "condo", "flat",
            "space", "living space", "personal space", "functional space", "open space",
            "vignette", "room setting", "lifestyle", "cozy", "inviting", "welcoming",
            
            # === ARCHITECTURAL ELEMENTS ===
            "architecture", "architectural", "built-in", "millwork", "molding", "wainscoting",
            "ceiling", "coffered ceiling", "tray ceiling", "vaulted ceiling", "exposed beams",
            "wall", "accent wall", "feature wall", "gallery wall", "shiplap", "paneling",
            "floor", "flooring", "hardwood", "tile", "carpet", "area rug", "runner",
            "window", "windows", "natural light", "bay window", "french doors", "skylight",
            "fireplace", "mantel", "hearth", "fireplace surround", "built-in shelves",
        ],
        
        # STYLE KEYWORDS (For enhanced classification)
        "style": [
            "traditional", "classic", "timeless", "formal", "elegant", "refined",
            "modern", "contemporary", "minimalist", "clean", "sleek", "streamlined",
            "mid-century modern", "mcm", "danish modern", "scandinavian", "nordic",
            "industrial", "urban", "loft", "warehouse", "exposed brick", "concrete",
            "bohemian", "boho", "eclectic", "vintage", "farmhouse", "rustic", "coastal",
        ]
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
    
    return score


async def classify_image_type(image_url: str, caption: Optional[str] = None) -> Dict[str, Any]:
    """
    Enhanced image classification using comprehensive keyword analysis.
    Determines if image is a scene (full room) or object (individual furniture piece).
    """
    try:
        # Combine all available text for analysis
        analysis_text = ""
        if caption:
            analysis_text += f" {caption}"
        if image_url:
            analysis_text += f" {image_url}"
        
        analysis_text = analysis_text.strip()
        
        if not analysis_text:
            return {
                "image_type": "unknown",
                "is_primary_object": False,
                "primary_category": None,
                "confidence": 0.0,
                "reason": "No text available for analysis"
            }
        
        # Get keyword categories
        keywords = get_comprehensive_keywords()
        
        # Calculate scores for each category
        object_score = calculate_keyword_score(analysis_text, keywords["object"])
        scene_score = calculate_keyword_score(analysis_text, keywords["scene"])
        style_score = calculate_keyword_score(analysis_text, keywords["style"])
        
        # Determine image type based on scores
        total_score = object_score + scene_score + style_score
        
        if total_score == 0:
            return {
                "image_type": "unknown",
                "is_primary_object": False,
                "primary_category": None,
                "confidence": 0.0,
                "reason": "No relevant keywords found"
            }
        
        # Classification logic
        if object_score > scene_score * 1.5:  # Strong object indicators
            image_type = "object"
            is_primary_object = True
            confidence = min(object_score / (object_score + scene_score), 0.95)
            reason = f"Strong object indicators (score: {object_score:.1f} vs scene: {scene_score:.1f})"
        elif scene_score > object_score * 1.2:  # Strong scene indicators
            image_type = "scene"
            is_primary_object = False
            confidence = min(scene_score / (object_score + scene_score), 0.95)
            reason = f"Strong scene indicators (score: {scene_score:.1f} vs object: {object_score:.1f})"
        elif object_score > scene_score:  # Slight object preference
            image_type = "hybrid"  # Could be a scene focused on one piece
            is_primary_object = True
            confidence = 0.6 + (object_score - scene_score) / total_score * 0.2
            reason = f"Slight object preference (object: {object_score:.1f}, scene: {scene_score:.1f})"
        else:  # Default to scene
            image_type = "scene"
            is_primary_object = False
            confidence = 0.5 + (scene_score - object_score) / total_score * 0.2
            reason = f"Default to scene (object: {object_score:.1f}, scene: {scene_score:.1f})"
        
        # Try to detect primary category for objects
        primary_category = None
        if is_primary_object or image_type == "object":
            primary_category = detect_primary_category_from_text(analysis_text)
        
        # Ensure confidence is reasonable
        confidence = max(0.1, min(confidence, 0.95))
        
        return {
            "image_type": image_type,
            "is_primary_object": is_primary_object,
            "primary_category": primary_category,
            "confidence": confidence,
            "reason": reason,
            "scores": {
                "object": object_score,
                "scene": scene_score,
                "style": style_score
            }
        }
        
    except Exception as e:
        logger.error(f"Image classification failed: {e}")
        return {
            "image_type": "unknown",
            "is_primary_object": False,
            "primary_category": None,
            "confidence": 0.0,
            "reason": f"Classification error: {str(e)}"
        }


def detect_primary_category_from_text(text: str) -> Optional[str]:
    """
    Enhanced primary category detection using comprehensive keyword matching.
    Returns the most likely furniture category based on text analysis.
    """
    from config.taxonomy import MODOMO_TAXONOMY
    
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


# Celery task functions
from celery_app import celery_app
from tasks import BaseTask, database_service

@celery_app.task(bind=True, autoretry_for=(Exception,), retry_kwargs={'max_retries': 3, 'countdown': 60})
def run_scene_reclassification_job(self, job_id: str, limit: int, force_reclassify: bool = False):
    """Background job to reclassify existing scenes"""
    processed = 0
    total_scenes = 0
    reclassified = 0
    
    try:
        logger.info(f"🔄 Starting scene reclassification job {job_id} for {limit} scenes")
        BaseTask.update_job_progress(job_id, "running", 0, limit, "Getting scenes for reclassification...")
        
        # Get scenes that need reclassification
        scenes_data = get_scenes_for_reclassification(limit, force_reclassify)
        total_scenes = len(scenes_data)
        
        if total_scenes == 0:
            logger.info(f"No scenes found for reclassification")
            return BaseTask.complete_job(job_id, 0, 0, {"message": "No scenes need reclassification"})
        
        logger.info(f"Found {total_scenes} scenes for reclassification")
        BaseTask.update_job_progress(job_id, "running", 0, total_scenes, f"Reclassifying {total_scenes} scenes...")
        
        # Process each scene
        for i, scene in enumerate(scenes_data):
            try:
                scene_id = scene["scene_id"]
                image_url = scene["image_url"]
                
                # Get current classification metadata
                current_metadata = scene.get("metadata", {})
                old_classification = {
                    "image_type": current_metadata.get("image_type", "unknown"),
                    "is_primary_object": current_metadata.get("is_primary_object", False),
                    "primary_category": current_metadata.get("primary_category")
                }
                
                # Run classification
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                new_classification = loop.run_until_complete(classify_image_type(image_url))
                loop.close()
                
                # Check if classification changed
                classification_changed = (
                    new_classification["image_type"] != old_classification["image_type"] or
                    new_classification.get("is_primary_object", False) != old_classification["is_primary_object"] or
                    new_classification.get("primary_category") != old_classification["primary_category"]
                )
                
                if classification_changed or force_reclassify:
                    # Update scene with new classification
                    updated_metadata = current_metadata.copy()
                    updated_metadata.update({
                        "image_type": new_classification["image_type"],
                        "is_primary_object": new_classification.get("is_primary_object", False),
                        "primary_category": new_classification.get("primary_category"),
                        "classification_confidence": new_classification.get("confidence", 0.0),
                        "classification_reason": new_classification.get("reason", "unknown"),
                        "reclassified_at": datetime.utcnow().isoformat(),
                        "previous_classification": old_classification
                    })
                    
                    # Update in database
                    success = update_scene_classification(scene_id, updated_metadata)
                    
                    if success:
                        reclassified += 1
                        logger.debug(f"✅ Reclassified scene {scene_id}: {old_classification['image_type']} → {new_classification['image_type']}")
                    else:
                        logger.warning(f"⚠️ Failed to update classification for scene {scene_id}")
                else:
                    logger.debug(f"🔄 No classification change for scene {scene_id}")
                
                processed += 1
                
                # Update progress
                progress_message = f"Processed {processed}/{total_scenes} scenes ({reclassified} reclassified)"
                BaseTask.update_job_progress(job_id, "running", processed, total_scenes, progress_message)
                BaseTask.update_celery_progress(processed, total_scenes, progress_message)
                
                # Log progress every 25 scenes
                if (i + 1) % 25 == 0:
                    logger.info(f"Reclassification progress: {processed}/{total_scenes} scenes, {reclassified} reclassified")
                    
            except Exception as scene_error:
                logger.error(f"Error reclassifying scene {scene.get('scene_id', 'unknown')}: {scene_error}")
                processed += 1
                continue
        
        # Complete the job
        result = {
            "processed": processed,
            "total": total_scenes,
            "reclassified": reclassified,
            "unchanged": processed - reclassified,
            "reclassification_rate": f"{(reclassified/processed)*100:.1f}%" if processed > 0 else "0%",
            "force_reclassify": force_reclassify,
            "message": f"Scene reclassification completed: {reclassified}/{processed} scenes reclassified"
        }
        
        logger.info(f"🔄 Scene reclassification job {job_id} completed: {reclassified}/{processed} reclassified")
        return BaseTask.complete_job(job_id, processed, total_scenes, result)
        
    except Exception as e:
        logger.error(f"❌ Scene reclassification job {job_id} failed: {e}")
        BaseTask.handle_task_error(job_id, e, processed, total_scenes or limit)
        raise

@celery_app.task(bind=True)
def classify_single_scene(self, scene_id: str, image_url: str):
    """Classify a single scene image"""
    try:
        logger.info(f"🔍 Classifying scene {scene_id}")
        
        # Run classification
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        classification = loop.run_until_complete(classify_image_type(image_url))
        loop.close()
        
        # Update scene in database
        metadata = {
            "image_type": classification["image_type"],
            "is_primary_object": classification.get("is_primary_object", False),
            "primary_category": classification.get("primary_category"),
            "classification_confidence": classification.get("confidence", 0.0),
            "classification_reason": classification.get("reason", "unknown"),
            "classified_at": datetime.utcnow().isoformat()
        }
        
        success = update_scene_classification(scene_id, metadata)
        
        if success:
            logger.info(f"✅ Successfully classified scene {scene_id}: {classification['image_type']}")
            return {
                "success": True,
                "scene_id": scene_id,
                "classification": classification
            }
        else:
            logger.error(f"❌ Failed to update classification for scene {scene_id}")
            return {
                "success": False,
                "scene_id": scene_id,
                "error": "Database update failed"
            }
            
    except Exception as e:
        logger.error(f"❌ Single scene classification failed for {scene_id}: {e}")
        return {
            "success": False,
            "scene_id": scene_id,
            "error": str(e)
        }

def get_scenes_for_reclassification(limit: int, force_reclassify: bool = False) -> List[Dict[str, Any]]:
    """Get scenes that need reclassification"""
    try:
        if not database_service:
            return []
        
        query = database_service.supabase.table("scenes").select(
            "scene_id, image_url, metadata"
        )
        
        if not force_reclassify:
            # Only get scenes without classification metadata
            query = query.is_("metadata->image_type", "null")
        
        result = query.order("created_at", desc=True).limit(limit).execute()
        
        return result.data or []
        
    except Exception as e:
        logger.error(f"Failed to get scenes for reclassification: {e}")
        return []

def update_scene_classification(scene_id: str, metadata: Dict[str, Any]) -> bool:
    """Update scene with classification metadata"""
    try:
        if not database_service:
            return False
        
        # Get current metadata and merge
        current_scene = database_service.supabase.table("scenes").select(
            "metadata"
        ).eq("scene_id", scene_id).execute()
        
        current_metadata = {}
        if current_scene.data:
            current_metadata = current_scene.data[0].get("metadata", {})
        
        # Merge metadata
        updated_metadata = current_metadata.copy()
        updated_metadata.update(metadata)
        
        # Update scene
        result = database_service.supabase.table("scenes").update({
            "metadata": updated_metadata
        }).eq("scene_id", scene_id).execute()
        
        return bool(result.data)
        
    except Exception as e:
        logger.error(f"Failed to update scene classification for {scene_id}: {e}")
        return False