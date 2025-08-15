"""
YOLOv8 + DETR multi-model object detection for interior design scenes
Primary: YOLOv8 for fast, accurate furniture detection
Fallback: DETR-ResNet-50 for comprehensive object detection
"""

import torch
from PIL import Image
from typing import List, Dict, Any
import numpy as np
from transformers import pipeline
import structlog

logger = structlog.get_logger()

class GroundingDINODetector:
    """Multi-model object detector with YOLOv8 primary and DETR fallback"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing multi-model detector on {self.device}")
        
        # Primary detector: YOLOv8
        self.yolo_detector = None
        self.yolo_available = False
        
        # Fallback detector: DETR-ResNet-50
        self.detr_detector = None
        self.detr_available = False
        
        # Initialize detectors
        self._init_yolo_detector()
        self._init_detr_detector()
        
        # Confidence thresholds
        self.yolo_confidence_threshold = 0.25  # YOLO default is good
        self.detr_confidence_threshold = 0.15  # Lower for DETR fallback
        self.min_box_area = 100  # Minimum bounding box area to filter tiny objects
    
    def _init_yolo_detector(self):
        """Initialize YOLOv8 detector as primary"""
        try:
            # Try to import and initialize YOLO
            from ultralytics import YOLO
            
            # Use YOLOv8n (nano) for speed, or YOLOv8s (small) for better accuracy
            # YOLOv8n is fastest, YOLOv8s is good balance, YOLOv8m/l/x are slower but more accurate
            self.yolo_detector = YOLO("yolov8s.pt")  # Will download on first use
            self.yolo_available = True
            logger.info("✅ YOLOv8 detector initialized successfully (primary)")
            
        except ImportError:
            logger.warning("⚠️ ultralytics not available - YOLO detector disabled")
            self.yolo_available = False
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize YOLO detector: {e}")
            self.yolo_available = False
    
    def _init_detr_detector(self):
        """Initialize DETR detector as fallback with proper parameter loading"""
        try:
            # Import required components explicitly to control loading
            from transformers import DetrForObjectDetection, DetrImageProcessor
            import warnings
            
            # Suppress the specific meta parameter warnings during model loading
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter.*")
                warnings.filterwarnings("ignore", message=".*Missing keys.*discovered while loading pretrained weights.*")
                
                # Load model and processor explicitly with proper device handling
                model = DetrForObjectDetection.from_pretrained(
                    "facebook/detr-resnet-50",
                    torch_dtype=torch.float32,
                    device_map=None  # We'll handle device placement manually
                )
                processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
                
                # Move to device after loading to avoid meta parameter issues
                model = model.to(self.device)
                model.eval()  # Set to evaluation mode
                
                # Create pipeline with pre-loaded components
                self.detr_detector = pipeline(
                    "object-detection",
                    model=model,
                    image_processor=processor,
                    device=0 if self.device == "cuda" else -1
                )
                
            self.detr_available = True
            logger.info("✅ DETR detector initialized successfully (fallback) - warnings suppressed")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize DETR detector: {e}")
            self.detr_available = False
        
    async def detect_objects(self, image_path: str, taxonomy: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """
        Detect objects using YOLOv8 (primary) with DETR fallback
        
        Args:
            image_path: Path to the image file
            taxonomy: Dictionary mapping categories to object types
            
        Returns:
            List of detected objects with bounding boxes and categories
        """
        results = []
        
        # Try YOLOv8 first (primary detector)
        if self.yolo_available:
            try:
                results = await self._detect_with_yolo(image_path, taxonomy)
                if results:
                    logger.info(f"✅ YOLOv8 detected {len(results)} objects in {image_path}")
                    return results
                else:
                    logger.info("⚠️ YOLOv8 found no objects, trying DETR fallback")
            except Exception as e:
                logger.warning(f"⚠️ YOLOv8 detection failed: {e}, trying DETR fallback")
        
        # Fallback to DETR if YOLO unavailable or failed
        if self.detr_available:
            try:
                results = await self._detect_with_detr(image_path, taxonomy)
                if results:
                    logger.info(f"✅ DETR detected {len(results)} objects in {image_path}")
                else:
                    logger.warning(f"⚠️ No objects detected by any detector in {image_path}")
            except Exception as e:
                logger.error(f"❌ DETR detection also failed: {e}")
        
        if not self.yolo_available and not self.detr_available:
            logger.error("❌ No detectors available - both YOLOv8 and DETR failed to initialize")
        
        return results
    
    async def _detect_with_yolo(self, image_path: str, taxonomy: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """Run detection with YOLOv8"""
        image = Image.open(image_path).convert("RGB")
        
        # Run YOLOv8 detection
        yolo_results = self.yolo_detector(image, conf=self.yolo_confidence_threshold, verbose=False)
        
        results = []
        for result in yolo_results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Extract box data
                    xyxy = box.xyxy[0].cpu().numpy()  # x1, y1, x2, y2
                    conf = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())
                    
                    # Get class name
                    class_name = self.yolo_detector.names[cls_id]
                    
                    # Convert to our format [x, y, width, height]
                    x1, y1, x2, y2 = xyxy
                    width, height = x2 - x1, y2 - y1
                    bbox = [float(x1), float(y1), float(width), float(height)]
                    
                    # Filter tiny objects
                    box_area = width * height
                    if box_area < self.min_box_area:
                        continue
                    
                    # Map to taxonomy
                    mapped_category = self._map_to_taxonomy(class_name, taxonomy)
                    if mapped_category:
                        # Generate comprehensive tags
                        tags = self._generate_comprehensive_tags(
                            mapped_category=mapped_category,
                            raw_label=class_name,
                            confidence=conf,
                            box_area=box_area,
                            detector='yolov8',
                            taxonomy=taxonomy
                        )
                        
                        results.append({
                            'bbox': bbox,
                            'category': mapped_category,
                            'confidence': conf,
                            'raw_label': class_name,
                            'tags': tags,
                            'box_area': int(box_area),
                            'detector': 'yolov8'
                        })
        
        return results
    
    async def _detect_with_detr(self, image_path: str, taxonomy: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """Run detection with DETR (fallback)"""
        image = Image.open(image_path).convert("RGB")
        
        # Run DETR detection
        detections = self.detr_detector(image)
        
        # Filter and format results with improved logic
        results = []
        
        # Sort detections by confidence (highest first)
        sorted_detections = sorted(detections, key=lambda x: x['score'], reverse=True)
        
        for detection in sorted_detections:
            if detection['score'] >= self.detr_confidence_threshold:
                # Map detected label to our taxonomy
                mapped_category = self._map_to_taxonomy(detection['label'], taxonomy)
                
                if mapped_category:
                    # Convert box format [x_min, y_min, x_max, y_max] to [x, y, width, height]
                    box = detection['box']
                    width = box['xmax'] - box['xmin']
                    height = box['ymax'] - box['ymin']
                    bbox = [box['xmin'], box['ymin'], width, height]
                    
                    # Filter out tiny objects
                    box_area = width * height
                    if box_area < self.min_box_area:
                        continue
                    
                    # Allow multiple objects of same category but limit to avoid spam
                    category_count = sum(1 for r in results if r['category'] == mapped_category)
                    if category_count >= 3:  # Max 3 of same category
                        continue
                    
                    # Generate comprehensive tags
                    tags = self._generate_comprehensive_tags(
                        mapped_category=mapped_category,
                        raw_label=detection['label'],
                        confidence=float(detection['score']),
                        box_area=box_area,
                        detector='detr',
                        taxonomy=taxonomy
                    )
                    
                    results.append({
                        'bbox': bbox,
                        'category': mapped_category,
                        'confidence': float(detection['score']),
                        'raw_label': detection['label'],
                        'tags': tags,
                        'box_area': int(box_area),
                        'detector': 'detr'
                    })
        
        # If we found very few objects, run a second pass with lower threshold
        if len(results) < 3:
            logger.info(f"Only found {len(results)} objects with DETR, running second pass with lower threshold")
            for detection in sorted_detections:
                if detection['score'] >= 0.1 and len(results) < 8:  # Even lower threshold
                    mapped_category = self._map_to_taxonomy(detection['label'], taxonomy)
                    if mapped_category and not any(r['raw_label'] == detection['label'] for r in results):
                        box = detection['box']
                        width = box['xmax'] - box['xmin']
                        height = box['ymax'] - box['ymin']
                        box_area = width * height
                        
                        if box_area >= self.min_box_area // 2:  # More lenient area requirement
                            bbox = [box['xmin'], box['ymin'], width, height]
                            results.append({
                                'bbox': bbox,
                                'category': mapped_category,
                                'confidence': float(detection['score']),
                                'raw_label': detection['label'],
                                'tags': [mapped_category, 'detr', 'low_confidence'],
                                'box_area': int(box_area),
                                'detector': 'detr'
                            })
        
        return results
    
    def _map_to_taxonomy(self, detected_label: str, taxonomy: Dict[str, List[str]]) -> str:
        """Map detected label to our furniture taxonomy"""
        detected_label = detected_label.lower()
        
        # Expanded mapping for more furniture detection
        label_mappings = {
            # Seating
            'couch': 'sofa',
            'sofa': 'sofa',
            'sectional': 'sectional',
            'chair': 'armchair',
            'armchair': 'armchair',
            'dining chair': 'dining_chair',
            'stool': 'stool',
            'bench': 'bench',
            'ottoman': 'ottoman',
            'loveseat': 'loveseat',
            'recliner': 'recliner',
            'chaise lounge': 'chaise_lounge',
            'chaise': 'chaise_lounge',
            'bar stool': 'bar_stool',
            'office chair': 'office_chair',
            'accent chair': 'accent_chair',
            'pouffe': 'pouffe',
            
            # Tables
            'dining table': 'dining_table',
            'table': 'coffee_table',
            'coffee table': 'coffee_table',
            'side table': 'side_table',
            'end table': 'end_table',
            'console table': 'console_table',
            'console': 'console_table',
            'desk': 'desk',
            'writing desk': 'writing_desk',
            'computer desk': 'computer_desk',
            'nightstand': 'nightstand',
            'bedside table': 'nightstand',
            'bar table': 'bar_table',
            'bistro table': 'bistro_table',
            'nesting tables': 'nesting_tables',
            'dressing table': 'dressing_table',
            'vanity table': 'dressing_table',
            
            # Storage
            'cabinet': 'cabinet',
            'bookshelf': 'bookshelf',
            'bookcase': 'bookshelf',
            'dresser': 'dresser',
            'wardrobe': 'wardrobe',
            'armoire': 'armoire',
            'closet': 'wardrobe',
            'chest': 'chest_of_drawers',
            'chest of drawers': 'chest_of_drawers',
            'credenza': 'credenza',
            'sideboard': 'sideboard',
            'buffet': 'buffet',
            'china cabinet': 'china_cabinet',
            'display cabinet': 'display_cabinet',
            'tv stand': 'tv_stand',
            'media console': 'media_console',
            'shoe cabinet': 'shoe_cabinet',
            
            # Bedroom
            'bed': 'bed_frame',
            'bed frame': 'bed_frame',
            'headboard': 'headboard',
            'footboard': 'footboard',
            'platform bed': 'platform_bed',
            'bunk bed': 'bunk_bed',
            'daybed': 'daybed',
            'crib': 'crib',
            'bassinet': 'bassinet',
            
            # Kitchen
            'refrigerator': 'refrigerator',
            'fridge': 'refrigerator',
            'stove': 'stove',
            'oven': 'oven',
            'microwave': 'microwave',
            'dishwasher': 'dishwasher',
            'kitchen island': 'kitchen_island',
            'kitchen sink': 'kitchen_sink',
            'sink': 'kitchen_sink',
            'countertop': 'countertop',
            'counter': 'countertop',
            'backsplash': 'backsplash',
            
            # Lighting
            'lamp': 'table_lamp',
            'table lamp': 'table_lamp',
            'floor lamp': 'floor_lamp',
            'pendant light': 'pendant_light',
            'chandelier': 'chandelier',
            'ceiling light': 'ceiling_light',
            'wall light': 'wall_sconce',
            'sconce': 'wall_sconce',
            'desk lamp': 'desk_lamp',
            'ceiling fan': 'ceiling_fan',
            
            # Window treatments
            'curtain': 'curtains',
            'curtains': 'curtains',
            'drapes': 'drapes',
            'blinds': 'blinds',
            'shades': 'shades',
            'shutters': 'shutters',
            'valance': 'valance',
            
            # Soft furnishings
            'rug': 'rug',
            'carpet': 'carpet',
            'pillow': 'pillow',
            'cushion': 'cushion',
            'throw pillow': 'throw_pillow',
            'blanket': 'blanket',
            'throw': 'throw',
            'bedding': 'bedding',
            'duvet': 'duvet',
            'comforter': 'comforter',
            'sheets': 'sheets',
            
            # Wall decor
            'artwork': 'wall_art',
            'painting': 'painting',
            'picture': 'photograph',
            'poster': 'poster',
            'wall art': 'wall_art',
            'wall clock': 'wall_clock',
            'wall shelf': 'wall_shelf',
            'floating shelf': 'floating_shelf',
            
            # Decor accessories
            'mirror': 'mirror',
            'plant': 'potted_plant',
            'potted plant': 'potted_plant',
            'hanging plant': 'hanging_plant',
            'vase': 'vase',
            'candle': 'candle',
            'sculpture': 'sculpture',
            'decorative bowl': 'decorative_bowl',
            'picture frame': 'picture_frame',
            'clock': 'clock',
            'planter': 'planter',
            'flower pot': 'flower_pot',
            
            # Architectural
            'door': 'door',
            'window': 'window',
            'french doors': 'french_doors',
            'sliding door': 'sliding_door',
            'fireplace': 'fireplace',
            'mantle': 'mantle',
            'column': 'column',
            'pillar': 'pillar',
            'archway': 'archway',
            
            # Electronics
            'tv': 'tv',
            'television': 'television',
            'stereo': 'stereo',
            'speakers': 'speakers',
            'gaming console': 'gaming_console',
            'computer': 'computer',
            'monitor': 'monitor',
            'printer': 'printer',
            
            # Bathroom
            'bathtub': 'bathtub',
            'tub': 'bathtub',
            'sink': 'sink_vanity',
            'vanity': 'sink_vanity',
            'toilet': 'toilet',
            'shower': 'shower',
            
            # General items that might appear
            'window': 'window',
            'door': 'door',
            'fireplace': 'fireplace',
            'tv': 'tv',
            'television': 'tv'
        }
        
        # Check direct mappings first
        if detected_label in label_mappings:
            return label_mappings[detected_label]
        
        # Check if label contains any of our taxonomy items
        for category, items in taxonomy.items():
            for item in items:
                if item.replace('_', ' ') in detected_label or detected_label in item.replace('_', ' '):
                    return item
        
        # Fuzzy matching for partial matches
        for category, items in taxonomy.items():
            for item in items:
                item_words = item.replace('_', ' ').split()
                detected_words = detected_label.split()
                
                # If any word matches, consider it a match
                if any(word in detected_words for word in item_words):
                    return item
        
        # Return None if no match found
        return None
    
    def _generate_comprehensive_tags(self, mapped_category: str, raw_label: str, confidence: float, 
                                   box_area: int, detector: str, taxonomy: Dict[str, List[str]]) -> List[str]:
        """Generate comprehensive tags for detected objects"""
        tags = []
        
        # Basic tags
        tags.append(mapped_category)
        tags.append(detector)
        
        # Add raw label if different from mapped category
        if raw_label.lower() != mapped_category.lower():
            tags.append(raw_label.replace(' ', '_').lower())
        
        # Find taxonomy category group
        category_group = None
        for group, items in taxonomy.items():
            if mapped_category in items:
                category_group = group
                break
        
        if category_group:
            tags.append(f"category_{category_group}")
        
        # Confidence-based tags
        if confidence >= 0.9:
            tags.append("high_confidence")
        elif confidence >= 0.7:
            tags.append("medium_confidence")
        else:
            tags.append("low_confidence")
        
        # Size-based tags
        if box_area > 50000:
            tags.append("large_object")
        elif box_area > 20000:
            tags.append("medium_object")
        else:
            tags.append("small_object")
        
        # Specific furniture type tags
        furniture_type_tags = {
            # Seating variations
            'sofa': ['furniture', 'seating', 'upholstered', 'living_room'],
            'sectional': ['furniture', 'seating', 'upholstered', 'modular', 'large'],
            'armchair': ['furniture', 'seating', 'upholstered', 'single_seat'],
            'dining_chair': ['furniture', 'seating', 'dining_room', 'hard_surface'],
            'stool': ['furniture', 'seating', 'backless', 'portable'],
            'bench': ['furniture', 'seating', 'linear', 'multi_person'],
            'loveseat': ['furniture', 'seating', 'upholstered', 'two_seat'],
            'recliner': ['furniture', 'seating', 'upholstered', 'adjustable', 'comfort'],
            'chaise_lounge': ['furniture', 'seating', 'upholstered', 'reclining', 'luxury'],
            'bar_stool': ['furniture', 'seating', 'backless', 'tall', 'kitchen'],
            'office_chair': ['furniture', 'seating', 'office', 'adjustable', 'ergonomic'],
            'accent_chair': ['furniture', 'seating', 'decorative', 'statement'],
            'ottoman': ['furniture', 'seating', 'footrest', 'storage', 'soft'],
            'pouffe': ['furniture', 'seating', 'soft', 'portable', 'accent'],
            
            # Tables
            'coffee_table': ['furniture', 'table', 'living_room', 'low_height', 'center'],
            'side_table': ['furniture', 'table', 'accent', 'small', 'portable'],
            'dining_table': ['furniture', 'table', 'dining_room', 'large', 'gathering'],
            'console_table': ['furniture', 'table', 'accent', 'narrow', 'entryway'],
            'desk': ['furniture', 'table', 'workspace', 'office', 'productivity'],
            'nightstand': ['furniture', 'table', 'bedroom', 'bedside', 'storage'],
            'end_table': ['furniture', 'table', 'accent', 'small', 'beside_seating'],
            'accent_table': ['furniture', 'table', 'decorative', 'small'],
            'writing_desk': ['furniture', 'table', 'workspace', 'traditional', 'study'],
            'computer_desk': ['furniture', 'table', 'workspace', 'technology', 'ergonomic'],
            'bar_table': ['furniture', 'table', 'tall', 'kitchen', 'casual_dining'],
            'bistro_table': ['furniture', 'table', 'small', 'cafe_style', 'intimate'],
            'nesting_tables': ['furniture', 'table', 'modular', 'space_saving', 'flexible'],
            'dressing_table': ['furniture', 'table', 'bedroom', 'vanity', 'grooming'],
            
            # Storage
            'bookshelf': ['furniture', 'storage', 'books', 'shelving', 'display'],
            'cabinet': ['furniture', 'storage', 'enclosed', 'organized'],
            'dresser': ['furniture', 'storage', 'bedroom', 'drawers', 'clothing'],
            'wardrobe': ['furniture', 'storage', 'clothing', 'tall', 'bedroom'],
            'armoire': ['furniture', 'storage', 'clothing', 'tall', 'traditional'],
            'chest_of_drawers': ['furniture', 'storage', 'drawers', 'bedroom'],
            'credenza': ['furniture', 'storage', 'dining_room', 'low', 'serving'],
            'sideboard': ['furniture', 'storage', 'dining_room', 'serving', 'decorative'],
            'buffet': ['furniture', 'storage', 'dining_room', 'serving', 'food'],
            'china_cabinet': ['furniture', 'storage', 'dining_room', 'display', 'glassware'],
            'display_cabinet': ['furniture', 'storage', 'display', 'collectibles', 'glass'],
            'tv_stand': ['furniture', 'storage', 'entertainment', 'media', 'low'],
            'media_console': ['furniture', 'storage', 'entertainment', 'modern', 'technology'],
            'shoe_cabinet': ['furniture', 'storage', 'shoes', 'entryway', 'organization'],
            'pantry_cabinet': ['furniture', 'storage', 'kitchen', 'food', 'organization'],
            
            # Bedroom
            'bed_frame': ['furniture', 'bedroom', 'sleeping', 'structure'],
            'mattress': ['bedding', 'sleeping', 'comfort', 'soft'],
            'headboard': ['furniture', 'bedroom', 'decorative', 'back_support'],
            'footboard': ['furniture', 'bedroom', 'decorative', 'structure'],
            'bed_base': ['furniture', 'bedroom', 'sleeping', 'foundation'],
            'platform_bed': ['furniture', 'bedroom', 'modern', 'low_profile'],
            'bunk_bed': ['furniture', 'bedroom', 'space_saving', 'children', 'stacked'],
            'daybed': ['furniture', 'bedroom', 'multi_purpose', 'sofa', 'guest'],
            'murphy_bed': ['furniture', 'bedroom', 'space_saving', 'wall_mounted', 'fold_away'],
            'crib': ['furniture', 'bedroom', 'baby', 'safety', 'small'],
            'bassinet': ['furniture', 'bedroom', 'baby', 'portable', 'newborn'],
            'changing_table': ['furniture', 'bedroom', 'baby', 'functional', 'storage'],
            
            # Kitchen & Appliances
            'upper_cabinet': ['kitchen', 'storage', 'wall_mounted', 'dishes'],
            'lower_cabinet': ['kitchen', 'storage', 'base', 'pots_pans'],
            'kitchen_island': ['kitchen', 'workspace', 'storage', 'center', 'prep'],
            'breakfast_bar': ['kitchen', 'seating', 'casual_dining', 'counter_height'],
            'pantry': ['kitchen', 'storage', 'food', 'organization', 'tall'],
            'spice_rack': ['kitchen', 'storage', 'spices', 'organization', 'small'],
            'wine_rack': ['kitchen', 'storage', 'wine', 'display', 'specialized'],
            'refrigerator': ['appliance', 'kitchen', 'cooling', 'food_storage', 'large'],
            'stove': ['appliance', 'kitchen', 'cooking', 'heat', 'burners'],
            'oven': ['appliance', 'kitchen', 'baking', 'cooking', 'enclosed'],
            'microwave': ['appliance', 'kitchen', 'cooking', 'quick', 'compact'],
            'dishwasher': ['appliance', 'kitchen', 'cleaning', 'dishes', 'automated'],
            'range_hood': ['appliance', 'kitchen', 'ventilation', 'cooking_fumes'],
            'garbage_disposal': ['appliance', 'kitchen', 'waste', 'under_sink'],
            'coffee_maker': ['appliance', 'kitchen', 'beverage', 'morning', 'automated'],
            'toaster': ['appliance', 'kitchen', 'breakfast', 'bread', 'small'],
            'blender': ['appliance', 'kitchen', 'mixing', 'smoothies', 'portable'],
            'kitchen_sink': ['fixture', 'kitchen', 'cleaning', 'water', 'essential'],
            'faucet': ['fixture', 'kitchen', 'water_control', 'spout'],
            'backsplash': ['kitchen', 'wall_covering', 'protective', 'decorative'],
            'countertop': ['kitchen', 'work_surface', 'prep', 'durable'],
            
            # Lighting & Electrical
            'pendant_light': ['lighting', 'hanging', 'ceiling', 'focused'],
            'floor_lamp': ['lighting', 'standing', 'portable', 'ambient'],
            'table_lamp': ['lighting', 'accent', 'desktop', 'task'],
            'wall_sconce': ['lighting', 'mounted', 'wall', 'decorative'],
            'chandelier': ['lighting', 'hanging', 'ceiling', 'luxury', 'statement'],
            'ceiling_light': ['lighting', 'ceiling', 'general', 'bright'],
            'track_lighting': ['lighting', 'ceiling', 'adjustable', 'modern'],
            'recessed_light': ['lighting', 'ceiling', 'hidden', 'clean'],
            'under_cabinet_light': ['lighting', 'kitchen', 'task', 'hidden'],
            'desk_lamp': ['lighting', 'task', 'adjustable', 'work'],
            'reading_light': ['lighting', 'task', 'focused', 'comfort'],
            'accent_lighting': ['lighting', 'decorative', 'mood', 'ambiance'],
            'string_lights': ['lighting', 'decorative', 'casual', 'festive'],
            'ceiling_fan': ['cooling', 'ceiling', 'air_circulation', 'comfort'],
            'smoke_detector': ['safety', 'ceiling', 'fire_protection', 'alarm'],
            'air_vent': ['ventilation', 'air_circulation', 'climate_control'],
            'skylight': ['architectural', 'ceiling', 'natural_light', 'opening'],
            'beam': ['architectural', 'structural', 'ceiling', 'decorative'],
            'molding': ['architectural', 'decorative', 'trim', 'detail'],
            'medallion': ['architectural', 'ceiling', 'decorative', 'ornate'],
            
            # Bathroom
            'toilet': ['bathroom', 'fixture', 'sanitary', 'essential'],
            'shower': ['bathroom', 'fixture', 'bathing', 'standing'],
            'bathtub': ['bathroom', 'fixture', 'bathing', 'soaking'],
            'sink_vanity': ['bathroom', 'fixture', 'storage', 'grooming'],
            'bathroom_sink': ['bathroom', 'fixture', 'washing', 'essential'],
            'shower_door': ['bathroom', 'glass', 'enclosure', 'wet_area'],
            'shower_curtain': ['bathroom', 'textile', 'privacy', 'water_barrier'],
            'medicine_cabinet': ['bathroom', 'storage', 'mirror', 'wall_mounted'],
            'towel_rack': ['bathroom', 'storage', 'towels', 'wall_mounted'],
            'toilet_paper_holder': ['bathroom', 'storage', 'essential', 'wall_mounted'],
            'linen_closet': ['bathroom', 'storage', 'towels', 'organized'],
            'bathroom_cabinet': ['bathroom', 'storage', 'toiletries', 'organization'],
            'vanity_cabinet': ['bathroom', 'storage', 'under_sink', 'plumbing'],
            'over_toilet_storage': ['bathroom', 'storage', 'space_saving', 'above_toilet'],
            
            # Window Treatments
            'curtains': ['textile', 'window_treatment', 'hanging', 'privacy'],
            'drapes': ['textile', 'window_treatment', 'heavy', 'formal'],
            'blinds': ['window_treatment', 'adjustable', 'light_control', 'horizontal'],
            'shades': ['window_treatment', 'fabric', 'light_control', 'roll_up'],
            'shutters': ['window_treatment', 'wood', 'adjustable', 'architectural'],
            'valance': ['textile', 'window_treatment', 'decorative', 'top_only'],
            'cornice': ['window_treatment', 'decorative', 'top_trim', 'formal'],
            'window_film': ['window_treatment', 'privacy', 'adhesive', 'modern'],
            
            # Soft Furnishings
            'rug': ['textile', 'floor_covering', 'decorative', 'area'],
            'carpet': ['textile', 'floor_covering', 'wall_to_wall', 'soft'],
            'pillow': ['textile', 'comfort', 'decorative', 'support'],
            'cushion': ['textile', 'comfort', 'seating', 'soft'],
            'throw_pillow': ['textile', 'decorative', 'accent', 'removable'],
            'blanket': ['textile', 'comfort', 'warmth', 'covering'],
            'throw': ['textile', 'decorative', 'casual', 'lightweight'],
            'bedding': ['textile', 'bedroom', 'sleeping', 'comfort'],
            'duvet': ['textile', 'bedroom', 'warmth', 'comforter'],
            'comforter': ['textile', 'bedroom', 'warmth', 'fluffy'],
            'sheets': ['textile', 'bedroom', 'sleeping', 'smooth'],
            'pillowcase': ['textile', 'bedroom', 'pillow_cover', 'washable'],
            'sofa_cushions': ['textile', 'seating', 'comfort', 'removable'],
            'chair_cushions': ['textile', 'seating', 'comfort', 'tied'],
            'seat_cushions': ['textile', 'seating', 'comfort', 'bottom'],
            'back_cushions': ['textile', 'seating', 'support', 'back'],
            
            # Wall Decor
            'wall_art': ['decor', 'wall_mounted', 'visual', 'artistic'],
            'painting': ['decor', 'wall_mounted', 'artistic', 'painted'],
            'photograph': ['decor', 'wall_mounted', 'visual', 'captured'],
            'poster': ['decor', 'wall_mounted', 'printed', 'casual'],
            'wall_sculpture': ['decor', 'wall_mounted', 'three_dimensional', 'artistic'],
            'wall_clock': ['decor', 'wall_mounted', 'functional', 'time'],
            'decorative_plate': ['decor', 'wall_mounted', 'ceramic', 'traditional'],
            'wall_shelf': ['storage', 'wall_mounted', 'display', 'functional'],
            'floating_shelf': ['storage', 'wall_mounted', 'modern', 'clean'],
            
            # Decor Accessories
            'mirror': ['decor', 'reflective', 'functional', 'light_enhancing'],
            'vase': ['decor', 'accent', 'flowers', 'ceramic'],
            'candle': ['decor', 'ambiance', 'fragrance', 'lighting'],
            'sculpture': ['decor', 'artistic', 'three_dimensional', 'statement'],
            'decorative_bowl': ['decor', 'accent', 'functional', 'display'],
            'picture_frame': ['decor', 'photo_display', 'memories', 'border'],
            'clock': ['decor', 'functional', 'time', 'mechanical'],
            'lamp_shade': ['lighting', 'cover', 'diffusion', 'decorative'],
            'decorative_object': ['decor', 'accent', 'ornamental', 'personal'],
            
            # Plants & Planters
            'potted_plant': ['decor', 'natural', 'living', 'air_purifying'],
            'hanging_plant': ['decor', 'natural', 'suspended', 'trailing'],
            'planter': ['decor', 'plant_container', 'decorative', 'drainage'],
            'flower_pot': ['decor', 'plant_container', 'ceramic', 'small'],
            'garden_planter': ['decor', 'plant_container', 'outdoor', 'large'],
            'herb_garden': ['kitchen', 'plants', 'edible', 'fresh'],
            
            # Architectural Elements
            'door': ['architectural', 'opening', 'entrance', 'security'],
            'window': ['architectural', 'opening', 'natural_light', 'view'],
            'french_doors': ['architectural', 'opening', 'glass', 'elegant'],
            'sliding_door': ['architectural', 'opening', 'space_saving', 'modern'],
            'bifold_door': ['architectural', 'opening', 'folding', 'closet'],
            'pocket_door': ['architectural', 'opening', 'hidden', 'space_saving'],
            'window_frame': ['architectural', 'structure', 'border', 'support'],
            'door_frame': ['architectural', 'structure', 'border', 'support'],
            'fireplace': ['architectural', 'heating', 'focal_point', 'gathering'],
            'mantle': ['architectural', 'decorative', 'fireplace', 'display'],
            'column': ['architectural', 'structural', 'vertical', 'support'],
            'pillar': ['architectural', 'structural', 'decorative', 'vertical'],
            'archway': ['architectural', 'opening', 'curved', 'decorative'],
            'niche': ['architectural', 'recessed', 'display', 'wall'],
            'built_in_shelf': ['architectural', 'storage', 'custom', 'integrated'],
            'wainscoting': ['architectural', 'wall_covering', 'traditional', 'lower_wall'],
            'chair_rail': ['architectural', 'trim', 'wall_protection', 'horizontal'],
            'hardwood_floor': ['flooring', 'wood', 'durable', 'classic'],
            'tile_floor': ['flooring', 'ceramic', 'water_resistant', 'easy_clean'],
            'carpet_floor': ['flooring', 'textile', 'soft', 'warm'],
            'laminate_floor': ['flooring', 'synthetic', 'affordable', 'wood_look'],
            'vinyl_floor': ['flooring', 'synthetic', 'water_resistant', 'flexible'],
            'stone_floor': ['flooring', 'natural', 'durable', 'luxury'],
            'accent_wall': ['wall_feature', 'decorative', 'focal_point', 'color'],
            'brick_wall': ['wall_feature', 'material', 'texture', 'rustic'],
            'stone_wall': ['wall_feature', 'material', 'texture', 'natural'],
            'wood_paneling': ['wall_feature', 'material', 'texture', 'warm'],
            'wallpaper': ['wall_feature', 'covering', 'pattern', 'decorative'],
            
            # Electronics & Technology
            'tv': ['electronics', 'entertainment', 'display', 'viewing'],
            'television': ['electronics', 'entertainment', 'display', 'broadcast'],
            'stereo': ['electronics', 'audio', 'music', 'sound'],
            'speakers': ['electronics', 'audio', 'sound_output', 'music'],
            'gaming_console': ['electronics', 'entertainment', 'gaming', 'interactive'],
            'dvd_player': ['electronics', 'entertainment', 'movies', 'media'],
            'sound_bar': ['electronics', 'audio', 'tv_enhancement', 'sleek'],
            'computer': ['electronics', 'productivity', 'work', 'technology'],
            'monitor': ['electronics', 'display', 'computer', 'visual'],
            'printer': ['electronics', 'office', 'documents', 'paper'],
            'desk_accessories': ['office', 'organization', 'productivity', 'small'],
            'filing_cabinet': ['office', 'storage', 'documents', 'organization'],
            'desk_organizer': ['office', 'organization', 'supplies', 'tidy'],
            'smart_speaker': ['electronics', 'audio', 'voice_control', 'smart_home'],
            'security_camera': ['electronics', 'security', 'monitoring', 'safety'],
            'thermostat': ['electronics', 'climate_control', 'temperature', 'energy'],
            'smart_switch': ['electronics', 'lighting_control', 'automation', 'smart_home'],
            'home_hub': ['electronics', 'smart_home', 'control_center', 'automation'],
            
            # Outdoor & Patio
            'patio_chair': ['outdoor', 'seating', 'weather_resistant', 'relaxation'],
            'outdoor_table': ['outdoor', 'table', 'weather_resistant', 'dining'],
            'patio_umbrella': ['outdoor', 'shade', 'sun_protection', 'adjustable'],
            'outdoor_sofa': ['outdoor', 'seating', 'weather_resistant', 'comfort'],
            'deck_chair': ['outdoor', 'seating', 'reclining', 'sun'],
            'garden_bench': ['outdoor', 'seating', 'garden', 'relaxation'],
            'outdoor_dining_set': ['outdoor', 'dining', 'set', 'entertaining'],
            'outdoor_plant': ['outdoor', 'plant', 'weather_resistant', 'landscaping'],
            'garden_sculpture': ['outdoor', 'decor', 'artistic', 'garden'],
            'outdoor_lighting': ['outdoor', 'lighting', 'weather_resistant', 'ambiance'],
            'wind_chime': ['outdoor', 'decor', 'sound', 'hanging'],
            'bird_feeder': ['outdoor', 'wildlife', 'feeding', 'nature'],
            
            # Specialty Items
            'treadmill': ['exercise', 'cardio', 'running', 'fitness'],
            'exercise_bike': ['exercise', 'cardio', 'cycling', 'fitness'],
            'weights': ['exercise', 'strength', 'muscle', 'fitness'],
            'yoga_mat': ['exercise', 'yoga', 'stretching', 'floor'],
            'exercise_ball': ['exercise', 'core', 'balance', 'flexibility'],
            'toy_chest': ['children', 'storage', 'toys', 'organization'],
            'kids_table': ['children', 'table', 'small_scale', 'play'],
            'kids_chair': ['children', 'seating', 'small_scale', 'colorful'],
            'high_chair': ['children', 'seating', 'feeding', 'safety'],
            'play_table': ['children', 'table', 'activity', 'learning'],
            'toy_storage': ['children', 'storage', 'toys', 'bins'],
            'conference_table': ['office', 'table', 'meeting', 'large'],
            'office_desk': ['office', 'table', 'work', 'productivity'],
            'executive_chair': ['office', 'seating', 'luxury', 'management'],
            'meeting_chair': ['office', 'seating', 'conference', 'professional'],
            'whiteboard': ['office', 'presentation', 'writing', 'communication'],
            'bulletin_board': ['office', 'display', 'notices', 'organization'],
            'screen': ['room_divider', 'privacy', 'portable', 'folding'],
            'room_divider': ['partition', 'space_separation', 'flexible', 'privacy'],
            'partition': ['room_divider', 'office', 'space_division', 'modular'],
            'bookcase_divider': ['storage', 'room_divider', 'books', 'dual_purpose'],
            'christmas_tree': ['seasonal', 'holiday', 'decoration', 'festive'],
            'holiday_decoration': ['seasonal', 'festive', 'celebration', 'temporary'],
            'seasonal_pillow': ['seasonal', 'textile', 'decorative', 'changeable'],
            'seasonal_wreath': ['seasonal', 'door_decoration', 'natural', 'festive'],
            'door_handle': ['hardware', 'functional', 'access', 'grip'],
            'cabinet_hardware': ['hardware', 'functional', 'access', 'small'],
            'light_switch': ['electrical', 'control', 'lighting', 'wall_mounted'],
            'outlet': ['electrical', 'power', 'connection', 'wall_mounted'],
            'vent_cover': ['ventilation', 'air_flow', 'cover', 'functional']
        }
        
        # Add specific furniture type tags
        if mapped_category in furniture_type_tags:
            tags.extend(furniture_type_tags[mapped_category])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_tags = []
        for tag in tags:
            if tag not in seen:
                seen.add(tag)
                unique_tags.append(tag)
        
        return unique_tags
    
    async def detect_with_prompts(self, image_path: str, prompts: List[str]) -> List[Dict[str, Any]]:
        """
        Detect objects using specific text prompts
        This would be the actual GroundingDINO functionality
        """
        # Placeholder implementation - would use actual GroundingDINO here
        # For now, fall back to standard detection
        taxonomy = {"furniture": prompts}
        return await self.detect_objects(image_path, taxonomy)
    
    def set_confidence_threshold(self, threshold: float):
        """Update confidence threshold for both detectors"""
        self.yolo_confidence_threshold = threshold
        self.detr_confidence_threshold = max(threshold - 0.1, 0.05)  # DETR slightly lower
        logger.info(f"Updated YOLO threshold to {threshold}, DETR to {self.detr_confidence_threshold}")
    
    def set_yolo_threshold(self, threshold: float):
        """Set YOLOv8 confidence threshold specifically"""
        self.yolo_confidence_threshold = threshold
        logger.info(f"Updated YOLOv8 confidence threshold to {threshold}")
    
    def set_detr_threshold(self, threshold: float):
        """Set DETR confidence threshold specifically"""
        self.detr_confidence_threshold = threshold
        logger.info(f"Updated DETR confidence threshold to {threshold}")
    
    def get_detector_status(self) -> Dict[str, Any]:
        """Get status of both detectors"""
        return {
            "yolo": {
                "available": self.yolo_available,
                "model": "yolov8s.pt" if self.yolo_available else None,
                "confidence_threshold": self.yolo_confidence_threshold
            },
            "detr": {
                "available": self.detr_available,
                "model": "facebook/detr-resnet-50" if self.detr_available else None,
                "confidence_threshold": self.detr_confidence_threshold
            },
            "device": self.device,
            "min_box_area": self.min_box_area
        }
    
    async def detect_with_multiple_scales(self, image_path: str, taxonomy: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """
        Enhanced detection using multiple image scales to catch more objects
        """
        try:
            image = Image.open(image_path).convert("RGB")
            original_size = image.size
            all_results = []
            
            # Try different scales to catch objects of various sizes
            scales = [1.0, 0.8, 1.2]  # Original, smaller, larger
            
            for scale in scales:
                if scale != 1.0:
                    new_size = (int(original_size[0] * scale), int(original_size[1] * scale))
                    scaled_image = image.resize(new_size)
                else:
                    scaled_image = image
                
                # Run detection on scaled image
                detections = self.detector(scaled_image)
                
                # Process detections and scale back coordinates
                for detection in detections:
                    if detection['score'] >= self.confidence_threshold * 0.8:  # Slightly lower threshold for scaled images
                        mapped_category = self._map_to_taxonomy(detection['label'], taxonomy)
                        
                        if mapped_category:
                            box = detection['box']
                            # Scale coordinates back to original image size
                            if scale != 1.0:
                                box = {
                                    'xmin': box['xmin'] / scale,
                                    'ymin': box['ymin'] / scale,
                                    'xmax': box['xmax'] / scale,
                                    'ymax': box['ymax'] / scale
                                }
                            
                            width = box['xmax'] - box['xmin']
                            height = box['ymax'] - box['ymin']
                            bbox = [box['xmin'], box['ymin'], width, height]
                            box_area = width * height
                            
                            if box_area >= self.min_box_area:
                                all_results.append({
                                    'bbox': bbox,
                                    'category': mapped_category,
                                    'confidence': float(detection['score']),
                                    'raw_label': detection['label'],
                                    'tags': [mapped_category, f'scale_{scale}'],
                                    'box_area': int(box_area),
                                    'detection_scale': scale
                                })
            
            # Remove duplicates (objects detected at multiple scales)
            unique_results = self._remove_duplicate_detections(all_results)
            
            logger.info(f"Multi-scale detection found {len(unique_results)} unique objects")
            return unique_results
            
        except Exception as e:
            logger.error(f"Multi-scale detection failed for {image_path}", error=str(e))
            # Fallback to regular detection
            return await self.detect_objects(image_path, taxonomy)
    
    def _remove_duplicate_detections(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate detections based on bounding box overlap"""
        if not detections:
            return []
        
        # Sort by confidence
        sorted_detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        unique_detections = []
        
        for detection in sorted_detections:
            is_duplicate = False
            
            for existing in unique_detections:
                # Check if bounding boxes overlap significantly
                overlap_ratio = self._calculate_bbox_overlap(detection['bbox'], existing['bbox'])
                
                # If high overlap and same category, consider it a duplicate
                if overlap_ratio > 0.7 and detection['category'] == existing['category']:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_detections.append(detection)
        
        return unique_detections
    
    def _calculate_bbox_overlap(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate overlap ratio between two bounding boxes"""
        x1_min, y1_min, w1, h1 = bbox1
        x2_min, y2_min, w2, h2 = bbox2
        
        x1_max, y1_max = x1_min + w1, y1_min + h1
        x2_max, y2_max = x2_min + w2, y2_min + h2
        
        # Calculate intersection
        intersect_xmin = max(x1_min, x2_min)
        intersect_ymin = max(y1_min, y2_min)
        intersect_xmax = min(x1_max, x2_max)
        intersect_ymax = min(y1_max, y2_max)
        
        if intersect_xmin >= intersect_xmax or intersect_ymin >= intersect_ymax:
            return 0.0
        
        intersect_area = (intersect_xmax - intersect_xmin) * (intersect_ymax - intersect_ymin)
        union_area = (w1 * h1) + (w2 * h2) - intersect_area
        
        return intersect_area / union_area if union_area > 0 else 0.0