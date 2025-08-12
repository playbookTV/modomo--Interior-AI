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
        """Initialize DETR detector as fallback"""
        try:
            self.detr_detector = pipeline(
                "object-detection",
                model="facebook/detr-resnet-50",
                device=0 if self.device == "cuda" else -1
            )
            self.detr_available = True
            logger.info("✅ DETR detector initialized successfully (fallback)")
            
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
                        results.append({
                            'bbox': bbox,
                            'category': mapped_category,
                            'confidence': conf,
                            'raw_label': class_name,
                            'tags': [mapped_category, 'yolo'],
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
                    
                    # Add tags for better categorization
                    tags = [mapped_category, 'detr']
                    if detection['label'] != mapped_category:
                        tags.append(detection['label'].replace(' ', '_'))
                    
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
            'ottoman': 'stool',
            
            # Tables
            'dining table': 'dining_table',
            'table': 'coffee_table',
            'coffee table': 'coffee_table',
            'side table': 'side_table',
            'end table': 'side_table',
            'console table': 'console_table',
            'console': 'console_table',
            'desk': 'desk',
            'nightstand': 'nightstand',
            'bedside table': 'nightstand',
            
            # Storage
            'cabinet': 'cabinet',
            'bookshelf': 'bookshelf',
            'bookcase': 'bookshelf',
            'dresser': 'dresser',
            'wardrobe': 'wardrobe',
            'closet': 'wardrobe',
            'chest': 'dresser',
            
            # Lighting
            'lamp': 'table_lamp',
            'table lamp': 'table_lamp',
            'floor lamp': 'floor_lamp',
            'pendant light': 'pendant_light',
            'chandelier': 'pendant_light',
            'wall light': 'wall_sconce',
            'sconce': 'wall_sconce',
            
            # Bedroom
            'bed': 'bed_frame',
            'bed frame': 'bed_frame',
            'headboard': 'headboard',
            'mattress': 'mattress',
            
            # Soft furnishings
            'rug': 'rug',
            'carpet': 'rug',
            'curtain': 'curtains',
            'curtains': 'curtains',
            'drapes': 'curtains',
            'pillow': 'pillow',
            'cushion': 'pillow',
            'blanket': 'blanket',
            'throw': 'blanket',
            
            # Decor
            'mirror': 'mirror',
            'plant': 'plant',
            'potted plant': 'plant',
            'vase': 'decorative_object',
            'artwork': 'wall_art',
            'painting': 'wall_art',
            'picture': 'wall_art',
            'frame': 'wall_art',
            
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