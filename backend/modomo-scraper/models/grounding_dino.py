"""
GroundingDINO integration for object detection in interior design scenes
"""

import torch
from PIL import Image
from typing import List, Dict, Any
import numpy as np
from transformers import pipeline
import structlog

logger = structlog.get_logger()

class GroundingDINODetector:
    """Object detector using GroundingDINO with furniture-specific prompts"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing GroundingDINO detector on {self.device}")
        
        # Use HuggingFace transformers pipeline for object detection
        # In production, replace with actual GroundingDINO model
        self.detector = pipeline(
            "object-detection",
            model="facebook/detr-resnet-50",
            device=0 if self.device == "cuda" else -1
        )
        
        # Confidence thresholds
        self.confidence_threshold = 0.3
        
    async def detect_objects(self, image_path: str, taxonomy: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """
        Detect objects in an image using the furniture taxonomy
        
        Args:
            image_path: Path to the image file
            taxonomy: Dictionary mapping categories to object types
            
        Returns:
            List of detected objects with bounding boxes and categories
        """
        try:
            # Load image
            image = Image.open(image_path).convert("RGB")
            
            # Get all furniture classes from taxonomy
            furniture_classes = []
            for category, items in taxonomy.items():
                furniture_classes.extend(items)
            
            # Run detection
            detections = self.detector(image)
            
            # Filter and format results
            results = []
            for detection in detections:
                if detection['score'] >= self.confidence_threshold:
                    # Map detected label to our taxonomy
                    mapped_category = self._map_to_taxonomy(detection['label'], taxonomy)
                    
                    if mapped_category:
                        # Convert box format [x_min, y_min, x_max, y_max] to [x, y, width, height]
                        box = detection['box']
                        bbox = [
                            box['xmin'],
                            box['ymin'],
                            box['xmax'] - box['xmin'],
                            box['ymax'] - box['ymin']
                        ]
                        
                        results.append({
                            'bbox': bbox,
                            'category': mapped_category,
                            'confidence': float(detection['score']),
                            'raw_label': detection['label']
                        })
            
            logger.info(f"Detected {len(results)} objects in {image_path}")
            return results
            
        except Exception as e:
            logger.error(f"Detection failed for {image_path}", error=str(e))
            return []
    
    def _map_to_taxonomy(self, detected_label: str, taxonomy: Dict[str, List[str]]) -> str:
        """Map detected label to our furniture taxonomy"""
        detected_label = detected_label.lower()
        
        # Direct mapping for common furniture items
        label_mappings = {
            'couch': 'sofa',
            'sofa': 'sofa',
            'chair': 'armchair',
            'dining table': 'dining_table',
            'table': 'coffee_table',
            'bed': 'bed_frame',
            'cabinet': 'cabinet',
            'bookshelf': 'bookshelf',
            'lamp': 'table_lamp',
            'mirror': 'mirror',
            'plant': 'plant',
            'rug': 'rug',
            'curtain': 'curtains',
            'pillow': 'pillow',
            'blanket': 'blanket'
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
        """Update confidence threshold for detections"""
        self.confidence_threshold = threshold
        logger.info(f"Updated confidence threshold to {threshold}")