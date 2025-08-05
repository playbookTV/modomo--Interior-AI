import cv2
import gc
import numpy as np
from typing import List, Dict, Tuple
from ultralytics import YOLO
import torch
from PIL import Image
import structlog
import contextlib

logger = structlog.get_logger()

class DetectedObject:
    def __init__(self, class_name: str, confidence: float, bbox: List[int], description: str = ""):
        self.class_name = class_name
        self.confidence = confidence
        self.bbox = bbox  # [x, y, width, height]
        self.description = description or f"{class_name.replace('_', ' ').title()}"

class ObjectDetector:
    """
    Real object detection using YOLOv8 for furniture and room objects
    """
    
    # Furniture and interior objects we care about for room makeovers
    FURNITURE_CLASSES = {
        'chair', 'couch', 'sofa', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
        'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
        'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush', 'potted plant', 'bowl', 'banana', 'apple',
        'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake'
    }
    
    def __init__(self, model_size: str = 'yolov8n.pt'):
        """
        Initialize YOLO model
        Args:
            model_size: 'yolov8n.pt' (nano), 'yolov8s.pt' (small), 'yolov8m.pt' (medium), 'yolov8l.pt' (large)
        """
        self.model_size = model_size
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._model_loaded = False
        logger.info(f"Object detector will use device: {self.device}")
        
    def _clear_gpu_memory(self):
        """Clear GPU memory and run garbage collection"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        
    @contextlib.contextmanager
    def _gpu_memory_context(self):
        """Context manager for GPU memory management during inference"""
        try:
            # Clear memory before inference
            self._clear_gpu_memory()
            yield
        finally:
            # Clear memory after inference
            self._clear_gpu_memory()
            
    def unload_model(self):
        """Unload model from memory to free up GPU/CPU resources"""
        try:
            if self.model is not None:
                # Move model to CPU first to free GPU memory
                if hasattr(self.model, 'to'):
                    self.model.to('cpu')
                del self.model
                self.model = None
                
            self._model_loaded = False
            self._clear_gpu_memory()
            logger.info("YOLO model unloaded successfully")
            
        except Exception as e:
            logger.error(f"Error unloading YOLO model: {e}")
            
    def __del__(self):
        """Cleanup on object destruction"""
        self.unload_model()
        
    async def load_model(self):
        """Load YOLO model"""
        if self._model_loaded:
            logger.info("YOLO model already loaded")
            return
            
        try:
            logger.info(f"Loading YOLO model: {self.model_size}")
            
            # Clear any existing memory first
            self._clear_gpu_memory()
            
            self.model = YOLO(self.model_size)
            self.model.to(self.device)
            
            self._model_loaded = True
            logger.info("YOLO model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise
            
    def detect_furniture_objects(self, image: Image.Image, confidence_threshold: float = 0.5) -> List[DetectedObject]:
        """
        Detect furniture and room objects in an image
        
        Args:
            image: PIL Image
            confidence_threshold: Minimum confidence for detection
            
        Returns:
            List of DetectedObject instances
        """
        if self.model is None or not self._model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
            
        with self._gpu_memory_context():
            try:
                # Convert PIL to numpy array
                img_array = np.array(image)
                
                # Run YOLO detection with memory management
                with torch.inference_mode():
                    results = self.model(img_array, conf=confidence_threshold, verbose=False)
            
            detected_objects = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get class ID and name
                        class_id = int(box.cls[0])
                        class_name = self.model.names[class_id]
                        confidence = float(box.conf[0])
                        
                        # Only include furniture/interior objects
                        if class_name.lower() in self.FURNITURE_CLASSES:
                            # Get bounding box coordinates
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            bbox = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]  # [x, y, width, height]
                            
                            # Create description
                            description = self._generate_description(class_name, bbox, image.size)
                            
                            detected_obj = DetectedObject(
                                class_name=class_name,
                                confidence=confidence,
                                bbox=bbox,
                                description=description
                            )
                            
                            detected_objects.append(detected_obj)
                            
            logger.info(f"Detected {len(detected_objects)} furniture objects")
            return detected_objects
            
        except Exception as e:
            logger.error(f"Object detection failed: {e}")
            raise
            
    def _generate_description(self, class_name: str, bbox: List[int], image_size: Tuple[int, int]) -> str:
        """Generate human-readable description of detected object"""
        width, height = image_size
        x, y, w, h = bbox
        
        # Calculate relative position
        center_x = x + w // 2
        center_y = y + h // 2
        
        # Determine position
        h_pos = "left" if center_x < width * 0.33 else "right" if center_x > width * 0.67 else "center"
        v_pos = "top" if center_y < height * 0.33 else "bottom" if center_y > height * 0.67 else "middle"
        
        # Determine size
        area_ratio = (w * h) / (width * height)
        size = "large" if area_ratio > 0.2 else "small" if area_ratio < 0.05 else "medium"
        
        # Create description
        clean_name = class_name.replace('_', ' ').title()
        position = f"{v_pos} {h_pos}".strip()
        
        return f"{size.title()} {clean_name.lower()} in the {position} of the room"
        
    def detect_room_layout(self, image: Image.Image) -> Dict:
        """
        Analyze room layout and spatial relationships
        
        Returns:
            Dictionary with room layout analysis
        """
        objects = self.detect_furniture_objects(image)
        
        # Group objects by type
        object_counts = {}
        for obj in objects:
            obj_type = obj.class_name
            object_counts[obj_type] = object_counts.get(obj_type, 0) + 1
            
        # Determine room type based on detected objects
        room_type = self._infer_room_type(object_counts)
        
        # Calculate furniture density
        total_furniture_area = sum((obj.bbox[2] * obj.bbox[3]) for obj in objects)
        image_area = image.size[0] * image.size[1]
        furniture_density = total_furniture_area / image_area if image_area > 0 else 0
        
        return {
            'room_type': room_type,
            'detected_objects': [
                {
                    'type': obj.class_name,
                    'confidence': obj.confidence,
                    'bounding_box': obj.bbox,
                    'description': obj.description
                }
                for obj in objects
            ],
            'object_counts': object_counts,
            'furniture_density': furniture_density,
            'total_objects': len(objects),
            'layout_analysis': self._analyze_spatial_layout(objects, image.size)
        }
        
    def _infer_room_type(self, object_counts: Dict[str, int]) -> str:
        """Infer room type based on detected objects"""
        if 'bed' in object_counts:
            return 'bedroom'
        elif 'couch' in object_counts or 'sofa' in object_counts:
            return 'living_room'  
        elif 'dining table' in object_counts:
            return 'dining_room'
        elif 'toilet' in object_counts or 'sink' in object_counts:
            return 'bathroom'
        elif 'refrigerator' in object_counts or 'microwave' in object_counts:
            return 'kitchen'
        else:
            return 'unknown'
            
    def _analyze_spatial_layout(self, objects: List[DetectedObject], image_size: Tuple[int, int]) -> Dict:
        """Analyze spatial relationships between objects"""
        width, height = image_size
        
        # Divide room into zones
        zones = {
            'left': [obj for obj in objects if (obj.bbox[0] + obj.bbox[2]//2) < width * 0.33],
            'center': [obj for obj in objects if width * 0.33 <= (obj.bbox[0] + obj.bbox[2]//2) <= width * 0.67],
            'right': [obj for obj in objects if (obj.bbox[0] + obj.bbox[2]//2) > width * 0.67]
        }
        
        # Find empty zones for product suggestions
        empty_zones = []
        for zone_name, zone_objects in zones.items():
            if len(zone_objects) == 0:
                empty_zones.append(zone_name)
                
        return {
            'zones': {zone: len(objs) for zone, objs in zones.items()},
            'empty_zones': empty_zones,
            'center_piece': self._find_center_piece(objects, image_size),
            'wall_space_available': self._estimate_wall_space(objects, image_size)
        }
        
    def _find_center_piece(self, objects: List[DetectedObject], image_size: Tuple[int, int]) -> Dict:
        """Find the main/center piece of furniture"""
        if not objects:
            return {}
            
        # Find largest object by area
        largest_obj = max(objects, key=lambda obj: obj.bbox[2] * obj.bbox[3])
        
        return {
            'type': largest_obj.class_name,
            'confidence': largest_obj.confidence,
            'description': largest_obj.description,
            'is_focal_point': True
        }
        
    def _estimate_wall_space(self, objects: List[DetectedObject], image_size: Tuple[int, int]) -> float:
        """Estimate available wall space for decorations"""
        width, height = image_size
        
        # Simple estimation: assume top 30% is wall space
        wall_area = width * (height * 0.3)
        
        # Subtract area occupied by wall-mounted objects
        occupied_wall_area = 0
        for obj in objects:
            obj_center_y = obj.bbox[1] + obj.bbox[3] // 2
            if obj_center_y < height * 0.4:  # Objects in upper area
                occupied_wall_area += obj.bbox[2] * obj.bbox[3]
                
        available_wall_ratio = max(0, (wall_area - occupied_wall_area) / wall_area)
        return available_wall_ratio 