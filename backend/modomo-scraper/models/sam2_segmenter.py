"""
SAM2 integration for object segmentation
"""

import torch
import numpy as np
from PIL import Image, ImageDraw
import cv2
from typing import List, Tuple, Optional
import tempfile
import os
import structlog

logger = structlog.get_logger()

class SAM2Segmenter:
    """Object segmentation using SAM2 (Segment Anything Model 2)"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing SAM2 segmenter on {self.device}")
        
        # In production, load actual SAM2 model
        # For now, use a placeholder implementation
        self.model = None  # Would load SAM2 checkpoint here
        
    async def segment(self, image_path: str, bbox: List[float]) -> str:
        """
        Generate segmentation mask for object in bounding box
        
        Args:
            image_path: Path to the image
            bbox: Bounding box [x, y, width, height]
            
        Returns:
            Path to the generated mask image
        """
        try:
            # Load image
            image = Image.open(image_path).convert("RGB")
            image_array = np.array(image)
            
            # Convert bbox to integer coordinates
            x, y, w, h = [int(coord) for coord in bbox]
            
            # Ensure bbox is within image bounds
            img_h, img_w = image_array.shape[:2]
            x = max(0, min(x, img_w - 1))
            y = max(0, min(y, img_h - 1))
            w = min(w, img_w - x)
            h = min(h, img_h - y)
            
            if w <= 0 or h <= 0:
                logger.warning(f"Invalid bbox for {image_path}: {bbox}")
                return None
            
            # For now, create a simple mask from the bounding box
            # In production, this would use actual SAM2 model
            mask = self._create_bbox_mask(image_array, [x, y, w, h])
            
            # Save mask to temporary file
            mask_path = self._save_mask(mask, image_path)
            
            logger.debug(f"Generated mask for bbox {bbox} in {image_path}")
            return mask_path
            
        except Exception as e:
            logger.error(f"Segmentation failed for {image_path}", error=str(e))
            return None
    
    async def segment_multiple(self, image_path: str, bboxes: List[List[float]]) -> List[str]:
        """
        Generate masks for multiple objects in the same image
        
        Args:
            image_path: Path to the image
            bboxes: List of bounding boxes
            
        Returns:
            List of mask file paths
        """
        mask_paths = []
        
        for bbox in bboxes:
            mask_path = await self.segment(image_path, bbox)
            if mask_path:
                mask_paths.append(mask_path)
        
        return mask_paths
    
    def _create_bbox_mask(self, image: np.ndarray, bbox: List[int]) -> np.ndarray:
        """
        Create a simple mask from bounding box
        In production, replace with actual SAM2 inference
        """
        x, y, w, h = bbox
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        # Create a simple rectangular mask
        mask[y:y+h, x:x+w] = 255
        
        # Apply some smoothing to make it look more natural
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask
    
    def _create_sam2_mask(self, image: np.ndarray, bbox: List[int]) -> np.ndarray:
        """
        Actual SAM2 mask generation (placeholder)
        This would use the real SAM2 model
        """
        # Placeholder for actual SAM2 inference
        # Would involve:
        # 1. Prepare image for SAM2 input
        # 2. Generate point prompts from bbox center
        # 3. Run SAM2 inference
        # 4. Post-process the mask
        
        return self._create_bbox_mask(image, bbox)
    
    def _save_mask(self, mask: np.ndarray, original_image_path: str) -> str:
        """Save mask to temporary file"""
        # Create temporary file for mask
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_path = f.name
        
        # Convert mask to PIL Image and save
        mask_image = Image.fromarray(mask, mode='L')
        mask_image.save(temp_path, 'PNG')
        
        return temp_path
    
    async def segment_with_points(self, image_path: str, points: List[Tuple[int, int]], 
                                labels: List[int]) -> str:
        """
        Generate mask using point prompts
        
        Args:
            image_path: Path to the image
            points: List of (x, y) coordinates
            labels: List of labels (1 for positive, 0 for negative)
            
        Returns:
            Path to generated mask
        """
        try:
            image = Image.open(image_path).convert("RGB")
            image_array = np.array(image)
            
            # For now, create a simple mask around the points
            mask = np.zeros((image_array.shape[0], image_array.shape[1]), dtype=np.uint8)
            
            for point, label in zip(points, labels):
                if label == 1:  # Positive point
                    cv2.circle(mask, point, 50, 255, -1)
            
            # Apply morphological operations to smooth the mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            mask_path = self._save_mask(mask, image_path)
            return mask_path
            
        except Exception as e:
            logger.error(f"Point-based segmentation failed for {image_path}", error=str(e))
            return None
    
    def cleanup_temp_files(self, mask_paths: List[str]):
        """Clean up temporary mask files"""
        for mask_path in mask_paths:
            try:
                if os.path.exists(mask_path):
                    os.unlink(mask_path)
            except Exception as e:
                logger.warning(f"Failed to cleanup mask file {mask_path}", error=str(e))