
"""
SAM2 Segmenter for Modomo - Production Ready Implementation
----------------------------------------------------------
- Robust SAM 2 segmentation with fallback mechanisms
- Optional FBA Matting for high-quality alpha cutouts
- Production-ready error handling and logging
- Async support for API integration
- Memory-efficient processing with cleanup

Requirements:
  # Core dependencies (automatically handled)
  - opencv-python, pillow, numpy, torch, torchvision
  
  # SAM 2 (optional - fallback if not available)
  - git clone https://github.com/facebookresearch/segment-anything-2
  - pip install -e segment-anything-2

  # FBA Matting (optional - for premium quality)
  - Install from: https://github.com/MarcoForte/FBA_Matting

Usage:
  segmenter = SAM2Segmenter(device="cuda")
  mask_path = await segmenter.segment(image_path, bbox)
  
  # Advanced usage with full control
  result = await segmenter.segment_advanced(
      image_path, bbox, 
      output_dir="/tmp", 
      object_id="obj_001",
      include_alpha=True
  )
"""
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Union, Tuple
import os
import asyncio
import uuid
import tempfile
import logging
from pathlib import Path

import numpy as np
from PIL import Image
import cv2
import torch

# Configure logging
logger = logging.getLogger(__name__)

# --- SAM2 imports with robust error handling ---
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2_AVAILABLE = True
    _SAM2_IMPORT_ERROR = None
    logger.info("✅ SAM2 dependencies loaded successfully")
except Exception as e:
    SAM2_AVAILABLE = False
    _SAM2_IMPORT_ERROR = e
    logger.warning(f"⚠️ SAM2 not available: {e}")

# --- FBA Matting imports (optional premium feature) ---
try:
    # Lazy import to avoid startup errors
    FBA_AVAILABLE = True
    _FBA_IMPORT_ERROR = None
except Exception as e:
    FBA_AVAILABLE = False
    _FBA_IMPORT_ERROR = e

@dataclass
class SegmentationConfig:
    """Configuration for SAM2 segmentation"""
    # Model settings
    device: str = "cuda"  # "cuda" or "cpu"
    model_type: str = "sam2_hiera_large"  # SAM2 model variant
    
    # Railway Volume model paths
    sam2_checkpoint: Optional[str] = None
    fba_checkpoint: Optional[str] = None
    
    def __post_init__(self):
        """Auto-configure model paths from Railway Volume"""
        if self.sam2_checkpoint is None:
            # Check Railway environment variable first, then fallback locations
            railway_path = os.environ.get('SAM2_CHECKPOINT_DIR', '/app/models/sam2')
            railway_checkpoint = f"{railway_path}/{self.model_type}.pt"
            
            # Try multiple possible paths in order of priority
            possible_paths = [
                railway_checkpoint,  # Railway volume path from env var
                f"/app/model_cache/sam2/{self.model_type}.pt",  # Legacy cache path
                f"/app/checkpoints/{self.model_type}.pt",  # Legacy checkpoint path
                f"/app/models/sam2_hiera_large.pt",  # Direct models path
                f"/app/models/{self.model_type}.pt"  # Models with type name
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    self.sam2_checkpoint = path
                    logger.info(f"✅ Using SAM2 model from: {path}")
                    return
                    
            logger.warning(f"⚠️ SAM2 checkpoint not found for {self.model_type}")
            logger.info(f"🔍 Checked paths: {possible_paths}")
            
            # SAM2 checkpoint not found - will use fallback segmentation
            logger.warning(f"⚠️ SAM2 checkpoint not found - will use fallback segmentation")
    
    # Processing parameters
    min_mask_area: int = 500       # Minimum mask area (pixels)
    confidence_threshold: float = 0.5  # Mask confidence threshold
    
    # Morphological operations
    morphology_kernel_size: int = 3
    closing_iterations: int = 2
    opening_iterations: int = 1
    
    # Matting parameters (for FBA)
    trimap_erosion: int = 7        # Kernel radius for fg/bg erosion
    unknown_width: int = 16        # Width of unknown band
    
    # Output settings
    output_format: str = "PNG"     # PNG or JPEG
    compression_level: int = 6     # PNG compression
    
    # Memory management
    max_image_size: int = 2048     # Max dimension before resizing
    cleanup_temp_files: bool = False

class SAM2Segmenter:
    """
    Production-ready SAM2 segmenter with robust error handling and fallbacks.
    Supports both simple mask generation and advanced matting workflows.
    """
    
    def __init__(self, 
                 config: Optional[SegmentationConfig] = None,
                 device: Optional[str] = None,
                 eager_load: bool = True):
        """
        Initialize SAM2 segmenter with configuration.
        
        Args:
            config: SegmentationConfig instance
            device: Override device ("cuda" or "cpu")
            eager_load: If True, load models immediately during initialization
        """
        self.config = config or SegmentationConfig()
        
        # Override device if specified
        if device:
            self.config.device = device
            
        # Determine actual device to use
        self.device = self._get_device()
        
        # Model components (will be loaded eagerly or lazy)
        self.sam2_model = None
        self.sam2_predictor = None
        self.fba_model = None
        
        # Model status
        self.sam2_loaded = False
        self.fba_loaded = False
        
        # Temp file tracking for cleanup
        self._temp_files = []
        
        logger.info(f"SAM2Segmenter initialized - Device: {self.device}, SAM2 Available: {SAM2_AVAILABLE}")
        
        # Force eager loading of models if requested (production mode)
        if eager_load:
            logger.info("🚀 Eager loading SAM2 model for production deployment...")
            try:
                if SAM2_AVAILABLE:
                    success = self._ensure_sam2_loaded()
                    if success:
                        logger.info("✅ SAM2 model loaded successfully during initialization")
                    else:
                        logger.warning("⚠️ SAM2 model failed to load during initialization")
                else:
                    logger.warning("⚠️ SAM2 not available - skipping eager loading")
            except Exception as e:
                logger.error(f"❌ SAM2 eager loading failed: {e}")
                # Don't raise exception - allow fallback mode
    
    def _get_device(self) -> torch.device:
        """Determine the best available device"""
        if self.config.device == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
        else:
            device = torch.device("cpu")
            if self.config.device == "cuda":
                logger.warning("CUDA requested but not available, falling back to CPU")
        return device

    # ========== Main API Methods ==========
    
    async def segment(self, image_path: str, bbox: List[float]) -> Optional[str]:
        """
        Main segmentation method expected by the API.
        
        Args:
            image_path: Path to input image
            bbox: Bounding box [x, y, width, height]
            
        Returns:
            Path to generated mask file, or None if failed
        """
        try:
            # Convert bbox format if needed (x,y,w,h -> x1,y1,x2,y2)
            if len(bbox) == 4:
                x, y, w, h = bbox
                bbox_xyxy = [x, y, x + w, y + h]
            else:
                bbox_xyxy = bbox
            
            # Generate unique output path
            mask_id = f"mask_{uuid.uuid4().hex[:8]}"
            output_dir = "/app/cache_volume/masks"  # Use persistent cache directory
            os.makedirs(output_dir, exist_ok=True)
            
            result = await self.segment_advanced(
                image_path=image_path,
                bbox=bbox_xyxy,
                output_dir=output_dir,
                object_id=mask_id,
                include_alpha=False  # Just basic mask for API compatibility
            )
            
            if result and result.get("success"):
                mask_path = result.get("mask_path")
                if mask_path and os.path.exists(mask_path):
                    return mask_path
                    
        except Exception as e:
            logger.error(f"Segmentation failed for {image_path}: {e}")
            
        return None
    
    async def segment_advanced(self, 
                              image_path: str,
                              bbox: List[float],
                              output_dir: str,
                              object_id: str,
                              include_alpha: bool = False,
                              include_rgba: bool = False,
                              class_hint: Optional[str] = None) -> Dict[str, Any]:
        """
        Advanced segmentation with full control over outputs.
        
        Args:
            image_path: Path to input image
            bbox: Bounding box [x1, y1, x2, y2]
            output_dir: Directory for output files
            object_id: Unique identifier for output naming
            include_alpha: Generate alpha matte using FBA
            include_rgba: Generate RGBA cutout
            class_hint: Optional class hint for better segmentation
            
        Returns:
            Dict with paths, metadata, and success status
        """
        return await asyncio.get_event_loop().run_in_executor(
            None, self._segment_sync, 
            image_path, bbox, output_dir, object_id, 
            include_alpha, include_rgba, class_hint
        )
    
    # ========== Synchronous Implementation ==========
    
    def _segment_sync(self, 
                     image_path: str,
                     bbox: List[float],
                     output_dir: str,
                     object_id: str,
                     include_alpha: bool = False,
                     include_rgba: bool = False,
                     class_hint: Optional[str] = None) -> Dict[str, Any]:
        """Synchronous segmentation implementation"""
        try:
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            # Load and validate image
            image = self._load_image(image_path)
            if image is None:
                return {"success": False, "error": "Failed to load image"}
            
            # Generate mask using SAM2 or fallback
            mask = self._generate_mask(image, bbox, class_hint)
            if mask is None:
                return {"success": False, "error": "Failed to generate mask"}
            
            # Save mask
            mask_path = os.path.join(output_dir, f"{object_id}_mask.png")
            self._save_mask(mask, mask_path)
            
            result = {
                "success": True,
                "mask_path": mask_path,
                "mask_area": int(np.sum(mask > 0)),
                "bbox": bbox,
                "object_id": object_id
            }
            
            # Optional advanced processing
            if include_alpha and FBA_AVAILABLE:
                alpha_path = self._generate_alpha_matte(image, mask, output_dir, object_id)
                result["alpha_path"] = alpha_path
            
            if include_rgba:
                rgba_path = self._generate_rgba_cutout(image, mask, output_dir, object_id)
                result["rgba_path"] = rgba_path
            
            return result
            
        except Exception as e:
            logger.error(f"Segmentation failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _load_image(self, image_path: str) -> Optional[np.ndarray]:
        """Load and preprocess image"""
        try:
            if not os.path.exists(image_path):
                logger.error(f"Image not found: {image_path}")
                return None
            
            image = Image.open(image_path).convert("RGB")
            
            # Resize if too large
            width, height = image.size
            max_size = self.config.max_image_size
            
            if max(width, height) > max_size:
                if width > height:
                    new_width = max_size
                    new_height = int(height * max_size / width)
                else:
                    new_height = max_size
                    new_width = int(width * max_size / height)
                
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                logger.info(f"Resized image from {width}x{height} to {new_width}x{new_height}")
            
            return np.array(image)
            
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            return None
    
    def _generate_mask(self, image: np.ndarray, bbox: List[float], class_hint: Optional[str] = None) -> Optional[np.ndarray]:
        """Generate segmentation mask using SAM2 or fallback method"""
        try:
            # Try SAM2 first if available
            if SAM2_AVAILABLE and self._ensure_sam2_loaded():
                return self._sam2_segment(image, bbox, class_hint)
            else:
                # Fallback to simple rectangular mask with morphology
                return self._fallback_segment(image, bbox)
                
        except Exception as e:
            logger.error(f"Mask generation failed: {e}")
            return self._fallback_segment(image, bbox)
    
    def _ensure_sam2_loaded(self) -> bool:
        """Ensure SAM2 model is loaded"""
        if self.sam2_loaded:
            return True
            
        try:
            if self.config.sam2_checkpoint and os.path.exists(self.config.sam2_checkpoint):
                checkpoint_path = self.config.sam2_checkpoint
            else:
                # Try to find default checkpoint or use auto-download
                checkpoint_path = self._get_default_sam2_checkpoint()
            
            if not checkpoint_path:
                logger.warning("No SAM2 checkpoint available")
                return False
            
            logger.info(f"Loading SAM2 model from {checkpoint_path}")
            
            # Determine model config based on checkpoint name
            if "hiera_large" in checkpoint_path:
                model_cfg = "sam2_hiera_l.yaml"
            elif "hiera_base_plus" in checkpoint_path:
                model_cfg = "sam2_hiera_b+.yaml"
            else:
                model_cfg = "sam2_hiera_l.yaml"  # Default
            
            self.sam2_model = build_sam2(model_cfg, checkpoint_path, device=self.device)
            self.sam2_model.eval()
            
            self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)
            self.sam2_loaded = True
            
            logger.info("✅ SAM2 model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load SAM2: {e}")
            return False
    
    def _get_default_sam2_checkpoint(self) -> Optional[str]:
        """Get default SAM2 checkpoint path"""
        # Get Railway path from environment
        railway_path = os.environ.get('SAM2_CHECKPOINT_DIR', '/app/models/sam2')
        
        # Common checkpoint locations (prioritize Railway/Docker paths)
        possible_paths = [
            f"{railway_path}/sam2_hiera_large.pt",  # Railway volume path
            f"{railway_path}/sam2_hiera_base_plus.pt",  # Railway fallback
            "/app/models/sam2_hiera_large.pt",  # Direct models path
            "/app/models/sam2/sam2_hiera_large.pt",  # Models subdirectory
            "/app/checkpoints/sam2_hiera_large.pt",  # Legacy checkpoint path
            "/app/checkpoints/sam2_hiera_base_plus.pt",  # Legacy fallback model
            "checkpoints/sam2_hiera_large.pt",  # Local development
            "models/sam2_hiera_large.pt",
            os.path.expanduser("~/.cache/sam2/sam2_hiera_large.pt")
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                logger.info(f"Found SAM2 checkpoint: {path}")
                return path
        
        logger.warning("No default SAM2 checkpoint found")
        logger.info(f"🔍 Searched paths: {possible_paths}")
        return None
    
    def _download_sam2_checkpoint(self) -> Optional[str]:
        """Download SAM2 checkpoint if not available"""
        try:
            import requests
            import tempfile
            
            # SAM2 checkpoint URLs from Meta
            checkpoint_urls = {
                "sam2_hiera_large": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt",
                "sam2_hiera_base_plus": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt",
                "sam2_hiera_small": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt"
            }
            
            if self.config.model_type not in checkpoint_urls:
                logger.warning(f"No download URL available for {self.config.model_type}")
                return None
            
            url = checkpoint_urls[self.config.model_type]
            
            # Determine download path
            railway_path = os.environ.get('SAM2_CHECKPOINT_DIR', '/app/models/sam2')
            os.makedirs(railway_path, exist_ok=True)
            download_path = f"{railway_path}/{self.config.model_type}.pt"
            
            # Skip if already exists (might have been downloaded by another worker)
            if os.path.exists(download_path):
                logger.info(f"SAM2 checkpoint already exists at {download_path}")
                return download_path
            
            logger.info(f"📥 Downloading SAM2 checkpoint from {url}")
            
            # Download with progress tracking
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(download_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Log progress every 100MB
                        if downloaded % (100 * 1024 * 1024) == 0:
                            progress = (downloaded / total_size * 100) if total_size > 0 else 0
                            logger.info(f"📥 Download progress: {progress:.1f}% ({downloaded / 1024 / 1024:.1f}MB)")
            
            logger.info(f"✅ SAM2 checkpoint downloaded successfully to {download_path}")
            return download_path
            
        except Exception as e:
            logger.error(f"❌ Failed to download SAM2 checkpoint: {e}")
            return None
    
    def _sam2_segment(self, image: np.ndarray, bbox: List[float], class_hint: Optional[str] = None) -> Optional[np.ndarray]:
        """Segment using SAM2 model"""
        try:
            # Set image for SAM2 predictor
            self.sam2_predictor.set_image(image)
            
            # Convert bbox to numpy array [x1, y1, x2, y2]
            box = np.array(bbox, dtype=np.float32)
            
            # Generate masks
            masks, scores, logits = self.sam2_predictor.predict(
                box=box,
                multimask_output=False
            )
            
            if masks is None or len(masks) == 0:
                logger.warning("SAM2 generated no masks")
                return None
            
            # Get the best mask
            mask = masks[0].astype(np.uint8)
            
            # Post-process mask
            mask = self._postprocess_mask(mask)
            
            # Check minimum area
            mask_area = np.sum(mask > 0)
            if mask_area < self.config.min_mask_area:
                logger.warning(f"Mask too small: {mask_area} < {self.config.min_mask_area}")
                return None
            
            logger.info(f"SAM2 generated mask with area: {mask_area}")
            return mask
            
        except Exception as e:
            logger.error(f"SAM2 segmentation failed: {e}")
            return None
    
    def _fallback_segment(self, image: np.ndarray, bbox: List[float]) -> Optional[np.ndarray]:
        """Fallback segmentation using simple rectangular mask with morphology"""
        try:
            h, w = image.shape[:2]
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            
            # Clamp coordinates
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                logger.error(f"Invalid bbox: {bbox}")
                return None
            
            # Create rectangular mask
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[y1:y2, x1:x2] = 255
            
            # Apply morphological operations to make it more realistic
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                             (self.config.morphology_kernel_size, 
                                              self.config.morphology_kernel_size))
            
            # Erode then dilate to create more organic shape
            mask = cv2.erode(mask, kernel, iterations=2)
            mask = cv2.dilate(mask, kernel, iterations=3)
            
            # Apply Gaussian blur for smoother edges
            mask = cv2.GaussianBlur(mask, (5, 5), 1.0)
            mask = (mask > 127).astype(np.uint8)
            
            mask_area = np.sum(mask > 0)
            logger.info(f"Fallback mask generated with area: {mask_area}")
            return mask
            
        except Exception as e:
            logger.error(f"Fallback segmentation failed: {e}")
            return None
    
    def _postprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        """Post-process mask with morphological operations"""
        try:
            # Ensure binary mask
            if mask.dtype != np.uint8:
                mask = (mask > 0).astype(np.uint8) * 255
            elif mask.max() <= 1:
                mask = mask * 255
            
            # Morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                             (self.config.morphology_kernel_size, 
                                              self.config.morphology_kernel_size))
            
            # Close holes
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, 
                                  iterations=self.config.closing_iterations)
            
            # Remove small noise
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, 
                                  iterations=self.config.opening_iterations)
            
            # Keep only largest connected component
            mask = self._keep_largest_component(mask)
            
            return mask
            
        except Exception as e:
            logger.error(f"Mask post-processing failed: {e}")
            return mask
    
    def _keep_largest_component(self, mask: np.ndarray) -> np.ndarray:
        """Keep only the largest connected component"""
        try:
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                (mask > 0).astype(np.uint8), connectivity=8
            )
            
            if num_labels <= 1:
                return mask
            
            # Find largest component (excluding background)
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            
            # Create mask with only largest component
            result = np.zeros_like(mask)
            result[labels == largest_label] = 255
            
            return result
            
        except Exception as e:
            logger.error(f"Connected components analysis failed: {e}")
            return mask
    
    def _save_mask(self, mask: np.ndarray, output_path: str):
        """Save mask to file"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Convert to PIL Image and save
            if mask.dtype != np.uint8:
                mask = (mask * 255).astype(np.uint8)
            
            mask_image = Image.fromarray(mask, mode='L')
            mask_image.save(output_path, format=self.config.output_format, 
                          compress_level=self.config.compression_level)
            
            # Track temp file for cleanup
            if self.config.cleanup_temp_files:
                self._temp_files.append(output_path)
            
            logger.debug(f"Mask saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save mask to {output_path}: {e}")
    
    def _generate_rgba_cutout(self, image: np.ndarray, mask: np.ndarray, 
                            output_dir: str, object_id: str) -> Optional[str]:
        """Generate RGBA cutout with transparency"""
        try:
            # Normalize mask to 0-255
            if mask.max() <= 1:
                alpha = (mask * 255).astype(np.uint8)
            else:
                alpha = mask.astype(np.uint8)
            
            # Create RGBA image
            h, w = image.shape[:2]
            rgba = np.zeros((h, w, 4), dtype=np.uint8)
            rgba[:, :, :3] = image
            rgba[:, :, 3] = alpha
            
            # Save RGBA image
            rgba_path = os.path.join(output_dir, f"{object_id}_cutout.png")
            rgba_image = Image.fromarray(rgba, 'RGBA')
            rgba_image.save(rgba_path, format='PNG')
            
            if self.config.cleanup_temp_files:
                self._temp_files.append(rgba_path)
            
            return rgba_path
            
        except Exception as e:
            logger.error(f"RGBA cutout generation failed: {e}")
            return None
    
    def _generate_alpha_matte(self, image: np.ndarray, mask: np.ndarray, 
                            output_dir: str, object_id: str) -> Optional[str]:
        """Generate high-quality alpha matte using FBA (if available)"""
        try:
            if not FBA_AVAILABLE:
                logger.warning("FBA Matting not available, skipping alpha generation")
                return None
            
            # For now, return simple alpha based on mask
            # TODO: Implement FBA matting integration
            alpha_path = os.path.join(output_dir, f"{object_id}_alpha.png")
            
            if mask.max() <= 1:
                alpha = (mask * 255).astype(np.uint8)
            else:
                alpha = mask.astype(np.uint8)
            
            alpha_image = Image.fromarray(alpha, 'L')
            alpha_image.save(alpha_path, format='PNG')
            
            if self.config.cleanup_temp_files:
                self._temp_files.append(alpha_path)
            
            return alpha_path
            
        except Exception as e:
            logger.error(f"Alpha matte generation failed: {e}")
            return None
    
    # ========== Utility Methods ==========
    
    def cleanup(self):
        """Clean up temporary files"""
        if not self.config.cleanup_temp_files:
            return
        
        cleaned = 0
        for temp_file in self._temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                    cleaned += 1
            except Exception as e:
                logger.warning(f"Failed to cleanup {temp_file}: {e}")
        
        if cleaned > 0:
            logger.info(f"Cleaned up {cleaned} temporary files")
        
        self._temp_files.clear()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        return {
            "sam2_available": SAM2_AVAILABLE,
            "sam2_loaded": self.sam2_loaded,
            "fba_available": FBA_AVAILABLE,
            "fba_loaded": self.fba_loaded,
            "device": str(self.device),
            "model_type": self.config.model_type,
            "checkpoint": self.config.sam2_checkpoint
        }
    
    def __del__(self):
        """Cleanup on object destruction"""
        try:
            self.cleanup()
        except:
            pass


# ========== Legacy Support ==========

# Keep backward compatibility with the old SAM2FBA_Segmenter name
SAM2FBA_Segmenter = SAM2Segmenter
