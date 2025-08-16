"""
CV2 Canny Edge Detection for Modomo - Interior Scene Analysis
------------------------------------------------------------
- Optimized Canny edge detection for furniture and architectural features
- Adaptive thresholding for robust edge detection
- Multiple preprocessing options for different scene types
- Async support for API integration
- Memory efficient (CPU-based processing)

Requirements:
  - opencv-python (cv2)
  - numpy, pillow
  - scipy (optional, for advanced filtering)

Usage:
  detector = EdgeDetector()
  edge_map_path = await detector.detect_edges(image_path)
"""

import os
import asyncio
import tempfile
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

import numpy as np
import cv2
from PIL import Image

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class EdgeConfig:
    """Configuration for edge detection"""
    # Canny parameters
    low_threshold: int = 50
    high_threshold: int = 150
    aperture_size: int = 3
    l2_gradient: bool = True
    
    # Preprocessing
    blur_kernel_size: int = 5
    blur_sigma: float = 1.0
    enable_denoising: bool = True
    
    # Output settings
    output_format: str = "binary"  # "binary", "grayscale", "colored"
    line_thickness: int = 1
    invert_edges: bool = False  # True = white edges on black background
    
    # Advanced options
    adaptive_thresholds: bool = True
    multi_scale: bool = False
    enhance_contrast: bool = True


class EdgeDetector:
    """CV2 Canny-based edge detection for interior scenes"""
    
    def __init__(self, config: Optional[EdgeConfig] = None):
        self.config = config or EdgeConfig()
        
        logger.info("📐 Initializing EdgeDetector with CV2 Canny")
        logger.info(f"🔧 Thresholds: {self.config.low_threshold}-{self.config.high_threshold}")
    
    def get_detector_info(self) -> Dict[str, Any]:
        """Get information about the edge detector"""
        return {
            "detector_available": True,
            "detector_name": "CV2 Canny",
            "method": "CPU-based",
            "low_threshold": self.config.low_threshold,
            "high_threshold": self.config.high_threshold,
            "adaptive_thresholds": self.config.adaptive_thresholds,
            "output_format": self.config.output_format
        }
    
    async def detect_edges(self, image_path: str, output_dir: Optional[str] = None) -> Optional[str]:
        """
        Generate edge map from image using Canny edge detection
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save edge map (uses temp if None)
            
        Returns:
            Path to generated edge map PNG file
        """
        try:
            # Prepare output directory
            if output_dir is None:
                output_dir = tempfile.gettempdir()
            
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate unique filename
            image_name = Path(image_path).stem
            edge_map_path = os.path.join(output_dir, f"{image_name}_edges.png")
            
            logger.info(f"📐 Generating edge map for {image_path}")
            
            # Load and preprocess image
            image = await self._load_image(image_path)
            if image is None:
                return None
            
            # Apply edge detection
            edges = await self._apply_canny_edge_detection(image)
            
            # Post-process and save
            await self._save_edge_visualization(edges, edge_map_path)
            
            logger.info(f"✅ Edge map saved: {edge_map_path}")
            return edge_map_path
            
        except Exception as e:
            logger.error(f"❌ Edge detection failed: {e}")
            return None
    
    async def _load_image(self, image_path: str) -> Optional[np.ndarray]:
        """Load and preprocess image for edge detection"""
        try:
            # Load image
            if image_path.startswith('http'):
                # Handle URL images
                import aiohttp
                import aiofiles
                
                temp_path = os.path.join(tempfile.gettempdir(), f"temp_edge_{os.getpid()}.jpg")
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(image_path) as response:
                        if response.status == 200:
                            async with aiofiles.open(temp_path, 'wb') as f:
                                async for chunk in response.content.iter_chunked(8192):
                                    await f.write(chunk)
                            image_path = temp_path
                        else:
                            logger.error(f"❌ Failed to download image: HTTP {response.status}")
                            return None
            
            # Load with OpenCV
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"❌ Failed to load image: {image_path}")
                return None
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Clean up temporary file if created
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.unlink(temp_path)
            
            return image
            
        except Exception as e:
            logger.error(f"❌ Failed to load image {image_path}: {e}")
            return None
    
    async def _apply_canny_edge_detection(self, image: np.ndarray) -> np.ndarray:
        """Apply Canny edge detection with preprocessing"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Enhance contrast if enabled
            if self.config.enhance_contrast:
                gray = cv2.equalizeHist(gray)
                logger.debug("🔧 Applied histogram equalization")
            
            # Apply denoising if enabled
            if self.config.enable_denoising:
                gray = cv2.bilateralFilter(gray, 9, 75, 75)
                logger.debug("🔧 Applied bilateral filtering for denoising")
            
            # Apply Gaussian blur
            if self.config.blur_kernel_size > 0:
                gray = cv2.GaussianBlur(
                    gray, 
                    (self.config.blur_kernel_size, self.config.blur_kernel_size),
                    self.config.blur_sigma
                )
                logger.debug(f"🔧 Applied Gaussian blur: kernel={self.config.blur_kernel_size}")
            
            # Determine thresholds
            if self.config.adaptive_thresholds:
                low_threshold, high_threshold = self._compute_adaptive_thresholds(gray)
                logger.debug(f"🔧 Adaptive thresholds: {low_threshold}-{high_threshold}")
            else:
                low_threshold = self.config.low_threshold
                high_threshold = self.config.high_threshold
            
            # Apply Canny edge detection
            edges = cv2.Canny(
                gray,
                low_threshold,
                high_threshold,
                apertureSize=self.config.aperture_size,
                L2gradient=self.config.l2_gradient
            )
            
            # Multi-scale edge detection (optional)
            if self.config.multi_scale:
                edges = await self._apply_multiscale_edges(gray, edges)
            
            logger.debug("✅ Canny edge detection completed")
            return edges
            
        except Exception as e:
            logger.error(f"❌ Edge detection processing failed: {e}")
            raise
    
    def _compute_adaptive_thresholds(self, gray: np.ndarray) -> Tuple[int, int]:
        """Compute adaptive thresholds based on image statistics"""
        try:
            # Compute image gradient magnitude
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
            
            # Use percentiles for adaptive thresholds
            low_percentile = np.percentile(gradient_magnitude, 10)
            high_percentile = np.percentile(gradient_magnitude, 90)
            
            # Scale to reasonable Canny thresholds
            low_threshold = max(int(low_percentile * 0.5), 30)
            high_threshold = min(int(high_percentile * 0.8), 200)
            
            # Ensure high > low
            if high_threshold <= low_threshold:
                high_threshold = low_threshold + 50
            
            return low_threshold, high_threshold
            
        except Exception as e:
            logger.warning(f"Adaptive threshold computation failed, using defaults: {e}")
            return self.config.low_threshold, self.config.high_threshold
    
    async def _apply_multiscale_edges(self, gray: np.ndarray, edges: np.ndarray) -> np.ndarray:
        """Apply multiscale edge detection for better results"""
        try:
            # Create different scales
            scales = [0.5, 1.0, 1.5]
            multiscale_edges = edges.copy()
            
            for scale in scales:
                if scale == 1.0:
                    continue
                
                # Resize image
                h, w = gray.shape
                new_h, new_w = int(h * scale), int(w * scale)
                scaled_gray = cv2.resize(gray, (new_w, new_h))
                
                # Apply Canny at this scale
                scaled_edges = cv2.Canny(
                    scaled_gray,
                    self.config.low_threshold,
                    self.config.high_threshold,
                    apertureSize=self.config.aperture_size
                )
                
                # Resize back to original size
                scaled_edges = cv2.resize(scaled_edges, (w, h))
                
                # Combine with main edges
                multiscale_edges = cv2.bitwise_or(multiscale_edges, scaled_edges)
            
            return multiscale_edges
            
        except Exception as e:
            logger.warning(f"Multiscale processing failed, using single scale: {e}")
            return edges
    
    async def _save_edge_visualization(self, edges: np.ndarray, output_path: str):
        """Save edge map with proper formatting"""
        try:
            # Apply post-processing based on output format
            if self.config.output_format == "binary":
                # Pure binary edges
                output_image = edges
            elif self.config.output_format == "grayscale":
                # Grayscale edges
                output_image = edges
            elif self.config.output_format == "colored":
                # Colored edges (white edges on transparent background)
                output_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGBA)
                # Make non-edge pixels transparent
                output_image[edges == 0] = [0, 0, 0, 0]  # Transparent
                output_image[edges > 0] = [255, 255, 255, 255]  # White edges
            else:
                output_image = edges
            
            # Invert if requested (white edges on black background)
            if self.config.invert_edges and self.config.output_format != "colored":
                output_image = cv2.bitwise_not(output_image)
            
            # Enhance line thickness if requested
            if self.config.line_thickness > 1:
                kernel = np.ones((self.config.line_thickness, self.config.line_thickness), np.uint8)
                if self.config.output_format == "colored":
                    # Handle RGBA separately
                    alpha_channel = output_image[:, :, 3]
                    alpha_channel = cv2.dilate(alpha_channel, kernel, iterations=1)
                    output_image[:, :, 3] = alpha_channel
                else:
                    output_image = cv2.dilate(output_image, kernel, iterations=1)
            
            # Save as PNG
            if self.config.output_format == "colored":
                # Save RGBA
                pil_image = Image.fromarray(output_image, 'RGBA')
                pil_image.save(output_path, 'PNG', optimize=True)
            else:
                # Save grayscale
                cv2.imwrite(output_path, output_image)
            
            logger.debug(f"💾 Edge visualization saved: {output_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save edge visualization: {e}")
            raise
    
    async def detect_edges_with_metadata(self, image_path: str, output_dir: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Generate edge map with additional metadata
        
        Returns:
            Dictionary with edge map path and metadata
        """
        edge_map_path = await self.detect_edges(image_path, output_dir)
        
        if edge_map_path is None:
            return None
        
        return {
            "edge_map_path": edge_map_path,
            "detector_name": "CV2 Canny",
            "low_threshold": self.config.low_threshold,
            "high_threshold": self.config.high_threshold,
            "adaptive_thresholds": self.config.adaptive_thresholds,
            "output_format": self.config.output_format,
            "generated_at": asyncio.get_event_loop().time()
        }


# Async convenience function
async def detect_edges_simple(image_path: str, output_dir: Optional[str] = None, **kwargs) -> Optional[str]:
    """
    Simple async function for edge detection
    
    Args:
        image_path: Path to input image
        output_dir: Output directory (uses temp if None)
        **kwargs: Additional configuration parameters
    
    Returns:
        Path to generated edge map
    """
    config = EdgeConfig(**kwargs)
    detector = EdgeDetector(config)
    
    return await detector.detect_edges(image_path, output_dir)


if __name__ == "__main__":
    # Test the edge detector
    import sys
    
    async def test_edge_detection():
        if len(sys.argv) != 2:
            print("Usage: python edge_detector.py <image_path>")
            return
        
        image_path = sys.argv[1]
        print(f"📐 Testing edge detection on: {image_path}")
        
        result = await detect_edges_simple(image_path)
        if result:
            print(f"✅ Edge map generated: {result}")
        else:
            print("❌ Edge detection failed")
    
    asyncio.run(test_edge_detection())