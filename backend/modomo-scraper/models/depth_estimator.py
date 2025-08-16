"""
Depth Estimation with Depth Anything V2
---------------------------------------
- State-of-the-art monocular depth estimation using Depth Anything V2
- CPU/GPU optimization support  
- Memory management for constrained environments
- Colormap visualization for depth maps

Usage:
  estimator = DepthEstimator(DepthConfig())
  depth_path = await estimator.estimate_depth(image_path)
"""

import os
import asyncio
import tempfile
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import structlog

# Configure logging
logger = structlog.get_logger(__name__)

@dataclass
class DepthConfig:
    """Configuration for depth estimation"""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    cpu_optimization: bool = True
    reduce_precision: bool = False  # Use float16 for GPU memory savings
    model_name: str = "depth-anything/Depth-Anything-V2-Large"  # Depth Anything V2 model
    max_image_size: int = 518  # Optimal input size for Depth Anything V2
    colormap: str = "inferno"  # Colormap for visualization

class DepthEstimator:
    """Depth Anything V2 depth estimator with optimizations"""
    
    def __init__(self, config: DepthConfig):
        self.config = config
        self.model = None
        self.processor = None
        self.is_available = False
        
        logger.info(f"🔍 Initializing DepthEstimator on {config.device}")
        self._setup_model()
    
    def _setup_model(self):
        """Initialize Depth Anything V2 model"""
        try:
            # Import Depth Anything V2
            from depth_anything_v2.dpt import DepthAnythingV2
            
            # Model configurations from official repo
            model_configs = {
                'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
                'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
                'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
                'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
            }
            
            # Use Large model for best quality (vitl)
            encoder = 'vitl'
            self.model = DepthAnythingV2(**model_configs[encoder])
            
            # Try to load pretrained weights
            model_path = self._download_model(encoder)
            if model_path and os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location='cpu')
                self.model.load_state_dict(checkpoint)
                logger.info("✅ Loaded Depth Anything V2 pretrained weights")
            else:
                logger.warning("⚠️ Using Depth Anything V2 without pretrained weights")
            
            # Move to device and set precision
            self.model = self.model.to(self.config.device).eval()
            if self.config.reduce_precision and self.config.device == "cuda":
                self.model = self.model.half()
                logger.info("🔧 Using half precision for GPU memory savings")
            
            # CPU optimizations
            if self.config.device == "cpu" and self.config.cpu_optimization:
                torch.set_num_threads(min(4, torch.get_num_threads()))
                logger.info("🔧 CPU optimizations enabled")
            
            self.is_available = True
            logger.info(f"✅ Depth Anything V2 initialized successfully on {self.config.device}")
            
        except ImportError as e:
            logger.error(f"❌ Depth Anything V2 not available: {e}")
            logger.error("💡 Install with: git clone https://github.com/DepthAnything/Depth-Anything-V2")
            raise RuntimeError("Depth Anything V2 is required for depth estimation")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Depth Anything V2: {e}")
            raise
    
    def _download_model(self, encoder: str) -> Optional[str]:
        """Download Depth Anything V2 model weights"""
        try:
            # Create model cache directory
            cache_dir = os.getenv("MODEL_CACHE_DIR", "/tmp/depth_anything_v2")
            os.makedirs(cache_dir, exist_ok=True)
            
            model_file = os.path.join(cache_dir, f"depth_anything_v2_{encoder}.pth")
            
            if not os.path.exists(model_file):
                logger.info(f"📦 Downloading Depth Anything V2 {encoder} model weights...")
                import urllib.request
                
                # Official Hugging Face model URLs
                model_urls = {
                    'vits': 'https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth',
                    'vitb': 'https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth',
                    'vitl': 'https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth',
                    'vitg': 'https://huggingface.co/depth-anything/Depth-Anything-V2-Giant/resolve/main/depth_anything_v2_vitg.pth'
                }
                
                if encoder in model_urls:
                    urllib.request.urlretrieve(model_urls[encoder], model_file)
                    logger.info("✅ Model weights downloaded successfully")
                else:
                    logger.error(f"❌ Unknown encoder: {encoder}")
                    return None
            
            return model_file
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to download model weights: {e}")
            return None
    
    # Preprocessing and postprocessing handled internally by Depth Anything V2
    
    def _create_depth_visualization(self, depth_array: np.ndarray, output_path: str) -> bool:
        """Create colorized depth map visualization"""
        try:
            # Apply colormap
            colormap = cm.get_cmap(self.config.colormap)
            colored_depth = colormap(depth_array)
            
            # Convert to 8-bit RGB
            colored_depth_rgb = (colored_depth[:, :, :3] * 255).astype(np.uint8)
            
            # Save as PNG
            depth_image = Image.fromarray(colored_depth_rgb, 'RGB')
            depth_image.save(output_path, 'PNG', optimize=True)
            
            logger.info(f"✅ Depth visualization saved: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create depth visualization: {e}")
            return False
    
    async def estimate_depth(self, image_path: str, output_dir: Optional[str] = None) -> Optional[str]:
        """
        Estimate depth for an image using Depth Anything V2
        
        Args:
            image_path: Path to input image
            output_dir: Output directory for depth map (optional)
            
        Returns:
            Path to generated depth map or None if failed
        """
        if not self.is_available:
            logger.error("❌ Depth estimator not available")
            return None
        
        try:
            # Load and validate image
            if not os.path.exists(image_path):
                logger.error(f"❌ Image not found: {image_path}")
                return None
            
            image = Image.open(image_path)
            logger.info(f"🖼️ Processing image: {image.size}")
            
            # Convert PIL image to OpenCV format (Depth Anything V2 expects BGR)
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Run depth estimation using official API
            with torch.no_grad():
                # Use the model's infer_image method
                depth_array = self.model.infer_image(image_cv)
            
            # Normalize depth array to 0-1 range for visualization
            depth_normalized = (depth_array - depth_array.min()) / (depth_array.max() - depth_array.min())
            
            # Create output path
            if output_dir is None:
                output_dir = tempfile.gettempdir()
            
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate filename
            base_name = Path(image_path).stem
            output_path = os.path.join(output_dir, f"{base_name}_depth.png")
            
            # Create and save visualization
            if self._create_depth_visualization(depth_normalized, output_path):
                logger.info(f"✅ Depth estimation completed: {output_path}")
                return output_path
            else:
                return None
            
        except Exception as e:
            logger.error(f"❌ Depth estimation failed: {e}")
            return None
        finally:
            # Cleanup GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def get_info(self) -> Dict[str, Any]:
        """Get depth estimator information"""
        return {
            "model": "Depth Anything V2",
            "version": "Large (ViT-L)",
            "device": self.config.device,
            "available": self.is_available,
            "input_size": self.config.max_image_size,
            "colormap": self.config.colormap,
            "cpu_optimized": self.config.cpu_optimization,
            "precision": "half" if self.config.reduce_precision else "full"
        }