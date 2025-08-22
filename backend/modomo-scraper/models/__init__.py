"""
Modomo AI Models Package

This package contains all AI model implementations for the Modomo scraper:
- SAM2 segmentation models
- GroundingDINO object detection  
- CLIP embeddings
- Color extraction
- Fusion and boundary refinement
"""

__version__ = "1.0.0"

# Make key classes available at package level
try:
    from .sam2_segmenter import SAM2Segmenter, SegmentationConfig
    from .grounding_dino import GroundingDINODetector
    from .clip_embedder import CLIPEmbedder
    from .color_extractor import ColorExtractor
    from .depth_estimator import DepthEstimator, DepthConfig
    from .edge_detector import EdgeDetector
except ImportError as e:
    # Graceful fallback for missing dependencies
    print(f"⚠️ Some AI models not available: {e}")

__all__ = [
    'SAM2Segmenter', 
    'SegmentationConfig',
    'GroundingDINODetector', 
    'CLIPEmbedder',
    'ColorExtractor',
    'DepthEstimator',
    'DepthConfig', 
    'EdgeDetector'
]