"""
AI Detection service for object detection, segmentation, and embedding
"""
import os
import uuid
try:
    import aiohttp
    import aiofiles
except ImportError:
    aiohttp = None
    aiofiles = None
from datetime import datetime
from typing import List, Dict, Any, Optional
import structlog

from utils.serialization import make_json_serializable

logger = structlog.get_logger(__name__)


class DetectionService:
    """Service for AI detection pipeline operations"""
    
    def __init__(self, detector=None, segmenter=None, embedder=None, color_extractor=None, 
                 depth_estimator=None, edge_detector=None, map_generator=None):
        self.detector = detector
        self.segmenter = segmenter
        self.embedder = embedder
        self.color_extractor = color_extractor
        
        # Map generation components
        self.depth_estimator = depth_estimator
        self.edge_detector = edge_detector
        self.map_generator = map_generator
        
        # Initialize map generator if components are available
        if not self.map_generator and (self.depth_estimator or self.edge_detector):
            self._init_map_generator()
    
    def is_available(self) -> bool:
        """Check if all AI models are available"""
        return all([
            self.detector is not None,
            self.segmenter is not None,
            self.embedder is not None
        ])
    
    def get_status(self) -> Dict[str, Any]:
        """Get detailed status of AI models"""
        status = {
            "detector_loaded": self.detector is not None,
            "segmenter_loaded": self.segmenter is not None,
            "embedder_loaded": self.embedder is not None,
            "color_extractor_loaded": self.color_extractor is not None
        }
        
        # Get detailed model info if available
        if self.detector and hasattr(self.detector, 'get_detector_status'):
            status["detector_details"] = self.detector.get_detector_status()
        
        if self.segmenter and hasattr(self.segmenter, 'get_model_info'):
            status["segmenter_details"] = self.segmenter.get_model_info()
        
        return status
    
    async def download_image(self, image_url: str, job_id: str) -> Optional[str]:
        """Download image from URL to temporary file"""
        try:
            image_path = f"/tmp/scene_{job_id}.jpg"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(image_url) as response:
                    if response.status == 200:
                        async with aiofiles.open(image_path, 'wb') as f:
                            async for chunk in response.content.iter_chunked(8192):
                                await f.write(chunk)
                        logger.info(f"✅ Downloaded image to {image_path}")
                        return image_path
                    else:
                        logger.error(f"❌ Failed to download image: HTTP {response.status}")
                        return None
        except Exception as e:
            logger.error(f"❌ Image download failed: {e}")
            return None
    
    async def run_detection_pipeline(self, image_url: str, job_id: str, taxonomy: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """Complete AI pipeline: detect -> segment -> embed -> colors"""
        try:
            logger.info(f"Starting detection pipeline for {image_url}")
            
            # Check if models are available
            if not self.is_available():
                logger.error("❌ AI models not available")
                return []
            
            # Step 1: Download image from URL
            image_path = await self.download_image(image_url, job_id)
            if not image_path:
                return []
            
            # Step 2: Object detection
            detections = await self.detector.detect_objects(image_path, taxonomy)
            if not detections:
                logger.warning(f"⚠️ No objects detected in {image_url}")
                return []
            
            # Clean up detection results to ensure JSON serialization
            detections = [make_json_serializable(detection) for detection in detections]
            
            # Step 3: Process each detection (segmentation, embedding, colors)
            for i, detection in enumerate(detections):
                try:
                    # Generate mask
                    mask_path = await self.segmenter.segment(image_path, detection['bbox'])
                    detection['mask_path'] = mask_path
                    
                    if mask_path:
                        logger.debug(f"✅ Generated mask for detection {i+1}: {mask_path}")
                    else:
                        logger.warning(f"⚠️ Failed to generate mask for detection {i+1}")
                        detection['mask_path'] = None
                
                    # Generate embedding
                    embedding = await self.embedder.embed_object(image_path, detection['bbox'])
                    detection['embedding'] = make_json_serializable(embedding)
                    
                    if embedding:
                        logger.debug(f"✅ Generated embedding for detection {i+1}")
                    else:
                        logger.warning(f"⚠️ Failed to generate embedding for detection {i+1}")
                        detection['embedding'] = []
                    
                    # Extract colors from object crop if color extractor is available
                    if self.color_extractor:
                        try:
                            color_data = await self.color_extractor.extract_colors(image_path, detection['bbox'])
                            detection['color_data'] = make_json_serializable(color_data)
                            
                            # Add color-based tags
                            if color_data and color_data.get('colors'):
                                color_names = [c.get('name') for c in color_data['colors'] if c.get('name')]
                                detection['tags'] = detection.get('tags', []) + color_names[:3]  # Add top 3 color names
                                logger.debug(f"✅ Extracted colors for detection {i+1}")
                        except Exception as color_error:
                            logger.warning(f"⚠️ Color extraction failed for detection {i+1}: {color_error}")
                            detection['color_data'] = None
                    else:
                        detection['color_data'] = None
                        
                except Exception as processing_error:
                    logger.error(f"❌ Processing failed for detection {i+1}: {processing_error}")
                    detection['mask_path'] = None
                    detection['embedding'] = []
                    detection['color_data'] = None
            
            # Cleanup temporary image file
            try:
                if os.path.exists(image_path):
                    os.unlink(image_path)
                    logger.debug(f"🧹 Cleaned up temporary image: {image_path}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup {image_path}: {cleanup_error}")
            
            # Cleanup segmenter temp files if available
            if self.segmenter and hasattr(self.segmenter, 'cleanup'):
                try:
                    self.segmenter.cleanup()
                except Exception as seg_cleanup_error:
                    logger.warning(f"Segmenter cleanup failed: {seg_cleanup_error}")
            
            logger.info(f"Detection pipeline complete: {len(detections)} objects processed")
            return detections
            
        except Exception as e:
            logger.error(f"Detection pipeline failed: {e}")
            
            # Cleanup on error
            try:
                if 'image_path' in locals() and os.path.exists(image_path):
                    os.unlink(image_path)
            except:
                pass
            
            return []
    
    async def extract_colors_from_url(self, image_url: str, bbox: Optional[List[float]] = None) -> Dict[str, Any]:
        """Extract colors from image URL with optional bounding box crop"""
        if not self.color_extractor:
            return {"error": "Color extraction service is not available"}
        
        try:
            # Download image temporarily
            temp_path = f"/tmp/color_analysis_{uuid.uuid4().hex}.jpg"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(image_url) as response:
                    if response.status == 200:
                        async with aiofiles.open(temp_path, 'wb') as f:
                            async for chunk in response.content.iter_chunked(8192):
                                await f.write(chunk)
                    else:
                        return {"error": f"Failed to fetch image: HTTP {response.status}"}
            
            # Extract colors
            color_data = await self.color_extractor.extract_colors(temp_path, bbox)
            
            # Cleanup
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            
            return color_data
            
        except Exception as e:
            logger.error(f"Color extraction API failed: {e}")
            return {"error": str(e)}
    
    def _init_map_generator(self):
        """Initialize map generator with available components"""
        try:
            from models.map_generator import MapGenerator
            self.map_generator = MapGenerator(
                depth_estimator=self.depth_estimator,
                edge_detector=self.edge_detector
            )
            logger.info("✅ Map generator initialized with available components")
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize map generator: {e}")
            self.map_generator = None
    
    async def generate_scene_maps(self, image_url: str, scene_id: str, map_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generate depth and edge maps for a scene"""
        if not self.map_generator:
            return {"error": "Map generator not available", "success": False}
        
        try:
            # Download image temporarily for map generation
            image_path = await self.download_image(image_url, f"maps_{scene_id}")
            if not image_path:
                return {"error": "Failed to download image", "success": False}
            
            # Generate maps
            results = await self.map_generator.generate_all_maps(image_path, scene_id, map_types)
            
            # Cleanup downloaded image
            try:
                if os.path.exists(image_path):
                    os.unlink(image_path)
            except:
                pass
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Scene map generation failed: {e}")
            return {"error": str(e), "success": False}
    
    async def generate_depth_map_only(self, image_url: str, scene_id: str) -> Dict[str, Any]:
        """Generate only depth map for a scene"""
        return await self.generate_scene_maps(image_url, scene_id, ["depth"])
    
    async def generate_edge_map_only(self, image_url: str, scene_id: str) -> Dict[str, Any]:
        """Generate only edge map for a scene"""
        return await self.generate_scene_maps(image_url, scene_id, ["edge"])
    
    def get_map_generation_status(self) -> Dict[str, Any]:
        """Get status of map generation components"""
        if self.map_generator:
            return self.map_generator.get_status()
        else:
            return {
                "map_generator_available": False,
                "depth_estimator": {"available": self.depth_estimator is not None},
                "edge_detector": {"available": self.edge_detector is not None},
                "error": "Map generator not initialized"
            }