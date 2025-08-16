"""
Map Generator - Orchestration service for depth and edge map generation
----------------------------------------------------------------------
- Coordinates depth estimation and edge detection
- Handles R2 upload for generated maps
- Manages sequential processing for memory efficiency
- Integrates with existing DetectionService architecture
- Provides unified interface for map generation

Usage:
  generator = MapGenerator(depth_estimator, edge_detector)
  results = await generator.generate_all_maps(image_path, scene_id)
"""

import os
import asyncio
import tempfile
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from datetime import datetime

import torch

# Configure logging
logger = logging.getLogger(__name__)

# Import performance monitoring
try:
    from utils.performance_monitor import get_performance_monitor
    PERFORMANCE_MONITORING = True
except ImportError:
    PERFORMANCE_MONITORING = False
    logger.warning("⚠️ Performance monitoring not available")

@dataclass
class MapGenerationConfig:
    """Configuration for map generation"""
    output_dir: Optional[str] = None
    r2_bucket: str = "training-data"
    enable_cleanup: bool = True
    max_concurrent_maps: int = 1  # Sequential for memory efficiency
    upload_to_r2: bool = True
    keep_local_copies: bool = False


class MapGenerator:
    """Orchestrates generation of depth and edge maps"""
    
    def __init__(
        self,
        depth_estimator=None,
        edge_detector=None,
        config: Optional[MapGenerationConfig] = None
    ):
        self.depth_estimator = depth_estimator
        self.edge_detector = edge_detector
        self.config = config or MapGenerationConfig()
        
        # Import R2 uploader
        self.r2_uploader = None
        self._init_r2_uploader()
        
        logger.info("🗺️ MapGenerator initialized")
        logger.info(f"📊 Depth estimator: {'✅' if depth_estimator else '❌'}")
        logger.info(f"📐 Edge detector: {'✅' if edge_detector else '❌'}")
        logger.info(f"☁️ R2 uploader: {'✅' if self.r2_uploader else '❌'}")
    
    def _init_r2_uploader(self):
        """Initialize R2 uploader if credentials available"""
        try:
            from services.r2_uploader import create_r2_uploader
            self.r2_uploader = create_r2_uploader()
            logger.info("✅ R2 uploader initialized")
        except Exception as e:
            logger.warning(f"⚠️ R2 uploader not available: {e}")
            self.r2_uploader = None
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all map generation components"""
        return {
            "map_generator_available": True,
            "depth_estimator": {
                "available": self.depth_estimator is not None,
                "info": self.depth_estimator.get_model_info() if self.depth_estimator else None
            },
            "edge_detector": {
                "available": self.edge_detector is not None,
                "info": self.edge_detector.get_detector_info() if self.edge_detector else None
            },
            "r2_uploader": {
                "available": self.r2_uploader is not None,
                "bucket": self.config.r2_bucket
            },
            "config": {
                "max_concurrent_maps": self.config.max_concurrent_maps,
                "upload_to_r2": self.config.upload_to_r2,
                "keep_local_copies": self.config.keep_local_copies
            }
        }
    
    async def generate_all_maps(
        self,
        image_path: str,
        scene_id: str,
        map_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate all requested map types for a scene
        
        Args:
            image_path: Path or URL to input image
            scene_id: Unique scene identifier
            map_types: List of map types to generate ["depth", "edge"]
            
        Returns:
            Dictionary with generation results and R2 keys
        """
        if map_types is None:
            map_types = ["depth", "edge"]
        
        logger.info(f"🗺️ Generating maps for scene {scene_id}: {map_types}")
        
        # Get performance recommendations
        if PERFORMANCE_MONITORING:
            monitor = get_performance_monitor()
            recommendations = monitor.get_recommendations()
            if recommendations.get("warnings"):
                logger.warning(f"⚠️ Performance warning: {recommendations['warnings']}")
            
            # Log estimated times
            estimates = recommendations.get("estimated_times", {})
            if estimates:
                total_estimate = sum(estimates.get(t.replace('_map', '_estimation'), 10) for t in map_types)
                logger.info(f"⏱️ Estimated processing time: {total_estimate:.1f}s")
        
        results = {
            "scene_id": scene_id,
            "maps_generated": {},
            "r2_keys": {},
            "local_paths": {},
            "errors": {},
            "generation_time": datetime.utcnow().isoformat(),
            "success": False
        }
        
        try:
            # Prepare output directory
            output_dir = self.config.output_dir or tempfile.mkdtemp(prefix=f"maps_{scene_id}_")
            
            # Generate maps sequentially to manage memory
            for map_type in map_types:
                try:
                    logger.info(f"📊 Generating {map_type} map for scene {scene_id}")
                    
                    # Track performance for this operation
                    operation_name = f"{map_type}_generation"
                    if PERFORMANCE_MONITORING:
                        monitor = get_performance_monitor()
                        with monitor.track_operation(operation_name):
                            if map_type == "depth" and self.depth_estimator:
                                result = await self._generate_depth_map(image_path, scene_id, output_dir)
                            elif map_type == "edge" and self.edge_detector:
                                result = await self._generate_edge_map(image_path, scene_id, output_dir)
                            else:
                                logger.warning(f"⚠️ Skipping {map_type} map - estimator not available")
                                continue
                    else:
                        # No performance monitoring
                        if map_type == "depth" and self.depth_estimator:
                            result = await self._generate_depth_map(image_path, scene_id, output_dir)
                        elif map_type == "edge" and self.edge_detector:
                            result = await self._generate_edge_map(image_path, scene_id, output_dir)
                        else:
                            logger.warning(f"⚠️ Skipping {map_type} map - estimator not available")
                            continue
                    
                    if result:
                        results["maps_generated"][map_type] = True
                        results["local_paths"][map_type] = result["local_path"]
                        
                        # Upload to R2 if enabled
                        if self.config.upload_to_r2 and self.r2_uploader:
                            r2_key = await self._upload_map_to_r2(
                                result["local_path"],
                                map_type,
                                scene_id
                            )
                            if r2_key:
                                results["r2_keys"][map_type] = r2_key
                                logger.info(f"☁️ Uploaded {map_type} map to R2: {r2_key}")
                        
                        # Cleanup local file if not keeping copies
                        if not self.config.keep_local_copies:
                            try:
                                os.unlink(result["local_path"])
                                logger.debug(f"🧹 Cleaned up local {map_type} map")
                            except OSError:
                                pass
                    else:
                        results["errors"][map_type] = f"Failed to generate {map_type} map"
                        logger.error(f"❌ Failed to generate {map_type} map")
                    
                    # Clear memory between operations (GPU/CPU)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    else:
                        import gc
                        gc.collect()  # CPU memory cleanup
                
                except Exception as e:
                    logger.error(f"❌ Error generating {map_type} map: {e}")
                    results["errors"][map_type] = str(e)
            
            # Check overall success
            results["success"] = len(results["maps_generated"]) > 0
            
            # Cleanup output directory if temporary
            if not self.config.keep_local_copies and output_dir.startswith(tempfile.gettempdir()):
                try:
                    os.rmdir(output_dir)
                except OSError:
                    pass
            
            logger.info(f"✅ Map generation completed for scene {scene_id}: {len(results['maps_generated'])} maps")
            return results
            
        except Exception as e:
            logger.error(f"❌ Map generation failed for scene {scene_id}: {e}")
            results["errors"]["general"] = str(e)
            return results
    
    async def _generate_depth_map(self, image_path: str, scene_id: str, output_dir: str) -> Optional[Dict[str, Any]]:
        """Generate depth map"""
        try:
            depth_map_path = await self.depth_estimator.estimate_depth(image_path, output_dir)
            
            if depth_map_path:
                return {
                    "local_path": depth_map_path,
                    "map_type": "depth",
                    "scene_id": scene_id
                }
            return None
            
        except Exception as e:
            logger.error(f"❌ Depth map generation failed: {e}")
            return None
    
    async def _generate_edge_map(self, image_path: str, scene_id: str, output_dir: str) -> Optional[Dict[str, Any]]:
        """Generate edge map"""
        try:
            edge_map_path = await self.edge_detector.detect_edges(image_path, output_dir)
            
            if edge_map_path:
                return {
                    "local_path": edge_map_path,
                    "map_type": "edge",
                    "scene_id": scene_id
                }
            return None
            
        except Exception as e:
            logger.error(f"❌ Edge map generation failed: {e}")
            return None
    
    async def _upload_map_to_r2(self, local_path: str, map_type: str, scene_id: str) -> Optional[str]:
        """Upload generated map to R2 storage"""
        try:
            if not self.r2_uploader:
                logger.warning("⚠️ R2 uploader not available")
                return None
            
            # Construct R2 key following the pattern: training-data/maps/{type}/{scene_id}_{type}.png
            filename = f"{scene_id}_{map_type}.png"
            r2_key = f"{self.config.r2_bucket}/maps/{map_type}/{filename}"
            
            # Upload file
            success = await self.r2_uploader.upload_file(local_path, r2_key)
            
            if success:
                logger.debug(f"☁️ Successfully uploaded {map_type} map to R2: {r2_key}")
                return r2_key
            else:
                logger.error(f"❌ Failed to upload {map_type} map to R2")
                return None
                
        except Exception as e:
            logger.error(f"❌ R2 upload failed for {map_type} map: {e}")
            return None
    
    async def generate_single_map(
        self,
        image_path: str,
        scene_id: str,
        map_type: str
    ) -> Optional[Dict[str, Any]]:
        """Generate a single map type"""
        return await self.generate_all_maps(image_path, scene_id, [map_type])
    
    def cleanup(self):
        """Clean up resources"""
        if self.config.enable_cleanup:
            try:
                if self.depth_estimator and hasattr(self.depth_estimator, 'cleanup'):
                    self.depth_estimator.cleanup()
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                else:
                    import gc
                    gc.collect()
                
                logger.debug("🧹 MapGenerator cleanup completed")
            except Exception as e:
                logger.warning(f"Cleanup warning: {e}")


class MockR2Uploader:
    """Mock R2 uploader for development/testing"""
    
    async def upload_file(self, local_path: str, r2_key: str) -> bool:
        """Mock upload - just log the operation"""
        logger.info(f"🚀 [MOCK] Would upload {local_path} to R2 key: {r2_key}")
        return True


# Async convenience function
async def generate_scene_maps_simple(
    image_path: str,
    scene_id: str,
    depth_estimator=None,
    edge_detector=None,
    map_types: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Simple async function for map generation
    
    Args:
        image_path: Path to input image
        scene_id: Scene identifier
        depth_estimator: Depth estimation model
        edge_detector: Edge detection model
        map_types: Types of maps to generate
    
    Returns:
        Map generation results
    """
    generator = MapGenerator(depth_estimator, edge_detector)
    
    try:
        return await generator.generate_all_maps(image_path, scene_id, map_types)
    finally:
        generator.cleanup()


if __name__ == "__main__":
    # Test the map generator
    import sys
    
    async def test_map_generation():
        if len(sys.argv) != 3:
            print("Usage: python map_generator.py <image_path> <scene_id>")
            return
        
        image_path = sys.argv[1]
        scene_id = sys.argv[2]
        
        print(f"🗺️ Testing map generation for scene {scene_id}")
        print(f"📷 Image: {image_path}")
        
        # Create mock estimators for testing
        result = await generate_scene_maps_simple(image_path, scene_id)
        
        if result["success"]:
            print(f"✅ Map generation successful: {result['maps_generated']}")
        else:
            print(f"❌ Map generation failed: {result['errors']}")
    
    asyncio.run(test_map_generation())