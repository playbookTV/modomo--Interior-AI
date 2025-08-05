"""
Model Manager for efficient loading, caching, and memory management of AI models
"""
import asyncio
import gc
import time
from typing import Dict, Optional, Any, Set
from enum import Enum
import torch
import structlog
from dataclasses import dataclass
import threading
from concurrent.futures import ThreadPoolExecutor

logger = structlog.get_logger()

class ModelType(Enum):
    OBJECT_DETECTOR = "object_detector"
    STYLE_TRANSFER = "style_transfer" 
    PRODUCT_RECOGNIZER = "product_recognizer"

@dataclass
class ModelInfo:
    """Information about a cached model"""
    model_type: ModelType
    model: Any
    last_used: float
    load_time: float
    memory_usage: Optional[int] = None
    use_count: int = 0

class ModelManager:
    """
    Centralized model manager for loading, caching, and memory management
    """
    
    def __init__(self, max_cache_size: int = 3, max_idle_time: float = 300.0, preload_priority: bool = True):
        """
        Args:
            max_cache_size: Maximum number of models to keep in cache
            max_idle_time: Maximum idle time before unloading model (seconds)
            preload_priority: Whether to prioritize preloading frequently used models
        """
        self.max_cache_size = max_cache_size
        self.max_idle_time = max_idle_time
        self.preload_priority = preload_priority
        self.models: Dict[ModelType, ModelInfo] = {}
        self.loading_locks: Dict[ModelType, asyncio.Lock] = {
            model_type: asyncio.Lock() for model_type in ModelType
        }
        self.cleanup_task: Optional[asyncio.Task] = None
        self._shutdown = False
        
        # Model priority based on usage frequency (can be learned over time)
        self.model_priority = {
            ModelType.OBJECT_DETECTOR: 3,  # Highest priority - used in every request
            ModelType.PRODUCT_RECOGNIZER: 2,  # Medium priority
            ModelType.STYLE_TRANSFER: 1  # Lower priority - memory intensive
        }
        
        # Start cleanup task
        self.start_cleanup_task()
        
    def start_cleanup_task(self):
        """Start the background cleanup task"""
        if self.cleanup_task is None or self.cleanup_task.done():
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())
            
    async def _cleanup_loop(self):
        """Background task to cleanup unused models"""
        while not self._shutdown:
            try:
                await asyncio.sleep(60)  # Check every minute
                await self._cleanup_unused_models()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in model cleanup loop: {e}")
                
    async def _cleanup_unused_models(self):
        """Remove models that haven't been used recently"""
        current_time = time.time()
        models_to_remove = []
        
        for model_type, model_info in self.models.items():
            idle_time = current_time - model_info.last_used
            if idle_time > self.max_idle_time:
                models_to_remove.append(model_type)
                
        for model_type in models_to_remove:
            await self._unload_model(model_type)
            logger.info(f"Unloaded unused model: {model_type.value}")
            
    async def _unload_model(self, model_type: ModelType):
        """Unload a specific model from cache"""
        if model_type in self.models:
            model_info = self.models[model_type]
            
            # Call model's unload method if available
            if hasattr(model_info.model, 'unload_models'):
                model_info.model.unload_models()
            elif hasattr(model_info.model, 'unload_model'):
                model_info.model.unload_model()
                
            # Remove from cache
            del self.models[model_type]
            
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()
            
    async def get_model(self, model_type: ModelType) -> Any:
        """
        Get a model, loading it if necessary
        
        Args:
            model_type: Type of model to get
            
        Returns:
            The loaded model instance
        """
        async with self.loading_locks[model_type]:
            # Check if model is already loaded
            if model_type in self.models:
                model_info = self.models[model_type]
                model_info.last_used = time.time()
                model_info.use_count += 1
                logger.debug(f"Returning cached model: {model_type.value}")
                return model_info.model
                
            # Load model
            logger.info(f"Loading new model: {model_type.value}")
            start_time = time.time()
            
            try:
                model = await self._create_model(model_type)
                load_time = time.time() - start_time
                
                # Store in cache
                model_info = ModelInfo(
                    model_type=model_type,
                    model=model,
                    last_used=time.time(),
                    load_time=load_time,
                    use_count=1
                )
                
                self.models[model_type] = model_info
                
                # Check cache size and evict if necessary
                await self._enforce_cache_limit()
                
                logger.info(f"Model loaded successfully: {model_type.value} ({load_time:.2f}s)")
                return model
                
            except Exception as e:
                logger.error(f"Failed to load model {model_type.value}: {e}")
                raise
                
    async def _create_model(self, model_type: ModelType) -> Any:
        """Create and initialize a model instance"""
        if model_type == ModelType.OBJECT_DETECTOR:
            from .object_detector import ObjectDetector
            model = ObjectDetector()
            await model.load_model()
            return model
            
        elif model_type == ModelType.STYLE_TRANSFER:
            from .style_transfer import StyleTransferModel
            model = StyleTransferModel()
            await model.load_models()
            return model
            
        elif model_type == ModelType.PRODUCT_RECOGNIZER:
            from .product_recognizer import ProductRecognizer
            model = ProductRecognizer()
            await model.load_models()
            return model
            
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
    async def _enforce_cache_limit(self):
        """Ensure cache doesn't exceed maximum size"""
        while len(self.models) > self.max_cache_size:
            if self.preload_priority:
                # Evict based on priority and usage
                evict_model_type = min(
                    self.models.keys(),
                    key=lambda mt: (
                        self.model_priority.get(mt, 0),  # Lower priority first
                        -self.models[mt].last_used,      # Then LRU
                        -self.models[mt].use_count       # Then least used
                    )
                )
            else:
                # Find least recently used model
                evict_model_type = min(
                    self.models.keys(),
                    key=lambda mt: self.models[mt].last_used
                )
            
            await self._unload_model(evict_model_type)
            logger.info(f"Evicted model: {evict_model_type.value}")
            
    async def warm_up_models(self, model_types: Set[ModelType]):
        """
        Pre-load models to reduce cold start latency
        
        Args:
            model_types: Set of model types to warm up
        """
        logger.info(f"Warming up models: {[mt.value for mt in model_types]}")
        
        # Load models concurrently
        tasks = [self.get_model(model_type) for model_type in model_types]
        
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
            logger.info("Model warm-up completed")
        except Exception as e:
            logger.error(f"Error during model warm-up: {e}")
            
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        stats = {
            "cached_models": len(self.models),
            "max_cache_size": self.max_cache_size,
            "models": {}
        }
        
        for model_type, model_info in self.models.items():
            stats["models"][model_type.value] = {
                "last_used": model_info.last_used,
                "load_time": model_info.load_time,
                "use_count": model_info.use_count,
                "idle_time": time.time() - model_info.last_used
            }
            
        return stats
        
    async def clear_cache(self):
        """Clear all cached models"""
        logger.info("Clearing model cache")
        
        for model_type in list(self.models.keys()):
            await self._unload_model(model_type)
            
        logger.info("Model cache cleared")
        
    async def shutdown(self):
        """Shutdown the model manager"""
        logger.info("Shutting down model manager")
        self._shutdown = True
        
        # Cancel cleanup task
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
                
        # Clear all models
        await self.clear_cache()
        
        logger.info("Model manager shutdown complete")
        
    def __del__(self):
        """Cleanup on destruction"""
        if not self._shutdown:
            # Run cleanup in background if event loop is running
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.shutdown())
            except RuntimeError:
                # No event loop running, perform synchronous cleanup
                for model_type in list(self.models.keys()):
                    if hasattr(self.models[model_type].model, 'unload_models'):
                        self.models[model_type].model.unload_models()
                    elif hasattr(self.models[model_type].model, 'unload_model'):
                        self.models[model_type].model.unload_model()

# Global model manager instance
model_manager = ModelManager()