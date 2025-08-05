"""
Redis caching service for AI analysis results
"""
import redis
import json
import hashlib
import pickle
import asyncio
import gzip
import time
from typing import Any, Optional, Dict, List, Union
from datetime import datetime, timedelta
import structlog
from PIL import Image
import io
import base64
import os
from contextlib import asynccontextmanager

logger = structlog.get_logger()

class RedisCache:
    """Redis-based caching for AI analysis results with performance optimizations"""
    
    def __init__(self, 
                 redis_url: str = None,
                 default_ttl: int = 3600,  # 1 hour default TTL
                 max_retries: int = 3,
                 connection_pool_size: int = 20,
                 enable_compression: bool = True,
                 compression_threshold: int = 1024):  # Compress data > 1KB
        """
        Initialize Redis cache with performance optimizations
        
        Args:
            redis_url: Redis connection URL (defaults to localhost)
            default_ttl: Default time-to-live in seconds
            max_retries: Maximum retry attempts for Redis operations
            connection_pool_size: Redis connection pool size
            enable_compression: Whether to compress large cache entries
            compression_threshold: Minimum size in bytes to trigger compression
        """
        self.redis_url = redis_url or os.getenv('REDIS_URL', 'redis://localhost:6379')
        self.default_ttl = default_ttl
        self.max_retries = max_retries
        self.connection_pool_size = connection_pool_size
        self.enable_compression = enable_compression
        self.compression_threshold = compression_threshold
        self.redis_client: Optional[redis.Redis] = None
        self._connected = False
        
        # Performance metrics
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_errors = 0
        
    async def connect(self):
        """Connect to Redis with connection pooling"""
        try:
            # Create connection pool for better performance
            pool = redis.ConnectionPool.from_url(
                self.redis_url,
                max_connections=self.connection_pool_size,
                decode_responses=False,  # Keep binary for pickle
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            self.redis_client = redis.Redis(connection_pool=pool)
            
            # Test connection
            await asyncio.get_event_loop().run_in_executor(
                None, self.redis_client.ping
            )
            
            self._connected = True
            logger.info(f"Redis cache connected successfully with pool size {self.connection_pool_size}")
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self._connected = False
            
    async def disconnect(self):
        """Disconnect from Redis"""
        if self.redis_client:
            await asyncio.get_event_loop().run_in_executor(
                None, self.redis_client.close
            )
            self._connected = False
            logger.info("Redis cache disconnected")
            
    def _generate_cache_key(self, image: Image.Image, analysis_type: str, **kwargs) -> str:
        """
        Generate a unique cache key for an image analysis
        
        Args:
            image: PIL Image
            analysis_type: Type of analysis (object_detection, style_transfer, etc.)
            **kwargs: Additional parameters that affect the analysis
            
        Returns:
            Unique cache key string
        """
        # Convert image to bytes for hashing
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='JPEG', quality=85)
        img_data = img_bytes.getvalue()
        
        # Create hash of image content
        img_hash = hashlib.sha256(img_data).hexdigest()[:16]
        
        # Create hash of parameters
        params_str = json.dumps(kwargs, sort_keys=True)
        params_hash = hashlib.sha256(params_str.encode()).hexdigest()[:8]
        
        return f"ai_cache:{analysis_type}:{img_hash}:{params_hash}"
        
    def _compress_data(self, data: bytes) -> bytes:
        """Compress data if it exceeds threshold"""
        if not self.enable_compression or len(data) < self.compression_threshold:
            return b'uncompressed:' + data
        
        try:
            compressed = gzip.compress(data)
            # Only use compression if it actually reduces size
            if len(compressed) < len(data) * 0.9:  # 10% reduction minimum
                return b'compressed:' + compressed
            else:
                return b'uncompressed:' + data
        except Exception as e:
            logger.warning(f"Compression failed: {e}")
            return b'uncompressed:' + data
    
    def _decompress_data(self, data: bytes) -> bytes:
        """Decompress data if it was compressed"""
        if data.startswith(b'compressed:'):
            try:
                return gzip.decompress(data[11:])  # Remove 'compressed:' prefix
            except Exception as e:
                logger.error(f"Decompression failed: {e}")
                raise
        elif data.startswith(b'uncompressed:'):
            return data[13:]  # Remove 'uncompressed:' prefix
        else:
            # Legacy data without prefix
            return data
    
    @asynccontextmanager
    async def _performance_timer(self, operation: str):
        """Context manager to track operation performance"""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            logger.debug(f"Redis {operation} took {duration:.4f}s")
        
    async def get(self, cache_key: str) -> Optional[Any]:
        """
        Get cached result with performance optimizations
        
        Args:
            cache_key: Cache key
            
        Returns:
            Cached result or None if not found
        """
        if not self._connected:
            self.cache_errors += 1
            return None
            
        try:
            async with self._performance_timer("get"):
                result = await asyncio.get_event_loop().run_in_executor(
                    None, self.redis_client.get, cache_key
                )
                
                if result:
                    # Decompress and deserialize the cached data
                    decompressed_data = self._decompress_data(result)
                    cached_data = pickle.loads(decompressed_data)
                    
                    self.cache_hits += 1
                    logger.debug(f"Cache hit: {cache_key}")
                    return cached_data
                else:
                    self.cache_misses += 1
                    logger.debug(f"Cache miss: {cache_key}")
                    return None
                    
        except Exception as e:
            self.cache_errors += 1
            logger.error(f"Redis get error: {e}")
            return None
            
    async def set(self, cache_key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set cached result with compression and performance tracking
        
        Args:
            cache_key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (uses default if None)
            
        Returns:
            True if successful, False otherwise
        """
        if not self._connected:
            self.cache_errors += 1
            return False
            
        try:
            async with self._performance_timer("set"):
                # Serialize the data
                serialized_value = pickle.dumps(value)
                
                # Apply compression if enabled
                compressed_value = self._compress_data(serialized_value)
                
                # Calculate compression ratio for logging
                compression_ratio = len(compressed_value) / len(serialized_value) if serialized_value else 1.0
                
                # Set with TTL
                ttl = ttl or self.default_ttl
                success = await asyncio.get_event_loop().run_in_executor(
                    None, self.redis_client.setex, cache_key, ttl, compressed_value
                )
                
                logger.debug(f"Cache set: {cache_key} (TTL: {ttl}s, compression: {compression_ratio:.2f})")
                return bool(success)
                
        except Exception as e:
            self.cache_errors += 1
            logger.error(f"Redis set error: {e}")
            return False
            
    async def delete(self, cache_key: str) -> bool:
        """Delete cached result"""
        if not self._connected:
            return False
            
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None, self.redis_client.delete, cache_key
            )
            logger.debug(f"Cache delete: {cache_key}")
            return bool(result)
            
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            return False
            
    async def clear_pattern(self, pattern: str) -> int:
        """
        Clear all keys matching a pattern
        
        Args:
            pattern: Redis key pattern (e.g., "ai_cache:object_detection:*")
            
        Returns:
            Number of keys deleted
        """
        if not self._connected:
            return 0
            
        try:
            keys = await asyncio.get_event_loop().run_in_executor(
                None, self.redis_client.keys, pattern
            )
            
            if keys:
                deleted = await asyncio.get_event_loop().run_in_executor(
                    None, self.redis_client.delete, *keys
                )
                logger.info(f"Cleared {deleted} keys matching pattern: {pattern}")
                return deleted
            else:
                return 0
                
        except Exception as e:
            logger.error(f"Redis clear pattern error: {e}")
            return 0
            
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        if not self._connected:
            return {"connected": False}
            
        try:
            info = await asyncio.get_event_loop().run_in_executor(
                None, self.redis_client.info, "memory"
            )
            
            # Count AI cache keys
            ai_keys = await asyncio.get_event_loop().run_in_executor(
                None, self.redis_client.keys, "ai_cache:*"
            )
            
            # Calculate hit rate
            total_requests = self.cache_hits + self.cache_misses
            hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0
            
            return {
                "connected": True,
                "memory_used": info.get('used_memory_human', 'Unknown'),
                "memory_used_bytes": info.get('used_memory', 0),
                "ai_cache_keys": len(ai_keys),
                "total_keys": await asyncio.get_event_loop().run_in_executor(
                    None, self.redis_client.dbsize
                ),
                "performance": {
                    "cache_hits": self.cache_hits,
                    "cache_misses": self.cache_misses,
                    "cache_errors": self.cache_errors,
                    "hit_rate_percent": round(hit_rate, 2),
                    "total_requests": total_requests
                },
                "configuration": {
                    "connection_pool_size": self.connection_pool_size,
                    "compression_enabled": self.enable_compression,
                    "compression_threshold": self.compression_threshold,
                    "default_ttl": self.default_ttl
                }
            }
            
        except Exception as e:
            logger.error(f"Redis stats error: {e}")
            return {"connected": False, "error": str(e)}
    
    def reset_performance_metrics(self):
        """Reset performance tracking metrics"""
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_errors = 0
        logger.info("Redis cache performance metrics reset")
    
    async def get_key_analysis(self, pattern: str = "ai_cache:*") -> Dict[str, Any]:
        """Analyze cache keys for optimization insights"""
        if not self._connected:
            return {"connected": False}
            
        try:
            keys = await asyncio.get_event_loop().run_in_executor(
                None, self.redis_client.keys, pattern
            )
            
            # Analyze key patterns
            analysis_types = {}
            total_size = 0
            
            for key in keys[:100]:  # Sample first 100 keys for performance
                key_str = key.decode('utf-8') if isinstance(key, bytes) else key
                parts = key_str.split(':')
                
                if len(parts) >= 2:
                    analysis_type = parts[1]
                    analysis_types[analysis_type] = analysis_types.get(analysis_type, 0) + 1
                
                # Get memory usage for this key
                try:
                    key_size = await asyncio.get_event_loop().run_in_executor(
                        None, self.redis_client.memory_usage, key
                    )
                    if key_size:
                        total_size += key_size
                except:
                    pass  # Skip if memory_usage not supported
            
            return {
                "total_keys": len(keys),
                "analysis_type_distribution": analysis_types,
                "sample_size_bytes": total_size,
                "average_key_size_bytes": total_size / min(len(keys), 100) if keys else 0
            }
            
        except Exception as e:
            logger.error(f"Key analysis error: {e}")
            return {"error": str(e)}

class CachedAnalysisService:
    """Service that wraps AI analyses with Redis caching"""
    
    def __init__(self, redis_cache: RedisCache):
        self.cache = redis_cache
        
    async def cached_object_detection(self, 
                                    image: Image.Image, 
                                    detector_func,
                                    confidence_threshold: float = 0.5) -> List[Dict]:
        """
        Cached object detection
        
        Args:
            image: PIL Image
            detector_func: Object detection function
            confidence_threshold: Detection confidence threshold
            
        Returns:
            List of detected objects
        """
        # Generate cache key
        cache_key = self.cache._generate_cache_key(
            image, 
            "object_detection",
            confidence_threshold=confidence_threshold
        )
        
        # Try to get from cache
        cached_result = await self.cache.get(cache_key)
        if cached_result is not None:
            logger.info("Using cached object detection result")
            return cached_result
            
        # Run analysis and cache result
        logger.info("Running object detection analysis")
        result = await detector_func(image, confidence_threshold)
        
        # Convert DetectedObject instances to dicts for caching
        serializable_result = [
            {
                "object_type": obj.object_type if hasattr(obj, 'object_type') else obj.class_name,
                "confidence": obj.confidence,
                "bounding_box": obj.bounding_box if hasattr(obj, 'bounding_box') else obj.bbox,
                "description": obj.description
            }
            for obj in result
        ]
        
        # Cache for 2 hours (object detection is relatively stable)
        await self.cache.set(cache_key, serializable_result, ttl=7200)
        
        return result
        
    async def cached_style_transfer(self,
                                  image: Image.Image,
                                  style_func,
                                  style: str,
                                  detected_objects: List[Dict],
                                  **kwargs) -> tuple:
        """
        Cached style transfer
        
        Args:
            image: PIL Image
            style_func: Style transfer function
            style: Style name
            detected_objects: Detected objects for context
            **kwargs: Additional style transfer parameters
            
        Returns:
            Tuple of (before_url, after_url)
        """
        # Generate cache key including style and objects
        cache_params = {
            "style": style,
            "num_objects": len(detected_objects),
            **kwargs
        }
        
        cache_key = self.cache._generate_cache_key(
            image,
            "style_transfer", 
            **cache_params
        )
        
        # Try to get from cache
        cached_result = await self.cache.get(cache_key)
        if cached_result is not None:
            logger.info("Using cached style transfer result")
            return cached_result
            
        # Run analysis and cache result
        logger.info("Running style transfer analysis")
        result = await style_func(image, style, detected_objects)
        
        # Cache for 4 hours (style transfer results can be reused longer)
        await self.cache.set(cache_key, result, ttl=14400)
        
        return result
        
    async def cached_product_recognition(self,
                                       image: Image.Image,
                                       product_func,
                                       detected_objects: List[Dict],
                                       style: str) -> List[Dict]:
        """
        Cached product recognition
        
        Args:
            image: PIL Image
            product_func: Product recognition function
            detected_objects: Detected objects
            style: Room style
            
        Returns:
            List of suggested products
        """
        # Generate cache key
        cache_params = {
            "style": style,
            "num_objects": len(detected_objects),
            "object_types": sorted([obj.get('object_type', obj.get('type', '')) for obj in detected_objects])
        }
        
        cache_key = self.cache._generate_cache_key(
            image,
            "product_recognition",
            **cache_params
        )
        
        # Try to get from cache
        cached_result = await self.cache.get(cache_key)
        if cached_result is not None:
            logger.info("Using cached product recognition result")
            return cached_result
            
        # Run analysis and cache result
        logger.info("Running product recognition analysis")
        result = await product_func(image, detected_objects, style)
        
        # Convert SuggestedProduct instances to dicts for caching
        serializable_result = []
        for product in result:
            if hasattr(product, 'dict'):
                serializable_result.append(product.dict())
            else:
                # Already a dict
                serializable_result.append(product)
        
        # Cache for 1 hour (product suggestions can change with inventory)
        await self.cache.set(cache_key, serializable_result, ttl=3600)
        
        return result

# Global cache instances
redis_cache = RedisCache()
cached_analysis_service = CachedAnalysisService(redis_cache)