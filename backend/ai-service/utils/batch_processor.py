"""
Batch processing service for handling multiple images efficiently
"""
import asyncio
import time
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass
from PIL import Image
import structlog
from concurrent.futures import ThreadPoolExecutor
import uuid
from datetime import datetime

logger = structlog.get_logger()

@dataclass
class BatchJob:
    """Represents a single job in a batch"""
    job_id: str
    image: Image.Image
    analysis_type: str
    parameters: Dict[str, Any]
    created_at: float
    priority: int = 0  # Higher number = higher priority

@dataclass
class BatchResult:
    """Result of a batch job"""
    job_id: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    processing_time: float = 0.0
    cached: bool = False

class BatchProcessor:
    """
    Enhanced batch processor for efficient handling of multiple AI analysis requests
    """
    
    def __init__(self, 
                 max_batch_size: int = 8,
                 batch_timeout: float = 2.0,
                 max_concurrent_batches: int = 2,
                 enable_gpu_batching: bool = True,
                 adaptive_batching: bool = True,
                 performance_monitoring: bool = True):
        """
        Initialize enhanced batch processor
        
        Args:
            max_batch_size: Maximum number of images to process in a single batch
            batch_timeout: Maximum time to wait before processing partial batch (seconds)
            max_concurrent_batches: Maximum number of concurrent batch processes
            enable_gpu_batching: Whether to use GPU batching optimizations
            adaptive_batching: Whether to adapt batch sizes based on performance
            performance_monitoring: Whether to track detailed performance metrics
        """
        self.max_batch_size = max_batch_size
        self.batch_timeout = batch_timeout
        self.max_concurrent_batches = max_concurrent_batches
        self.enable_gpu_batching = enable_gpu_batching
        self.adaptive_batching = adaptive_batching
        self.performance_monitoring = performance_monitoring
        
        # Job queues by analysis type
        self.job_queues: Dict[str, List[BatchJob]] = {
            'object_detection': [],
            'style_transfer': [],
            'product_recognition': []
        }
        
        # Job results storage
        self.results: Dict[str, BatchResult] = {}
        
        # Processing locks and semaphores
        self.queue_locks: Dict[str, asyncio.Lock] = {
            analysis_type: asyncio.Lock() 
            for analysis_type in self.job_queues.keys()
        }
        self.processing_semaphore = asyncio.Semaphore(max_concurrent_batches)
        
        # Background processing tasks
        self.processing_tasks: Dict[str, Optional[asyncio.Task]] = {
            analysis_type: None 
            for analysis_type in self.job_queues.keys()
        }
        
        self._shutdown = False
        
        # Performance monitoring
        if self.performance_monitoring:
            self.batch_metrics = {
                analysis_type: {
                    'total_batches': 0,
                    'total_jobs': 0,
                    'total_processing_time': 0.0,
                    'successful_jobs': 0,
                    'failed_jobs': 0,
                    'average_batch_size': 0.0,
                    'average_processing_time': 0.0
                }
                for analysis_type in self.job_queues.keys()
            }
        
        # Adaptive batching parameters
        if self.adaptive_batching:
            self.optimal_batch_sizes = {
                analysis_type: max_batch_size 
                for analysis_type in self.job_queues.keys()
            }
            self.batch_performance_history = {
                analysis_type: []
                for analysis_type in self.job_queues.keys()
            }
        
        # Start background processors
        self.start_processors()
        
    def _update_performance_metrics(self, analysis_type: str, batch_size: int, 
                                   processing_time: float, successful_jobs: int, failed_jobs: int):
        """Update performance metrics for adaptive batching"""
        if not self.performance_monitoring:
            return
            
        metrics = self.batch_metrics[analysis_type]
        metrics['total_batches'] += 1
        metrics['total_jobs'] += batch_size
        metrics['total_processing_time'] += processing_time
        metrics['successful_jobs'] += successful_jobs
        metrics['failed_jobs'] += failed_jobs
        
        # Calculate averages
        metrics['average_batch_size'] = metrics['total_jobs'] / metrics['total_batches']
        metrics['average_processing_time'] = metrics['total_processing_time'] / metrics['total_batches']
        
        # Update adaptive batching
        if self.adaptive_batching:
            self._update_optimal_batch_size(analysis_type, batch_size, processing_time, 
                                          successful_jobs, failed_jobs)
    
    def _update_optimal_batch_size(self, analysis_type: str, batch_size: int, 
                                 processing_time: float, successful_jobs: int, failed_jobs: int):
        """Update optimal batch size based on performance"""
        history = self.batch_performance_history[analysis_type]
        
        # Calculate efficiency (successful jobs per second)
        efficiency = successful_jobs / processing_time if processing_time > 0 else 0
        
        history.append({
            'batch_size': batch_size,
            'efficiency': efficiency,
            'processing_time': processing_time,
            'success_rate': successful_jobs / batch_size if batch_size > 0 else 0
        })
        
        # Keep only recent history (last 10 batches)
        if len(history) > 10:
            history.pop(0)
        
        # Find optimal batch size based on efficiency
        if len(history) >= 3:
            best_performance = max(history, key=lambda x: x['efficiency'])
            self.optimal_batch_sizes[analysis_type] = min(
                max(best_performance['batch_size'], 2),  # Minimum batch size of 2
                self.max_batch_size  # Don't exceed max
            )
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        if not self.performance_monitoring:
            return {"performance_monitoring": False}
        
        metrics = {
            "performance_monitoring": True,
            "batch_metrics": self.batch_metrics.copy(),
            "queue_stats": self.get_queue_stats(),
            "configuration": {
                "max_batch_size": self.max_batch_size,
                "batch_timeout": self.batch_timeout,
                "max_concurrent_batches": self.max_concurrent_batches,
                "adaptive_batching": self.adaptive_batching,
                "enable_gpu_batching": self.enable_gpu_batching
            }
        }
        
        if self.adaptive_batching:
            metrics["adaptive_batching"] = {
                "optimal_batch_sizes": self.optimal_batch_sizes.copy(),
                "performance_history_length": {
                    analysis_type: len(history) 
                    for analysis_type, history in self.batch_performance_history.items()
                }
            }
        
        return metrics
    
    def reset_performance_metrics(self):
        """Reset all performance metrics"""
        if self.performance_monitoring:
            for analysis_type in self.job_queues.keys():
                self.batch_metrics[analysis_type] = {
                    'total_batches': 0,
                    'total_jobs': 0,
                    'total_processing_time': 0.0,
                    'successful_jobs': 0,
                    'failed_jobs': 0,
                    'average_batch_size': 0.0,
                    'average_processing_time': 0.0
                }
        
        if self.adaptive_batching:
            for analysis_type in self.job_queues.keys():
                self.optimal_batch_sizes[analysis_type] = self.max_batch_size
                self.batch_performance_history[analysis_type] = []
        
        logger.info("Batch processor performance metrics reset")
        
    def start_processors(self):
        """Start background batch processors for each analysis type"""
        for analysis_type in self.job_queues.keys():
            if self.processing_tasks[analysis_type] is None or self.processing_tasks[analysis_type].done():
                self.processing_tasks[analysis_type] = asyncio.create_task(
                    self._process_queue(analysis_type)
                )
                
    async def submit_job(self, 
                        image: Image.Image,
                        analysis_type: str,
                        parameters: Dict[str, Any],
                        priority: int = 0) -> str:
        """
        Submit a job for batch processing
        
        Args:
            image: PIL Image to process
            analysis_type: Type of analysis ('object_detection', 'style_transfer', 'product_recognition')
            parameters: Analysis parameters
            priority: Job priority (higher = processed first)
            
        Returns:
            Job ID for tracking
        """
        if analysis_type not in self.job_queues:
            raise ValueError(f"Unsupported analysis type: {analysis_type}")
            
        job_id = str(uuid.uuid4())
        job = BatchJob(
            job_id=job_id,
            image=image,
            analysis_type=analysis_type,
            parameters=parameters,
            created_at=time.time(),
            priority=priority
        )
        
        async with self.queue_locks[analysis_type]:
            self.job_queues[analysis_type].append(job)
            # Sort by priority (highest first) then by creation time (oldest first)
            self.job_queues[analysis_type].sort(
                key=lambda j: (-j.priority, j.created_at)
            )
            
        logger.debug(f"Submitted batch job {job_id} for {analysis_type}")
        return job_id
        
    async def get_result(self, job_id: str, timeout: float = 30.0) -> Optional[BatchResult]:
        """
        Get result for a specific job
        
        Args:
            job_id: Job ID to get result for
            timeout: Maximum time to wait for result
            
        Returns:
            BatchResult or None if not found/timeout
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if job_id in self.results:
                result = self.results[job_id]
                # Clean up old results after retrieving
                del self.results[job_id]
                return result
                
            await asyncio.sleep(0.1)  # Check every 100ms
            
        logger.warning(f"Timeout waiting for batch job result: {job_id}")
        return None
        
    async def get_batch_results(self, job_ids: List[str], timeout: float = 60.0) -> Dict[str, BatchResult]:
        """
        Get results for multiple jobs
        
        Args:
            job_ids: List of job IDs
            timeout: Maximum time to wait for all results
            
        Returns:
            Dictionary mapping job_id to BatchResult
        """
        results = {}
        start_time = time.time()
        
        remaining_jobs = set(job_ids)
        
        while remaining_jobs and (time.time() - start_time < timeout):
            for job_id in list(remaining_jobs):
                if job_id in self.results:
                    results[job_id] = self.results[job_id]
                    del self.results[job_id]
                    remaining_jobs.remove(job_id)
                    
            if remaining_jobs:
                await asyncio.sleep(0.1)
                
        # Log any jobs that timed out
        if remaining_jobs:
            logger.warning(f"Timeout waiting for batch jobs: {remaining_jobs}")
            
        return results
        
    async def _process_queue(self, analysis_type: str):
        """Background processor for a specific analysis type queue"""
        while not self._shutdown:
            try:
                await asyncio.sleep(0.1)  # Prevent tight loop
                
                # Check if we have jobs to process
                async with self.queue_locks[analysis_type]:
                    if not self.job_queues[analysis_type]:
                        continue
                        
                    # Get batch of jobs
                    batch = self._get_batch(analysis_type)
                    
                if batch:
                    # Process batch with semaphore to limit concurrency
                    async with self.processing_semaphore:
                        await self._process_batch(analysis_type, batch)
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in batch processor for {analysis_type}: {e}")
                await asyncio.sleep(1.0)  # Back off on error
                
    def _get_batch(self, analysis_type: str) -> List[BatchJob]:
        """Get a batch of jobs from the queue with adaptive sizing (called with lock held)"""
        queue = self.job_queues[analysis_type]
        
        if not queue:
            return []
        
        # Determine optimal batch size
        if self.adaptive_batching and analysis_type in self.optimal_batch_sizes:
            optimal_size = self.optimal_batch_sizes[analysis_type]
        else:
            optimal_size = self.max_batch_size
            
        # Use adaptive batch size but don't exceed queue length
        batch_size = min(optimal_size, len(queue), self.max_batch_size)
        
        # Check if we should wait for more jobs or process immediately
        oldest_job_age = time.time() - queue[0].created_at
        
        # Process immediately if:
        # - Queue is full (>= optimal size)
        # - Oldest job exceeds timeout
        # - High priority job waiting
        # - Queue size is at least 2 and we're using adaptive batching with good performance
        should_process_now = (
            len(queue) >= optimal_size or
            oldest_job_age >= self.batch_timeout or
            any(job.priority > 0 for job in queue[:batch_size]) or
            (self.adaptive_batching and len(queue) >= 2 and oldest_job_age >= self.batch_timeout * 0.5)
        )
        
        if should_process_now:
            batch = queue[:batch_size]
            self.job_queues[analysis_type] = queue[batch_size:]
            logger.debug(f"Created batch of {len(batch)} jobs for {analysis_type} (optimal: {optimal_size})")
            return batch
        else:
            return []
            
    async def _process_batch(self, analysis_type: str, batch: List[BatchJob]):
        """Process a batch of jobs with performance tracking"""
        logger.info(f"Processing batch of {len(batch)} {analysis_type} jobs")
        start_time = time.time()
        successful_jobs = 0
        failed_jobs = 0
        
        try:
            if analysis_type == 'object_detection':
                results = await self._process_object_detection_batch(batch)
            elif analysis_type == 'style_transfer':
                results = await self._process_style_transfer_batch(batch)
            elif analysis_type == 'product_recognition':
                results = await self._process_product_recognition_batch(batch)
            else:
                # Fallback to individual processing
                results = await self._process_individual_jobs(batch)
                
            # Store results and count successes/failures
            for job, result in zip(batch, results):
                self.results[job.job_id] = result
                if result.success:
                    successful_jobs += 1
                else:
                    failed_jobs += 1
                    
            processing_time = time.time() - start_time
            logger.info(f"Completed {analysis_type} batch in {processing_time:.2f}s "
                       f"({successful_jobs} success, {failed_jobs} failed)")
            
            # Update performance metrics
            self._update_performance_metrics(
                analysis_type, len(batch), processing_time, successful_jobs, failed_jobs
            )
            
        except Exception as e:
            logger.error(f"Batch processing failed for {analysis_type}: {e}")
            processing_time = time.time() - start_time
            failed_jobs = len(batch)
            
            # Store error results for all jobs in batch
            for job in batch:
                self.results[job.job_id] = BatchResult(
                    job_id=job.job_id,
                    success=False,
                    error=str(e),
                    processing_time=processing_time / len(batch)  # Distribute time across jobs
                )
            
            # Update performance metrics for failed batch
            self._update_performance_metrics(
                analysis_type, len(batch), processing_time, 0, failed_jobs
            )
                
    async def _process_object_detection_batch(self, batch: List[BatchJob]) -> List[BatchResult]:
        """Process batch of object detection jobs"""
        from ..main import cached_analysis_service
        
        results = []
        
        # If GPU batching is enabled and batch size > 1, use optimized batch processing
        if self.enable_gpu_batching and len(batch) > 1:
            # TODO: Implement GPU-optimized batch object detection
            # For now, fall back to individual processing
            results = await self._process_individual_jobs(batch)
        else:
            # Process individually with caching
            for job in batch:
                start_time = time.time()
                try:
                    # Import here to avoid circular imports
                    from ..main import real_object_detection
                    
                    result = await real_object_detection(job.image)
                    
                    results.append(BatchResult(
                        job_id=job.job_id,
                        success=True,
                        result=result,
                        processing_time=time.time() - start_time
                    ))
                    
                except Exception as e:
                    results.append(BatchResult(
                        job_id=job.job_id,
                        success=False,
                        error=str(e),
                        processing_time=time.time() - start_time
                    ))
                    
        return results
        
    async def _process_style_transfer_batch(self, batch: List[BatchJob]) -> List[BatchResult]:
        """Process batch of style transfer jobs"""
        results = []
        
        # Style transfer is memory intensive, process one at a time
        for job in batch:
            start_time = time.time()
            try:
                # Import here to avoid circular imports
                from ..main import real_style_transfer
                
                style = job.parameters.get('style', 'modern')
                detected_objects = job.parameters.get('detected_objects', [])
                
                result = await real_style_transfer(job.image, style, detected_objects)
                
                results.append(BatchResult(
                    job_id=job.job_id,
                    success=True,
                    result=result,
                    processing_time=time.time() - start_time
                ))
                
            except Exception as e:
                results.append(BatchResult(
                    job_id=job.job_id,
                    success=False,
                    error=str(e),
                    processing_time=time.time() - start_time
                ))
                
        return results
        
    async def _process_product_recognition_batch(self, batch: List[BatchJob]) -> List[BatchResult]:
        """Process batch of product recognition jobs"""
        results = []
        
        # Process individually with caching
        for job in batch:
            start_time = time.time()
            try:
                # Import here to avoid circular imports
                from ..main import real_product_recognition
                
                detected_objects = job.parameters.get('detected_objects', [])
                style = job.parameters.get('style', 'modern')
                
                result = await real_product_recognition(job.image, detected_objects, style)
                
                results.append(BatchResult(
                    job_id=job.job_id,
                    success=True,
                    result=result,
                    processing_time=time.time() - start_time
                ))
                
            except Exception as e:
                results.append(BatchResult(
                    job_id=job.job_id,
                    success=False,
                    error=str(e),
                    processing_time=time.time() - start_time
                ))
                
        return results
        
    async def _process_individual_jobs(self, batch: List[BatchJob]) -> List[BatchResult]:
        """Fallback individual processing"""
        results = []
        
        for job in batch:
            results.append(BatchResult(
                job_id=job.job_id,
                success=False,
                error="Batch processing not implemented for this analysis type",
                processing_time=0.0
            ))
            
        return results
        
    def get_queue_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics about processing queues"""
        stats = {}
        
        for analysis_type, queue in self.job_queues.items():
            stats[analysis_type] = {
                "queued_jobs": len(queue),
                "oldest_job_age": time.time() - queue[0].created_at if queue else 0,
                "high_priority_jobs": sum(1 for job in queue if job.priority > 0)
            }
            
        stats["pending_results"] = len(self.results)
        return stats
        
    async def shutdown(self):
        """Shutdown the batch processor"""
        logger.info("Shutting down batch processor")
        self._shutdown = True
        
        # Cancel processing tasks
        for task in self.processing_tasks.values():
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                    
        logger.info("Batch processor shutdown complete")

# Global batch processor instance
batch_processor = BatchProcessor()