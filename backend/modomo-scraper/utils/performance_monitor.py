"""
Performance Monitor for CPU/GPU environments
------------------------------------------
- Monitor system resources during map generation
- Provide performance warnings and recommendations
- Estimate processing times for different environments
- Help users optimize their setup

Usage:
  monitor = PerformanceMonitor()
  with monitor.track_operation("depth_estimation"):
      # Your operation here
  
  recommendations = monitor.get_recommendations()
"""

import time
import psutil
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from contextlib import contextmanager

import torch

logger = logging.getLogger(__name__)

@dataclass
class SystemSpecs:
    """System specifications for performance assessment"""
    cpu_count: int
    cpu_freq: float
    memory_total: float  # GB
    memory_available: float  # GB
    gpu_available: bool
    gpu_memory: Optional[float] = None  # GB
    gpu_name: Optional[str] = None


@dataclass
class PerformanceMetrics:
    """Performance metrics for an operation"""
    operation_name: str
    duration: float  # seconds
    cpu_usage_avg: float  # percentage
    memory_usage_peak: float  # GB
    gpu_usage_avg: Optional[float] = None  # percentage
    gpu_memory_peak: Optional[float] = None  # GB


class PerformanceMonitor:
    """Monitor system performance during AI operations"""
    
    def __init__(self):
        self.system_specs = self._get_system_specs()
        self.metrics_history: List[PerformanceMetrics] = []
        
        logger.info(f"🖥️ System specs: {self._format_system_specs()}")
        
        # Performance expectations (rough estimates)
        self.expected_times = {
            "cpu": {
                "depth_estimation": 30,  # seconds for average indoor scene
                "edge_detection": 2,     # seconds for average scene
                "full_pipeline": 120     # seconds including detection + segmentation
            },
            "gpu": {
                "depth_estimation": 5,   # seconds for average indoor scene  
                "edge_detection": 1,     # seconds for average scene
                "full_pipeline": 20      # seconds including detection + segmentation
            }
        }
    
    def _get_system_specs(self) -> SystemSpecs:
        """Get current system specifications"""
        try:
            # CPU info
            cpu_count = psutil.cpu_count(logical=False) or psutil.cpu_count()
            cpu_freq = psutil.cpu_freq().current if psutil.cpu_freq() else 0.0
            
            # Memory info
            memory = psutil.virtual_memory()
            memory_total = memory.total / (1024**3)  # Convert to GB
            memory_available = memory.available / (1024**3)
            
            # GPU info
            gpu_available = torch.cuda.is_available()
            gpu_memory = None
            gpu_name = None
            
            if gpu_available:
                try:
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    gpu_name = torch.cuda.get_device_name(0)
                except:
                    pass
            
            return SystemSpecs(
                cpu_count=cpu_count,
                cpu_freq=cpu_freq,
                memory_total=memory_total,
                memory_available=memory_available,
                gpu_available=gpu_available,
                gpu_memory=gpu_memory,
                gpu_name=gpu_name
            )
        except Exception as e:
            logger.warning(f"Failed to get system specs: {e}")
            return SystemSpecs(
                cpu_count=1, cpu_freq=0.0, memory_total=0.0, 
                memory_available=0.0, gpu_available=False
            )
    
    def _format_system_specs(self) -> str:
        """Format system specs for logging"""
        specs = self.system_specs
        result = f"CPU: {specs.cpu_count} cores @ {specs.cpu_freq:.1f}MHz, RAM: {specs.memory_total:.1f}GB"
        
        if specs.gpu_available:
            result += f", GPU: {specs.gpu_name} ({specs.gpu_memory:.1f}GB)"
        else:
            result += ", GPU: None (CPU-only mode)"
        
        return result
    
    @contextmanager
    def track_operation(self, operation_name: str):
        """Context manager to track operation performance"""
        logger.info(f"⏱️ Starting performance tracking: {operation_name}")
        
        # Initial measurements
        start_time = time.time()
        start_memory = psutil.virtual_memory().used / (1024**3)
        
        cpu_usage_samples = []
        memory_usage_samples = []
        gpu_usage_samples = []
        gpu_memory_samples = []
        
        try:
            # Start monitoring in background
            import threading
            monitoring = True
            
            def monitor():
                while monitoring:
                    try:
                        cpu_usage_samples.append(psutil.cpu_percent())
                        memory_usage_samples.append(psutil.virtual_memory().used / (1024**3))
                        
                        if self.system_specs.gpu_available:
                            try:
                                # This requires nvidia-ml-py if available
                                import nvidia_ml_py as nvml
                                nvml.nvmlInit()
                                handle = nvml.nvmlDeviceGetHandleByIndex(0)
                                gpu_util = nvml.nvmlDeviceGetUtilizationRates(handle)
                                gpu_mem = nvml.nvmlDeviceGetMemoryInfo(handle)
                                
                                gpu_usage_samples.append(gpu_util.gpu)
                                gpu_memory_samples.append(gpu_mem.used / (1024**3))
                            except:
                                pass  # GPU monitoring not available
                        
                        time.sleep(0.5)  # Sample every 500ms
                    except:
                        break
            
            monitor_thread = threading.Thread(target=monitor, daemon=True)
            monitor_thread.start()
            
            yield
            
        finally:
            # Stop monitoring
            monitoring = False
            end_time = time.time()
            
            # Calculate metrics
            duration = end_time - start_time
            cpu_usage_avg = sum(cpu_usage_samples) / len(cpu_usage_samples) if cpu_usage_samples else 0
            memory_usage_peak = max(memory_usage_samples) if memory_usage_samples else start_memory
            gpu_usage_avg = sum(gpu_usage_samples) / len(gpu_usage_samples) if gpu_usage_samples else None
            gpu_memory_peak = max(gpu_memory_samples) if gpu_memory_samples else None
            
            # Store metrics
            metrics = PerformanceMetrics(
                operation_name=operation_name,
                duration=duration,
                cpu_usage_avg=cpu_usage_avg,
                memory_usage_peak=memory_usage_peak,
                gpu_usage_avg=gpu_usage_avg,
                gpu_memory_peak=gpu_memory_peak
            )
            
            self.metrics_history.append(metrics)
            
            # Log results
            logger.info(f"✅ {operation_name} completed in {duration:.1f}s")
            logger.info(f"📊 CPU: {cpu_usage_avg:.1f}%, RAM: {memory_usage_peak:.1f}GB")
            if gpu_usage_avg is not None:
                logger.info(f"🎮 GPU: {gpu_usage_avg:.1f}%, VRAM: {gpu_memory_peak:.1f}GB")
    
    def get_performance_warning(self) -> List[str]:
        """Get performance warning based on system specs"""
        specs = self.system_specs
        warnings = []
        
        # Memory warnings
        if specs.memory_total < 8:
            warnings.append("⚠️ Low RAM (<8GB) may cause slow processing")
        
        # CPU warnings
        if specs.cpu_count < 4:
            warnings.append("⚠️ Limited CPU cores may result in slower processing")
        
        # GPU warnings
        if not specs.gpu_available:
            warnings.append("🔄 CPU-only mode: Expect 3-6x longer processing times")
        elif specs.gpu_memory and specs.gpu_memory < 6:
            warnings.append("⚠️ Limited GPU memory may require sequential processing")
        
        return warnings
    
    def get_time_estimates(self, operations: List[str]) -> Dict[str, float]:
        """Get estimated processing times for operations"""
        device_type = "gpu" if self.system_specs.gpu_available else "cpu"
        estimates = {}
        
        for operation in operations:
            base_time = self.expected_times[device_type].get(operation, 10)
            
            # Adjust based on system specs
            if device_type == "cpu":
                # Adjust for CPU specs
                cpu_factor = max(0.5, min(2.0, 4.0 / self.system_specs.cpu_count))
                memory_factor = max(0.8, min(1.5, 8.0 / self.system_specs.memory_total))
                base_time *= cpu_factor * memory_factor
            
            estimates[operation] = base_time
        
        return estimates
    
    def get_recommendations(self) -> Dict[str, Any]:
        """Get performance recommendations"""
        specs = self.system_specs
        recommendations = {
            "system_status": "good",
            "warnings": self.get_performance_warning(),
            "suggestions": []
        }
        
        if not specs.gpu_available:
            recommendations["system_status"] = "cpu_only"
            recommendations["suggestions"].extend([
                "Consider using a GPU-enabled environment for faster processing",
                "Process scenes in smaller batches to manage memory usage",
                "Use edge detection only for faster results (skip depth maps)"
            ])
        
        if specs.memory_total < 8:
            recommendations["system_status"] = "limited_memory"
            recommendations["suggestions"].append(
                "Close other applications to free up memory during processing"
            )
        
        if specs.gpu_available and specs.gpu_memory and specs.gpu_memory < 6:
            recommendations["suggestions"].append(
                "GPU memory is limited - models will process sequentially"
            )
        
        # Add performance expectations
        estimates = self.get_time_estimates(["depth_estimation", "edge_detection"])
        recommendations["estimated_times"] = estimates
        
        return recommendations
    
    def get_latest_metrics(self) -> Optional[PerformanceMetrics]:
        """Get metrics from the most recent operation"""
        return self.metrics_history[-1] if self.metrics_history else None
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all recorded metrics"""
        if not self.metrics_history:
            return {"message": "No performance data available"}
        
        total_operations = len(self.metrics_history)
        total_time = sum(m.duration for m in self.metrics_history)
        avg_cpu = sum(m.cpu_usage_avg for m in self.metrics_history) / total_operations
        
        return {
            "total_operations": total_operations,
            "total_processing_time": total_time,
            "average_cpu_usage": avg_cpu,
            "operations": [
                {
                    "name": m.operation_name,
                    "duration": m.duration,
                    "cpu_usage": m.cpu_usage_avg
                }
                for m in self.metrics_history
            ]
        }


# Global performance monitor instance
_performance_monitor = None

def get_performance_monitor() -> PerformanceMonitor:
    """Get or create global performance monitor"""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor


if __name__ == "__main__":
    # Test the performance monitor
    monitor = PerformanceMonitor()
    
    print("System Specifications:")
    print(f"  {monitor._format_system_specs()}")
    
    print("\nPerformance Recommendations:")
    recommendations = monitor.get_recommendations()
    for key, value in recommendations.items():
        print(f"  {key}: {value}")
    
    # Test operation tracking
    print("\nTesting operation tracking...")
    with monitor.track_operation("test_operation"):
        time.sleep(2)  # Simulate work
    
    latest = monitor.get_latest_metrics()
    if latest:
        print(f"Latest operation took {latest.duration:.1f}s")