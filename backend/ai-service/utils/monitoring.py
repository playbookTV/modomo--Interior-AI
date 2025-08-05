"""
Advanced monitoring and observability system for AI service
"""
import time
import psutil
import threading
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import structlog
import asyncio
from contextlib import asynccontextmanager
import json
from collections import defaultdict, deque
import uuid

logger = structlog.get_logger()

@dataclass
class MetricPoint:
    """Single metric data point"""
    timestamp: float
    value: float
    labels: Dict[str, str] = field(default_factory=dict)

@dataclass
class RequestMetrics:
    """Request-level metrics tracking"""
    request_id: str
    endpoint: str
    method: str
    start_time: float
    end_time: Optional[float] = None
    status_code: Optional[int] = None
    error: Optional[str] = None
    user_agent: Optional[str] = None
    processing_stages: Dict[str, float] = field(default_factory=dict)

class MetricsCollector:
    """Central metrics collection and storage system"""
    
    def __init__(self, max_points_per_metric: int = 1000):
        self.max_points_per_metric = max_points_per_metric
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_points_per_metric))
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = defaultdict(float)
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self.request_metrics: Dict[str, RequestMetrics] = {}
        self._lock = threading.Lock()
        
        # System metrics collection
        self._system_metrics_task = None
        self._collect_system_metrics = True
        
    def start_system_monitoring(self):
        """Start background system metrics collection"""
        if self._system_metrics_task is None:
            self._system_metrics_task = threading.Thread(
                target=self._collect_system_metrics_worker,
                daemon=True
            )
            self._system_metrics_task.start()
            logger.info("System metrics collection started")
    
    def stop_system_monitoring(self):
        """Stop background system metrics collection"""
        self._collect_system_metrics = False
        if self._system_metrics_task:
            self._system_metrics_task.join(timeout=5)
        logger.info("System metrics collection stopped")
    
    def _collect_system_metrics_worker(self):
        """Background worker for system metrics collection"""
        while self._collect_system_metrics:
            try:
                now = time.time()
                
                # CPU metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                self.record_gauge("system_cpu_percent", cpu_percent, {"component": "system"})
                
                # Memory metrics
                memory = psutil.virtual_memory()
                self.record_gauge("system_memory_percent", memory.percent, {"component": "system"})
                self.record_gauge("system_memory_available_mb", memory.available / 1024 / 1024, {"component": "system"})
                
                # Disk metrics
                disk = psutil.disk_usage('/')
                self.record_gauge("system_disk_percent", disk.percent, {"component": "system"})
                
                # GPU metrics (if available)
                try:
                    import GPUtil
                    gpus = GPUtil.getGPUs()
                    for i, gpu in enumerate(gpus):
                        labels = {"component": "gpu", "gpu_id": str(i)}
                        self.record_gauge("gpu_utilization_percent", gpu.load * 100, labels)
                        self.record_gauge("gpu_memory_percent", gpu.memoryUtil * 100, labels)
                        self.record_gauge("gpu_temperature_celsius", gpu.temperature, labels)
                except ImportError:
                    pass  # GPU monitoring not available
                
                time.sleep(30)  # Collect every 30 seconds
                
            except Exception as e:
                logger.warning(f"System metrics collection error: {e}")
                time.sleep(60)  # Back off on error
    
    def record_counter(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """Record a counter metric (always increasing)"""
        with self._lock:
            key = self._make_key(name, labels)
            self.counters[key] += value
            self.metrics[key].append(MetricPoint(
                timestamp=time.time(),
                value=self.counters[key],
                labels=labels or {}
            ))
    
    def record_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a gauge metric (can go up or down)"""
        with self._lock:
            key = self._make_key(name, labels)
            self.gauges[key] = value
            self.metrics[key].append(MetricPoint(
                timestamp=time.time(),
                value=value,
                labels=labels or {}
            ))
    
    def record_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a histogram metric (for timing/distribution data)"""
        with self._lock:
            key = self._make_key(name, labels)
            self.histograms[key].append(value)
            # Keep only recent values for memory efficiency
            if len(self.histograms[key]) > 1000:
                self.histograms[key] = self.histograms[key][-1000:]
    
    def _make_key(self, name: str, labels: Optional[Dict[str, str]]) -> str:
        """Create a unique key for the metric"""
        if not labels:
            return name
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all collected metrics"""
        with self._lock:
            return {
                "counters": dict(self.counters),
                "gauges": dict(self.gauges),
                "histogram_stats": {
                    key: {
                        "count": len(values),
                        "mean": sum(values) / len(values) if values else 0,
                        "min": min(values) if values else 0,
                        "max": max(values) if values else 0,
                        "p95": self._percentile(values, 95) if values else 0,
                        "p99": self._percentile(values, 99) if values else 0
                    }
                    for key, values in self.histograms.items()
                },
                "active_requests": len(self.request_metrics),
                "collection_timestamp": datetime.utcnow().isoformat()
            }
    
    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile of values"""
        if not values:
            return 0
        sorted_values = sorted(values)
        k = (len(sorted_values) - 1) * percentile / 100
        f = int(k)
        c = k - f
        if f == len(sorted_values) - 1:
            return sorted_values[f]
        return sorted_values[f] * (1 - c) + sorted_values[f + 1] * c
    
    def start_request_tracking(self, endpoint: str, method: str, **kwargs) -> str:
        """Start tracking a request"""
        request_id = str(uuid.uuid4())
        self.request_metrics[request_id] = RequestMetrics(
            request_id=request_id,
            endpoint=endpoint,
            method=method,
            start_time=time.time(),
            **kwargs
        )
        return request_id
    
    def end_request_tracking(self, request_id: str, status_code: int, error: Optional[str] = None):
        """End tracking a request"""
        if request_id in self.request_metrics:
            req = self.request_metrics[request_id]
            req.end_time = time.time()
            req.status_code = status_code
            req.error = error
            
            # Record metrics
            duration = req.end_time - req.start_time
            labels = {
                "endpoint": req.endpoint,
                "method": req.method,
                "status": str(status_code)
            }
            
            self.record_counter("http_requests_total", 1.0, labels)
            self.record_histogram("http_request_duration_seconds", duration, labels)
            
            if error:
                self.record_counter("http_errors_total", 1.0, labels)
            
            # Clean up completed request
            del self.request_metrics[request_id]
    
    def add_request_stage(self, request_id: str, stage: str, duration: float):
        """Add a processing stage timing to a request"""
        if request_id in self.request_metrics:
            self.request_metrics[request_id].processing_stages[stage] = duration
            
            # Record stage timing
            labels = {
                "endpoint": self.request_metrics[request_id].endpoint,
                "stage": stage
            }
            self.record_histogram("request_stage_duration_seconds", duration, labels)

class AlertManager:
    """Alerting system for monitoring critical conditions"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.alert_rules: List[Dict[str, Any]] = []
        self.active_alerts: Dict[str, Dict[str, Any]] = {}
        self.alert_history: deque = deque(maxlen=1000)
        
    def add_alert_rule(self, 
                      name: str,
                      condition: str,
                      threshold: float,
                      duration: int = 60,  # seconds
                      severity: str = "warning",
                      description: str = ""):
        """Add an alert rule"""
        rule = {
            "name": name,
            "condition": condition,
            "threshold": threshold,
            "duration": duration,
            "severity": severity,
            "description": description,
            "created_at": time.time()
        }
        self.alert_rules.append(rule)
        logger.info(f"Added alert rule: {name}")
    
    def check_alerts(self) -> List[Dict[str, Any]]:
        """Check all alert rules and return active alerts"""
        current_time = time.time()
        triggered_alerts = []
        
        for rule in self.alert_rules:
            try:
                # Evaluate the condition
                if self._evaluate_condition(rule):
                    alert_key = rule["name"]
                    
                    if alert_key not in self.active_alerts:
                        # New alert
                        alert = {
                            "name": rule["name"],
                            "severity": rule["severity"],
                            "description": rule["description"],
                            "threshold": rule["threshold"],
                            "triggered_at": current_time,
                            "status": "firing"
                        }
                        self.active_alerts[alert_key] = alert
                        self.alert_history.append(alert.copy())
                        triggered_alerts.append(alert)
                        logger.warning(f"ALERT TRIGGERED: {rule['name']} - {rule['description']}")
                    
                else:
                    # Check if we should resolve an active alert
                    if rule["name"] in self.active_alerts:
                        resolved_alert = self.active_alerts[rule["name"]].copy()
                        resolved_alert["status"] = "resolved"
                        resolved_alert["resolved_at"] = current_time
                        self.alert_history.append(resolved_alert)
                        del self.active_alerts[rule["name"]]
                        logger.info(f"ALERT RESOLVED: {rule['name']}")
                        
            except Exception as e:
                logger.error(f"Error evaluating alert rule {rule['name']}: {e}")
        
        return triggered_alerts
    
    def _evaluate_condition(self, rule: Dict[str, Any]) -> bool:
        """Evaluate an alert condition"""
        condition = rule["condition"]
        threshold = rule["threshold"]
        
        # Simple condition evaluation (can be extended)
        if condition == "high_cpu":
            cpu_metrics = [
                point.value for point in self.metrics_collector.metrics.get("system_cpu_percent{component=system}", [])
                if time.time() - point.timestamp < 300  # Last 5 minutes
            ]
            if cpu_metrics:
                return max(cpu_metrics) > threshold
        
        elif condition == "high_memory":
            memory_metrics = [
                point.value for point in self.metrics_collector.metrics.get("system_memory_percent{component=system}", [])
                if time.time() - point.timestamp < 300
            ]
            if memory_metrics:
                return max(memory_metrics) > threshold
        
        elif condition == "high_error_rate":
            # Calculate error rate from counters
            error_count = self.metrics_collector.counters.get("http_errors_total", 0)
            total_requests = self.metrics_collector.counters.get("http_requests_total", 1)
            error_rate = (error_count / total_requests) * 100
            return error_rate > threshold
        
        elif condition == "slow_response_time":
            # Check P95 response time
            response_times = self.metrics_collector.histograms.get("http_request_duration_seconds", [])
            if response_times:
                p95 = self.metrics_collector._percentile(response_times, 95)
                return p95 > threshold
        
        return False
    
    def get_alert_status(self) -> Dict[str, Any]:
        """Get current alert status"""
        return {
            "active_alerts": list(self.active_alerts.values()),
            "total_rules": len(self.alert_rules),
            "alert_history_count": len(self.alert_history),
            "last_check": datetime.utcnow().isoformat()
        }

class MonitoringService:
    """Main monitoring service that coordinates all monitoring components"""
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager(self.metrics_collector)
        self._monitoring_task = None
        self._running = False
        
        # Setup default alert rules
        self._setup_default_alerts()
    
    def _setup_default_alerts(self):
        """Setup default monitoring alert rules"""
        self.alert_manager.add_alert_rule(
            name="high_cpu_usage",
            condition="high_cpu",
            threshold=80.0,
            severity="warning",
            description="CPU usage is above 80%"
        )
        
        self.alert_manager.add_alert_rule(
            name="high_memory_usage",
            condition="high_memory",
            threshold=85.0,
            severity="warning",
            description="Memory usage is above 85%"
        )
        
        self.alert_manager.add_alert_rule(
            name="high_error_rate",
            condition="high_error_rate",
            threshold=5.0,
            severity="critical",
            description="HTTP error rate is above 5%"
        )
        
        self.alert_manager.add_alert_rule(
            name="slow_response_time",
            condition="slow_response_time",
            threshold=10.0,
            severity="warning",
            description="P95 response time is above 10 seconds"
        )
    
    async def start(self):
        """Start the monitoring service"""
        if not self._running:
            self._running = True
            self.metrics_collector.start_system_monitoring()
            
            # Start alert checking task
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            logger.info("Monitoring service started")
    
    async def stop(self):
        """Stop the monitoring service"""
        self._running = False
        self.metrics_collector.stop_system_monitoring()
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Monitoring service stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self._running:
            try:
                # Check alerts every 30 seconds
                self.alert_manager.check_alerts()
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(60)  # Back off on error
    
    @asynccontextmanager
    async def track_request(self, endpoint: str, method: str, **kwargs):
        """Context manager for tracking request metrics"""
        request_id = self.metrics_collector.start_request_tracking(endpoint, method, **kwargs)
        try:
            yield request_id
            self.metrics_collector.end_request_tracking(request_id, 200)
        except Exception as e:
            self.metrics_collector.end_request_tracking(request_id, 500, str(e))
            raise
    
    def record_ai_operation(self, operation: str, duration: float, success: bool, **labels):
        """Record AI operation metrics"""
        base_labels = {"operation": operation, **labels}
        
        self.metrics_collector.record_counter("ai_operations_total", 1.0, base_labels)
        self.metrics_collector.record_histogram("ai_operation_duration_seconds", duration, base_labels)
        
        if success:
            self.metrics_collector.record_counter("ai_operations_success_total", 1.0, base_labels)
        else:
            self.metrics_collector.record_counter("ai_operations_error_total", 1.0, base_labels)
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive monitoring status"""
        return {
            "service_status": "running" if self._running else "stopped",
            "metrics": self.metrics_collector.get_metrics_summary(),
            "alerts": self.alert_manager.get_alert_status(),
            "monitoring_info": {
                "system_monitoring": self.metrics_collector._collect_system_metrics,
                "alert_rules_count": len(self.alert_manager.alert_rules),
                "uptime_seconds": time.time() - (self._start_time if hasattr(self, '_start_time') else time.time())
            }
        }

# Global monitoring service instance
monitoring_service = MonitoringService()