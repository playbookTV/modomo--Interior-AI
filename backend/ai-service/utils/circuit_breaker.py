"""
Circuit Breaker implementation for AI model fault tolerance
"""
import asyncio
import time
from typing import Dict, Any, Callable, Optional, Awaitable
from enum import Enum
from dataclasses import dataclass
import structlog

logger = structlog.get_logger()

class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit breaker is open, calls fail fast
    HALF_OPEN = "half_open"  # Testing if service has recovered

@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""
    failure_threshold: int = 5          # Number of failures before opening
    success_threshold: int = 3          # Number of successes to close from half-open
    timeout: float = 60.0              # Timeout before trying half-open (seconds)
    expected_exception: type = Exception  # Exception type that counts as failure

class CircuitBreaker:
    """
    Circuit breaker for protecting against cascading failures
    """
    
    def __init__(self, name: str, config: CircuitBreakerConfig = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.call_count = 0
        self.success_total = 0
        self.failure_total = 0
        
    def _should_attempt_reset(self) -> bool:
        """Check if circuit should try to reset from open to half-open"""
        return (
            self.state == CircuitState.OPEN and
            time.time() - self.last_failure_time >= self.config.timeout
        )
        
    def _on_success(self):
        """Handle successful call"""
        self.call_count += 1
        self.success_total += 1
        
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self._close_circuit()
        elif self.state == CircuitState.CLOSED:
            self.failure_count = 0
            
    def _on_failure(self, exception: Exception):
        """Handle failed call"""
        self.call_count += 1
        self.failure_total += 1
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == CircuitState.HALF_OPEN:
            self._open_circuit()
        elif (self.state == CircuitState.CLOSED and 
              self.failure_count >= self.config.failure_threshold):
            self._open_circuit()
            
    def _open_circuit(self):
        """Open the circuit breaker"""
        if self.state != CircuitState.OPEN:
            logger.warning(f"Circuit breaker '{self.name}' opened after {self.failure_count} failures")
            self.state = CircuitState.OPEN
            self.success_count = 0
            
    def _close_circuit(self):
        """Close the circuit breaker"""
        if self.state != CircuitState.CLOSED:
            logger.info(f"Circuit breaker '{self.name}' closed after {self.success_count} successes")
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            
    def _half_open_circuit(self):
        """Set circuit to half-open state"""
        logger.info(f"Circuit breaker '{self.name}' half-open, testing service")
        self.state = CircuitState.HALF_OPEN
        self.success_count = 0
        
    async def call(self, func: Callable[..., Awaitable[Any]], *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection
        
        Args:
            func: Async function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerOpenError: When circuit is open
            Original exception: When function fails
        """
        # Check if we should attempt reset
        if self._should_attempt_reset():
            self._half_open_circuit()
            
        # Fail fast if circuit is open
        if self.state == CircuitState.OPEN:
            raise CircuitBreakerOpenError(
                f"Circuit breaker '{self.name}' is open. "
                f"Last failure: {time.time() - self.last_failure_time:.1f}s ago"
            )
            
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
            
        except self.config.expected_exception as e:
            self._on_failure(e)
            raise
        except Exception as e:
            # Unexpected exceptions don't count as circuit breaker failures
            logger.error(f"Unexpected error in circuit breaker '{self.name}': {e}")
            raise
            
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics"""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "call_count": self.call_count,
            "success_total": self.success_total,
            "failure_total": self.failure_total,
            "success_rate": self.success_total / max(self.call_count, 1),
            "last_failure_time": self.last_failure_time,
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "success_threshold": self.config.success_threshold,
                "timeout": self.config.timeout
            }
        }
        
    def reset(self):
        """Reset circuit breaker to closed state"""
        logger.info(f"Circuit breaker '{self.name}' manually reset")
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0

class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open"""
    pass

class CircuitBreakerManager:
    """
    Manager for multiple circuit breakers
    """
    
    def __init__(self):
        self.breakers: Dict[str, CircuitBreaker] = {}
        
    def get_breaker(self, name: str, config: CircuitBreakerConfig = None) -> CircuitBreaker:
        """Get or create a circuit breaker"""
        if name not in self.breakers:
            self.breakers[name] = CircuitBreaker(name, config)
        return self.breakers[name]
        
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get stats for all circuit breakers"""
        return {name: breaker.get_stats() for name, breaker in self.breakers.items()}
        
    def reset_all(self):
        """Reset all circuit breakers"""
        for breaker in self.breakers.values():
            breaker.reset()
            
    def reset_breaker(self, name: str):
        """Reset specific circuit breaker"""
        if name in self.breakers:
            self.breakers[name].reset()

# Global circuit breaker manager
circuit_breaker_manager = CircuitBreakerManager()