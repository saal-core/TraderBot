"""Circuit breaker pattern for external service protection.

Prevents cascading failures by temporarily blocking calls to failing services.

States:
- CLOSED: Normal operation, requests pass through
- OPEN: Service is failing, requests are blocked
- HALF_OPEN: Testing if service recovered
"""
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Callable, TypeVar, Optional, Any
import threading
import logging

T = TypeVar('T')

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = auto()      # Normal operation
    OPEN = auto()        # Failing, reject requests
    HALF_OPEN = auto()   # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    
    failure_threshold: int = 5          # Failures before opening
    success_threshold: int = 2          # Successes to close from half-open
    timeout_seconds: int = 30           # Time before trying half-open
    half_open_max_calls: int = 3        # Max calls in half-open state


class CircuitBreakerOpenError(Exception):
    """Raised when circuit is open and request is rejected."""
    
    def __init__(self, name: str, retry_after: Optional[float] = None):
        self.name = name
        self.retry_after = retry_after
        super().__init__(f"Circuit '{name}' is OPEN. Request rejected.")


class CircuitBreaker:
    """
    Circuit breaker for external service calls.
    
    Usage:
        breaker = CircuitBreaker("fmp_api", CircuitBreakerConfig(failure_threshold=3))
        
        # With call method
        result = breaker.call(api_function, *args, fallback=cached_data)
        
        # With decorator
        @breaker.protect
        def api_call():
            return requests.get(url)
    """
    
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        """
        Initialize circuit breaker.
        
        Args:
            name: Name for logging and identification
            config: Configuration options
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._half_open_calls = 0
        self._lock = threading.RLock()
    
    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        return self._state
    
    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self._state == CircuitState.CLOSED
    
    @property
    def is_open(self) -> bool:
        """Check if circuit is open (blocking requests)."""
        return self._state == CircuitState.OPEN
    
    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state."""
        old_state = self._state
        self._state = new_state
        
        if new_state == CircuitState.CLOSED:
            self._failure_count = 0
            self._success_count = 0
        elif new_state == CircuitState.HALF_OPEN:
            self._half_open_calls = 0
            self._success_count = 0
        
        logger.info(f"Circuit '{self.name}': {old_state.name} -> {new_state.name}")
    
    def _should_allow_request(self) -> bool:
        """Check if a request should be allowed through."""
        with self._lock:
            if self._state == CircuitState.CLOSED:
                return True
            
            if self._state == CircuitState.OPEN:
                # Check if timeout has passed
                if self._last_failure_time:
                    elapsed = datetime.utcnow() - self._last_failure_time
                    if elapsed > timedelta(seconds=self.config.timeout_seconds):
                        self._transition_to(CircuitState.HALF_OPEN)
                        return True
                return False
            
            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls < self.config.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False
            
            return False
    
    def _record_success(self) -> None:
        """Record a successful call."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._transition_to(CircuitState.CLOSED)
    
    def _record_failure(self) -> None:
        """Record a failed call."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = datetime.utcnow()
            
            if self._state == CircuitState.HALF_OPEN:
                self._transition_to(CircuitState.OPEN)
            elif self._failure_count >= self.config.failure_threshold:
                self._transition_to(CircuitState.OPEN)
    
    def call(
        self,
        func: Callable[..., T],
        *args,
        fallback: Optional[Callable[..., T]] = None,
        **kwargs
    ) -> T:
        """
        Execute function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args: Arguments to pass to func
            fallback: Optional fallback function if circuit is open or func fails
            **kwargs: Keyword arguments to pass to func
            
        Returns:
            Result from func or fallback
            
        Raises:
            CircuitBreakerOpenError: If circuit is open and no fallback provided
        """
        if not self._should_allow_request():
            if fallback:
                logger.warning(f"Circuit '{self.name}' is OPEN, using fallback")
                return fallback(*args, **kwargs)
            raise CircuitBreakerOpenError(self.name)
        
        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result
        except Exception as e:
            self._record_failure()
            logger.error(f"Circuit '{self.name}' recorded failure: {e}")
            if fallback:
                logger.warning(f"Using fallback for '{self.name}'")
                return fallback(*args, **kwargs)
            raise
    
    def protect(self, func: Callable[..., T]) -> Callable[..., T]:
        """
        Decorator to protect a function with circuit breaker.
        
        Usage:
            @breaker.protect
            def api_call():
                return requests.get(url)
        """
        def wrapper(*args, **kwargs) -> T:
            return self.call(func, *args, **kwargs)
        return wrapper
    
    def reset(self) -> None:
        """Manually reset the circuit breaker to closed state."""
        with self._lock:
            self._transition_to(CircuitState.CLOSED)
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = None
    
    def stats(self) -> dict:
        """Get circuit breaker statistics."""
        with self._lock:
            return {
                "name": self.name,
                "state": self._state.name,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "last_failure": self._last_failure_time.isoformat() if self._last_failure_time else None
            }


# Registry for managing multiple circuit breakers
class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers."""
    
    _breakers: dict[str, CircuitBreaker] = {}
    _lock = threading.Lock()
    
    @classmethod
    def get(cls, name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """Get or create a circuit breaker by name."""
        with cls._lock:
            if name not in cls._breakers:
                cls._breakers[name] = CircuitBreaker(name, config)
            return cls._breakers[name]
    
    @classmethod
    def status(cls) -> dict:
        """Get status of all circuit breakers."""
        return {name: breaker.state.name for name, breaker in cls._breakers.items()}
    
    @classmethod
    def reset_all(cls) -> None:
        """Reset all circuit breakers."""
        for breaker in cls._breakers.values():
            breaker.reset()
