"""Simple metrics collector for application monitoring.

Provides counters, histograms, and gauges for tracking application performance.
Can be extended to export to Prometheus, StatsD, or other monitoring systems.
"""
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional, Callable, TypeVar
from contextlib import contextmanager
from functools import wraps
import time
import threading

T = TypeVar('T')


@dataclass
class MetricsCollector:
    """
    Simple metrics collector for application monitoring.
    
    Usage:
        metrics = MetricsCollector()
        
        # Counters
        metrics.increment("queries_processed")
        metrics.increment("errors", value=1, tags={"type": "database"})
        
        # Histograms (for timing/distributions)
        metrics.histogram("query_latency_ms", 150.5)
        
        # Gauges (current values)
        metrics.gauge("active_connections", 5)
        
        # Timer context manager
        with metrics.timer("database_query"):
            execute_query()
    """
    
    _counters: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    _histograms: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    _gauges: dict[str, float] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    
    def _make_key(self, name: str, tags: Optional[dict] = None) -> str:
        """Create a unique key for a metric with tags."""
        if not tags:
            return name
        tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name}{{{tag_str}}}"
    
    def increment(self, name: str, value: int = 1, tags: Optional[dict] = None) -> None:
        """Increment a counter."""
        key = self._make_key(name, tags)
        with self._lock:
            self._counters[key] += value
    
    def histogram(self, name: str, value: float, tags: Optional[dict] = None) -> None:
        """Record a value in a histogram."""
        key = self._make_key(name, tags)
        with self._lock:
            self._histograms[key].append(value)
    
    def gauge(self, name: str, value: float, tags: Optional[dict] = None) -> None:
        """Set a gauge value."""
        key = self._make_key(name, tags)
        with self._lock:
            self._gauges[key] = value
    
    @contextmanager
    def timer(self, name: str, tags: Optional[dict] = None):
        """
        Context manager for timing operations.
        
        Usage:
            with metrics.timer("database_query"):
                execute_query()
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self.histogram(f"{name}_ms", elapsed_ms, tags)
    
    def get_counter(self, name: str, tags: Optional[dict] = None) -> int:
        """Get a counter value."""
        key = self._make_key(name, tags)
        return self._counters.get(key, 0)
    
    def get_gauge(self, name: str, tags: Optional[dict] = None) -> Optional[float]:
        """Get a gauge value."""
        key = self._make_key(name, tags)
        return self._gauges.get(key)
    
    def get_histogram_stats(self, name: str, tags: Optional[dict] = None) -> dict:
        """Get histogram statistics."""
        key = self._make_key(name, tags)
        values = self._histograms.get(key, [])
        if not values:
            return {"count": 0, "avg": 0, "min": 0, "max": 0, "p50": 0, "p95": 0}
        
        sorted_values = sorted(values)
        count = len(values)
        
        return {
            "count": count,
            "avg": sum(values) / count,
            "min": sorted_values[0],
            "max": sorted_values[-1],
            "p50": sorted_values[int(count * 0.5)],
            "p95": sorted_values[int(count * 0.95)] if count >= 20 else sorted_values[-1]
        }
    
    def get_stats(self) -> dict:
        """Get all metrics statistics."""
        with self._lock:
            return {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": {
                    k: self.get_histogram_stats(k)
                    for k in self._histograms.keys()
                }
            }
    
    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._counters.clear()
            self._histograms.clear()
            self._gauges.clear()


# Global metrics instance
_metrics: Optional[MetricsCollector] = None


def get_metrics() -> MetricsCollector:
    """Get the global metrics instance."""
    global _metrics
    if _metrics is None:
        _metrics = MetricsCollector()
    return _metrics


def timed(metric_name: str, tags: Optional[dict] = None):
    """
    Decorator to time function execution.
    
    Usage:
        @timed("process_query")
        def process_query(query: str):
            # ... processing ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            metrics = get_metrics()
            with metrics.timer(metric_name, tags):
                return func(*args, **kwargs)
        return wrapper
    return decorator
