"""In-memory LRU cache with TTL support.

Provides thread-safe caching with:
- LRU (Least Recently Used) eviction
- Time-to-live (TTL) expiration
- Configurable maximum size
"""
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Generic, TypeVar, Optional, Any, Callable
from collections import OrderedDict
import threading
import hashlib
import json
import functools

T = TypeVar('T')


@dataclass
class CacheEntry(Generic[T]):
    """A cache entry with value and expiration tracking."""
    
    value: T
    created_at: datetime
    ttl_seconds: int
    
    @property
    def is_expired(self) -> bool:
        """Check if the entry has expired."""
        if self.ttl_seconds <= 0:
            return False  # No expiration
        return datetime.utcnow() > self.created_at + timedelta(seconds=self.ttl_seconds)


class MemoryCache:
    """
    Thread-safe in-memory LRU cache with TTL support.
    
    Usage:
        cache = MemoryCache(max_size=1000, default_ttl=3600)
        
        # Set and get
        cache.set("key", value, ttl=300)
        value = cache.get("key")
        
        # With decorator
        @cache.cached(ttl=300)
        def expensive_operation(arg):
            return compute(arg)
    """
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        """
        Initialize the cache.
        
        Args:
            max_size: Maximum number of entries (default: 1000)
            default_ttl: Default time-to-live in seconds (default: 3600)
        """
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None
            
            entry = self._cache[key]
            if entry.is_expired:
                del self._cache[key]
                self._misses += 1
                return None
            
            # Move to end (LRU - most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            return entry.value
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (uses default if not specified)
        """
        with self._lock:
            # Evict if at capacity
            while len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)  # Remove oldest
            
            self._cache[key] = CacheEntry(
                value=value,
                created_at=datetime.utcnow(),
                ttl_seconds=ttl if ttl is not None else self._default_ttl
            )
    
    def delete(self, key: str) -> None:
        """Delete a value from the cache."""
        with self._lock:
            self._cache.pop(key, None)
    
    def clear(self) -> None:
        """Clear all cached values."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
    
    def stats(self) -> dict:
        """Get cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = (self._hits / total * 100) if total > 0 else 0
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "default_ttl": self._default_ttl,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate_percent": round(hit_rate, 2)
            }
    
    def cached(self, ttl: Optional[int] = None, key_prefix: str = ""):
        """
        Decorator to cache function results.
        
        Args:
            ttl: Time-to-live for cached results
            key_prefix: Prefix for cache keys
            
        Usage:
            @cache.cached(ttl=300)
            def expensive_function(arg1, arg2):
                return compute(arg1, arg2)
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Build cache key from function name and arguments
                key_data = json.dumps({
                    "args": [str(a) for a in args],
                    "kwargs": {k: str(v) for k, v in sorted(kwargs.items())}
                }, sort_keys=True)
                cache_key = f"{key_prefix}{func.__name__}:{hashlib.md5(key_data.encode()).hexdigest()}"
                
                # Try cache first
                cached_value = self.get(cache_key)
                if cached_value is not None:
                    return cached_value
                
                # Execute function
                result = func(*args, **kwargs)
                
                # Cache result
                self.set(cache_key, result, ttl)
                
                return result
            
            return wrapper
        return decorator


# Global cache instance for convenience
_default_cache: Optional[MemoryCache] = None


def get_cache(max_size: int = 1000, default_ttl: int = 3600) -> MemoryCache:
    """Get or create the default cache instance."""
    global _default_cache
    if _default_cache is None:
        _default_cache = MemoryCache(max_size=max_size, default_ttl=default_ttl)
    return _default_cache
