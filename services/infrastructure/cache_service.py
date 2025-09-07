"""
Cache Service - Infrastructure service for caching operations.

This service provides clean abstractions for caching with multiple backends,
TTL management, and automatic serialization/deserialization.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import logging
import json
import hashlib
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class CacheBackend(Enum):
    """Supported cache backend types."""
    MEMORY = "memory"
    REDIS = "redis" 
    FILESYSTEM = "filesystem"


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    created_at: datetime
    expires_at: Optional[datetime]
    access_count: int = 0
    last_accessed: Optional[datetime] = None


class CacheBackendInterface(ABC):
    """Abstract interface for cache backends."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get value by key."""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[timedelta] = None) -> bool:
        """Set key-value pair with optional TTL."""
        pass
    
    @abstractmethod  
    def delete(self, key: str) -> bool:
        """Delete key."""
        pass
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists."""
        pass
    
    @abstractmethod
    def clear(self) -> bool:
        """Clear all cached data."""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        pass


class MemoryCacheBackend(CacheBackendInterface):
    """In-memory cache backend implementation."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._cache: Dict[str, CacheEntry] = {}
        self._hit_count = 0
        self._miss_count = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get value by key."""
        entry = self._cache.get(key)
        
        if entry is None:
            self._miss_count += 1
            return None
        
        # Check expiration
        if entry.expires_at and datetime.utcnow() > entry.expires_at:
            del self._cache[key]
            self._miss_count += 1
            return None
        
        # Update access stats
        entry.access_count += 1
        entry.last_accessed = datetime.utcnow()
        self._hit_count += 1
        
        return entry.value
    
    def set(self, key: str, value: Any, ttl: Optional[timedelta] = None) -> bool:
        """Set key-value pair with optional TTL."""
        try:
            # Handle cache size limit
            if len(self._cache) >= self.max_size and key not in self._cache:
                self._evict_lru()
            
            expires_at = None
            if ttl:
                expires_at = datetime.utcnow() + ttl
            
            self._cache[key] = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.utcnow(),
                expires_at=expires_at
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to cache key '{key}': {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key."""
        if key in self._cache:
            del self._cache[key]
            return True
        return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists and is not expired."""
        entry = self._cache.get(key)
        
        if entry is None:
            return False
        
        # Check expiration
        if entry.expires_at and datetime.utcnow() > entry.expires_at:
            del self._cache[key]
            return False
        
        return True
    
    def clear(self) -> bool:
        """Clear all cached data."""
        self._cache.clear()
        self._hit_count = 0
        self._miss_count = 0
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._hit_count + self._miss_count
        hit_rate = (self._hit_count / total_requests) * 100 if total_requests > 0 else 0
        
        # Clean expired entries for accurate count
        self._clean_expired()
        
        return {
            "backend": "memory",
            "total_keys": len(self._cache),
            "max_size": self.max_size,
            "hit_count": self._hit_count,
            "miss_count": self._miss_count,
            "hit_rate_percent": round(hit_rate, 2),
            "memory_usage_estimate": len(str(self._cache))  # Rough estimate
        }
    
    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self._cache:
            return
        
        # Find entry with oldest last_accessed (or created_at if never accessed)
        lru_key = min(
            self._cache.keys(),
            key=lambda k: self._cache[k].last_accessed or self._cache[k].created_at
        )
        
        del self._cache[lru_key]
    
    def _clean_expired(self) -> None:
        """Remove expired entries."""
        now = datetime.utcnow()
        expired_keys = [
            key for key, entry in self._cache.items()
            if entry.expires_at and now > entry.expires_at
        ]
        
        for key in expired_keys:
            del self._cache[key]


class CacheService:
    """
    Infrastructure service for caching operations.
    
    Provides:
    - Multiple cache backend support (memory, Redis, filesystem)
    - Automatic key hashing and serialization
    - TTL management and expiration
    - Cache statistics and monitoring
    - Namespace support for logical separation
    """
    
    def __init__(
        self,
        backend: CacheBackend = CacheBackend.MEMORY,
        namespace: str = "nirvana_cvar",
        default_ttl: Optional[timedelta] = None
    ):
        self.namespace = namespace
        self.default_ttl = default_ttl or timedelta(hours=1)
        
        # Initialize backend
        if backend == CacheBackend.MEMORY:
            self.backend = MemoryCacheBackend(max_size=1000)
        else:
            # For future implementation: Redis, filesystem backends
            logger.warning(f"Backend {backend.value} not implemented, using memory")
            self.backend = MemoryCacheBackend(max_size=1000)
    
    def get(self, key: str, namespace: Optional[str] = None) -> Optional[Any]:
        """
        Get cached value by key.
        
        Args:
            key: Cache key
            namespace: Optional namespace override
            
        Returns:
            Cached value or None if not found/expired
        """
        full_key = self._build_key(key, namespace)
        
        try:
            cached_data = self.backend.get(full_key)
            
            if cached_data is None:
                return None
            
            # Deserialize if needed
            return self._deserialize(cached_data)
            
        except Exception as e:
            logger.error(f"Cache get failed for key '{key}': {e}")
            return None
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[timedelta] = None,
        namespace: Optional[str] = None
    ) -> bool:
        """
        Set cached value with optional TTL.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live (uses default if not specified)
            namespace: Optional namespace override
            
        Returns:
            True if successfully cached
        """
        full_key = self._build_key(key, namespace)
        cache_ttl = ttl or self.default_ttl
        
        try:
            # Serialize value
            serialized_value = self._serialize(value)
            
            return self.backend.set(full_key, serialized_value, cache_ttl)
            
        except Exception as e:
            logger.error(f"Cache set failed for key '{key}': {e}")
            return False
    
    def delete(self, key: str, namespace: Optional[str] = None) -> bool:
        """Delete cached value by key."""
        full_key = self._build_key(key, namespace)
        return self.backend.delete(full_key)
    
    def exists(self, key: str, namespace: Optional[str] = None) -> bool:
        """Check if key exists in cache."""
        full_key = self._build_key(key, namespace)
        return self.backend.exists(full_key)
    
    def clear(self, namespace: Optional[str] = None) -> bool:
        """Clear all cache entries (or specific namespace)."""
        if namespace is None:
            return self.backend.clear()
        else:
            # For namespace-specific clearing, would need backend support
            logger.warning("Namespace-specific clearing not implemented")
            return False
    
    def get_or_set(
        self,
        key: str,
        factory_func: callable,
        ttl: Optional[timedelta] = None,
        namespace: Optional[str] = None
    ) -> Any:
        """
        Get cached value or compute and cache it using factory function.
        
        Args:
            key: Cache key
            factory_func: Function to call if cache miss
            ttl: Time to live
            namespace: Optional namespace override
            
        Returns:
            Cached or computed value
        """
        
        # Try to get from cache first
        cached_value = self.get(key, namespace)
        
        if cached_value is not None:
            return cached_value
        
        # Compute value using factory function
        try:
            computed_value = factory_func()
            
            # Cache the computed value
            self.set(key, computed_value, ttl, namespace)
            
            return computed_value
            
        except Exception as e:
            logger.error(f"Factory function failed for cache key '{key}': {e}")
            raise
    
    def cache_cvar_data(
        self,
        symbol: str,
        cvar_data: Dict[str, Any],
        ttl: Optional[timedelta] = None
    ) -> bool:
        """
        Cache CVaR calculation data for a symbol.
        
        Args:
            symbol: Financial symbol
            cvar_data: CVaR calculation result
            ttl: Cache duration
            
        Returns:
            True if cached successfully
        """
        key = f"cvar_data:{symbol}"
        return self.set(key, cvar_data, ttl, "cvar")
    
    def get_cached_cvar_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get cached CVaR data for a symbol."""
        key = f"cvar_data:{symbol}"
        return self.get(key, "cvar")
    
    def cache_price_data(
        self,
        symbol: str,
        exchange: str,
        price_data: List[Dict[str, Any]],
        ttl: Optional[timedelta] = None
    ) -> bool:
        """Cache price data for a symbol."""
        key = f"price_data:{symbol}:{exchange}"
        return self.set(key, price_data, ttl, "prices")
    
    def get_cached_price_data(
        self,
        symbol: str,
        exchange: str
    ) -> Optional[List[Dict[str, Any]]]:
        """Get cached price data for a symbol."""
        key = f"price_data:{symbol}:{exchange}"
        return self.get(key, "prices")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        backend_stats = self.backend.get_stats()
        
        return {
            "service_info": {
                "namespace": self.namespace,
                "default_ttl_seconds": self.default_ttl.total_seconds()
            },
            "backend_stats": backend_stats,
            "status": "operational"
        }
    
    def generate_key_hash(self, key: str) -> str:
        """Generate consistent hash for cache key."""
        return hashlib.md5(key.encode()).hexdigest()
    
    # Private helper methods
    
    def _build_key(self, key: str, namespace: Optional[str] = None) -> str:
        """Build full cache key with namespace."""
        ns = namespace or self.namespace
        return f"{ns}:{key}"
    
    def _serialize(self, value: Any) -> str:
        """Serialize value for caching."""
        if isinstance(value, (str, int, float, bool)):
            return json.dumps({"type": "primitive", "value": value})
        else:
            return json.dumps({"type": "object", "value": value})
    
    def _deserialize(self, serialized_value: str) -> Any:
        """Deserialize cached value."""
        try:
            data = json.loads(serialized_value)
            return data.get("value")
        except (json.JSONDecodeError, KeyError):
            # Fallback: return as-is
            return serialized_value
