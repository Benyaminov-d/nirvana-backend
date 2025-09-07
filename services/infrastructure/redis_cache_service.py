"""
Redis Cache Service - Core caching infrastructure layer.

This module provides a clean abstraction over Redis operations with:
- Type-safe caching operations
- Configurable TTL policies
- Comprehensive error handling
- JSON serialization/deserialization
- Key management and namespacing
- Performance monitoring
"""

import json
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, TypeVar, Generic, Callable
from dataclasses import dataclass
from enum import Enum

import redis
from redis.exceptions import RedisError, ConnectionError, TimeoutError

from config import get_config


logger = logging.getLogger(__name__)
T = TypeVar('T')


class CacheKeyType(Enum):
    """Standard cache key types for consistent namespacing."""
    CVAR_RESULT = "cvar:result"
    VALIDATION_FLAGS = "validation:flags" 
    FIVE_STARS_FILTER = "filter:five_stars"
    MU_PARAMETER = "mu:computed"
    QUERY_RESULT = "query:result"
    TICKER_FEED = "ticker:feed"
    SEARCH_RESULT = "search:result"


@dataclass
class CachePolicy:
    """Cache policy configuration for different data types."""
    ttl_seconds: int
    key_type: CacheKeyType
    enable_compression: bool = False
    max_size_bytes: Optional[int] = None


class StandardCachePolicies:
    """Pre-defined cache policies for common use cases."""
    
    CVAR_RESULTS = CachePolicy(
        ttl_seconds=7 * 24 * 3600,  # 7 days
        key_type=CacheKeyType.CVAR_RESULT
    )
    
    VALIDATION_FLAGS = CachePolicy(
        ttl_seconds=24 * 3600,  # 24 hours
        key_type=CacheKeyType.VALIDATION_FLAGS
    )
    
    FIVE_STARS_FILTER = CachePolicy(
        ttl_seconds=3600,  # 1 hour
        key_type=CacheKeyType.FIVE_STARS_FILTER
    )
    
    MU_PARAMETERS = CachePolicy(
        ttl_seconds=24 * 3600,  # 24 hours
        key_type=CacheKeyType.MU_PARAMETER
    )
    
    TICKER_FEED = CachePolicy(
        ttl_seconds=300,  # 5 minutes
        key_type=CacheKeyType.TICKER_FEED
    )
    
    SEARCH_RESULTS = CachePolicy(
        ttl_seconds=600,  # 10 minutes
        key_type=CacheKeyType.SEARCH_RESULT
    )


class RedisCacheService:
    """
    Production-ready Redis caching service.
    
    Features:
    - Type-safe operations with generic methods
    - Automatic JSON serialization/deserialization
    - Configurable cache policies
    - Comprehensive error handling with fallback strategies
    - Performance monitoring and logging
    - Key management with consistent namespacing
    """
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.redis_client = None
        self._connection_healthy = False
        
        self._init_redis_connection()
    
    def _init_redis_connection(self) -> None:
        """Initialize Redis connection with retry logic."""
        try:
            self.redis_client = redis.from_url(
                self.config.redis.url,
                max_connections=self.config.redis.max_connections,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                decode_responses=False  # We handle encoding ourselves
            )
            
            # Test connection
            self.redis_client.ping()
            self._connection_healthy = True
            
            logger.info("Redis cache service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis connection: {e}")
            self._connection_healthy = False
    
    def _generate_cache_key(self, key_type: CacheKeyType, *key_parts: str) -> str:
        """
        Generate consistent cache keys with proper namespacing.
        
        Args:
            key_type: Type of cache key for namespacing
            key_parts: Variable parts of the key
            
        Returns:
            Properly formatted cache key
        """
        # Join key parts and create hash for long keys
        key_content = ":".join(str(part) for part in key_parts if part)
        
        # Use hash for long keys to avoid Redis key length limits
        if len(key_content) > 100:
            key_hash = hashlib.sha256(key_content.encode()).hexdigest()[:16]
            return f"{key_type.value}:hash:{key_hash}"
        
        return f"{key_type.value}:{key_content}"
    
    def _serialize_value(self, value: Any) -> bytes:
        """Serialize value to bytes for Redis storage."""
        try:
            if value is None:
                return b'null'
            
            serialized = json.dumps(value, default=str, ensure_ascii=False)
            return serialized.encode('utf-8')
            
        except (TypeError, ValueError) as e:
            logger.warning(f"Failed to serialize value: {e}")
            raise ValueError(f"Value is not serializable: {type(value)}")
    
    def _deserialize_value(self, data: bytes) -> Any:
        """Deserialize bytes from Redis to Python value."""
        try:
            if data == b'null':
                return None
            
            decoded = data.decode('utf-8')
            return json.loads(decoded)
            
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.warning(f"Failed to deserialize value: {e}")
            return None
    
    def _is_healthy(self) -> bool:
        """Check if Redis connection is healthy."""
        if not self._connection_healthy or not self.redis_client:
            return False
        
        try:
            self.redis_client.ping()
            return True
        except RedisError:
            self._connection_healthy = False
            logger.warning("Redis connection health check failed")
            return False
    
    def set(
        self, 
        key_type: CacheKeyType,
        value: T,
        policy: CachePolicy,
        *key_parts: str
    ) -> bool:
        """
        Store value in cache with specified policy.
        
        Args:
            key_type: Type of cache key
            value: Value to cache
            policy: Cache policy with TTL and configuration
            key_parts: Variable parts of the cache key
            
        Returns:
            True if successfully cached, False otherwise
        """
        if not self._is_healthy():
            logger.debug("Redis not healthy, skipping cache set")
            return False
        
        try:
            cache_key = self._generate_cache_key(key_type, *key_parts)
            serialized = self._serialize_value(value)
            
            # Check size limits if configured
            if policy.max_size_bytes and len(serialized) > policy.max_size_bytes:
                logger.warning(f"Value exceeds size limit: {len(serialized)} bytes")
                return False
            
            # Set with TTL
            success = self.redis_client.setex(
                cache_key, 
                policy.ttl_seconds, 
                serialized
            )
            
            if success:
                logger.debug(f"Cached value with key: {cache_key}")
            
            return bool(success)
            
        except Exception as e:
            logger.warning(f"Failed to cache value: {e}")
            return False
    
    def get(
        self, 
        key_type: CacheKeyType, 
        *key_parts: str,
        default: Optional[T] = None
    ) -> Optional[T]:
        """
        Retrieve value from cache.
        
        Args:
            key_type: Type of cache key
            key_parts: Variable parts of the cache key
            default: Default value if not found
            
        Returns:
            Cached value or default
        """
        if not self._is_healthy():
            logger.debug("Redis not healthy, returning default")
            return default
        
        try:
            cache_key = self._generate_cache_key(key_type, *key_parts)
            data = self.redis_client.get(cache_key)
            
            if data is None:
                logger.debug(f"Cache miss for key: {cache_key}")
                return default
            
            value = self._deserialize_value(data)
            logger.debug(f"Cache hit for key: {cache_key}")
            return value
            
        except Exception as e:
            logger.warning(f"Failed to retrieve cached value: {e}")
            return default
    
    def delete(self, key_type: CacheKeyType, *key_parts: str) -> bool:
        """
        Delete value from cache.
        
        Args:
            key_type: Type of cache key
            key_parts: Variable parts of the cache key
            
        Returns:
            True if successfully deleted, False otherwise
        """
        if not self._is_healthy():
            return False
        
        try:
            cache_key = self._generate_cache_key(key_type, *key_parts)
            deleted = self.redis_client.delete(cache_key)
            
            if deleted:
                logger.debug(f"Deleted cache key: {cache_key}")
            
            return bool(deleted)
            
        except Exception as e:
            logger.warning(f"Failed to delete cached value: {e}")
            return False
    
    def exists(self, key_type: CacheKeyType, *key_parts: str) -> bool:
        """Check if key exists in cache."""
        if not self._is_healthy():
            return False
        
        try:
            cache_key = self._generate_cache_key(key_type, *key_parts)
            return bool(self.redis_client.exists(cache_key))
            
        except Exception as e:
            logger.warning(f"Failed to check key existence: {e}")
            return False
    
    def invalidate_pattern(self, pattern: str) -> int:
        """
        Invalidate multiple keys matching pattern.
        
        Args:
            pattern: Redis key pattern (e.g., "cvar:result:*")
            
        Returns:
            Number of keys deleted
        """
        if not self._is_healthy():
            return 0
        
        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                deleted = self.redis_client.delete(*keys)
                logger.info(f"Invalidated {deleted} keys matching pattern: {pattern}")
                return deleted
            
            return 0
            
        except Exception as e:
            logger.warning(f"Failed to invalidate pattern {pattern}: {e}")
            return 0
    
    def get_or_compute(
        self,
        key_type: CacheKeyType,
        policy: CachePolicy,
        compute_func: Callable[[], T],
        *key_parts: str
    ) -> T:
        """
        Get value from cache or compute if not present.
        
        Args:
            key_type: Type of cache key
            policy: Cache policy
            compute_func: Function to compute value if not cached
            key_parts: Variable parts of the cache key
            
        Returns:
            Cached value or computed value
        """
        # Try to get from cache first
        cached_value = self.get(key_type, *key_parts)
        if cached_value is not None:
            return cached_value
        
        # Compute value
        try:
            computed_value = compute_func()
            
            # Cache the computed value
            self.set(key_type, computed_value, policy, *key_parts)
            
            return computed_value
            
        except Exception as e:
            logger.error(f"Failed to compute value for cache: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        if not self._is_healthy():
            return {"status": "unhealthy"}
        
        try:
            info = self.redis_client.info()
            
            # Count keys by type
            key_counts = {}
            for key_type in CacheKeyType:
                pattern = f"{key_type.value}:*"
                count = len(self.redis_client.keys(pattern))
                key_counts[key_type.value] = count
            
            return {
                "status": "healthy",
                "redis_version": info.get("redis_version"),
                "used_memory": info.get("used_memory_human"),
                "connected_clients": info.get("connected_clients"),
                "total_commands_processed": info.get("total_commands_processed"),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "hit_rate": self._calculate_hit_rate(info),
                "key_counts": key_counts
            }
            
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {"status": "error", "error": str(e)}
    
    def _calculate_hit_rate(self, info: Dict) -> float:
        """Calculate cache hit rate percentage."""
        hits = info.get("keyspace_hits", 0)
        misses = info.get("keyspace_misses", 0)
        
        if hits + misses == 0:
            return 0.0
        
        return round((hits / (hits + misses)) * 100, 2)
    
    def clear_all(self) -> bool:
        """Clear all cached data. Use with caution."""
        if not self._is_healthy():
            return False
        
        try:
            self.redis_client.flushdb()
            logger.warning("Cleared all cached data")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return False


# Global cache service instance
_cache_service: Optional[RedisCacheService] = None


def get_cache_service() -> RedisCacheService:
    """Get global cache service instance."""
    global _cache_service
    
    if _cache_service is None:
        _cache_service = RedisCacheService()
    
    return _cache_service
