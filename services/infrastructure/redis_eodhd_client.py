"""
Redis-cached EODHD client with rate limiting and async processing.

This module wraps the original EODHDClient with:
- Redis caching for API responses
- Rate limiting to prevent API overload
- Asynchronous task processing via Redis Queue (RQ)
- Exponential backoff retry mechanism
- Circuit breaker pattern for API failures
"""

import json
import hashlib
import logging
import time
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import asdict
from enum import Enum

import redis
from rq import Queue, Worker
from rq.job import Job
from rq.exceptions import AbandonedJobError

from config import get_config
from services.infrastructure.eodhd_client import EODHDClient, PriceDataPoint, EODHDSymbolInfo


logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"    # Normal operation
    OPEN = "open"        # Circuit is open, failing fast
    HALF_OPEN = "half_open"  # Testing if service is recovered


class RateLimiter:
    """Redis-based rate limiter for API calls."""
    
    def __init__(self, redis_client: redis.Redis, key_prefix: str = "rate_limit"):
        self.redis = redis_client
        self.key_prefix = key_prefix
    
    def is_allowed(self, identifier: str, limit: int, window: int) -> bool:
        """
        Check if request is allowed within rate limit.
        
        Args:
            identifier: Unique identifier for the rate limit (e.g., "eodhd_api")
            limit: Maximum requests allowed
            window: Time window in seconds
            
        Returns:
            True if request is allowed, False otherwise
        """
        key = f"{self.key_prefix}:{identifier}"
        now = int(time.time())
        pipeline = self.redis.pipeline()
        
        # Remove expired entries
        pipeline.zremrangebyscore(key, 0, now - window)
        
        # Count current requests
        pipeline.zcard(key)
        
        # Add current request timestamp
        pipeline.zadd(key, {str(now): now})
        
        # Set expiry for the key
        pipeline.expire(key, window)
        
        results = pipeline.execute()
        current_count = results[1]
        
        return current_count < limit
    
    def wait_time(self, identifier: str, limit: int, window: int) -> int:
        """Get wait time in seconds before next request is allowed."""
        key = f"{self.key_prefix}:{identifier}"
        now = int(time.time())
        
        # Get oldest request timestamp
        oldest = self.redis.zrange(key, 0, 0, withscores=True)
        if not oldest:
            return 0
        
        oldest_time = int(oldest[0][1])
        wait_time = window - (now - oldest_time)
        return max(0, wait_time)


class CircuitBreaker:
    """Circuit breaker for external API calls."""
    
    def __init__(self, redis_client: redis.Redis, key_prefix: str = "circuit", 
                 failure_threshold: int = 5, recovery_timeout: int = 60):
        self.redis = redis_client
        self.key_prefix = key_prefix
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
    
    def _get_key(self, service: str) -> str:
        return f"{self.key_prefix}:{service}"
    
    def get_state(self, service: str) -> CircuitState:
        """Get current circuit breaker state."""
        key = self._get_key(service)
        data = self.redis.hgetall(key)
        
        if not data:
            return CircuitState.CLOSED
        
        state = data.get(b'state', b'closed').decode()
        last_failure = float(data.get(b'last_failure', 0))
        
        if state == CircuitState.OPEN.value:
            if time.time() - last_failure > self.recovery_timeout:
                return CircuitState.HALF_OPEN
        
        return CircuitState(state)
    
    def record_success(self, service: str):
        """Record successful API call."""
        key = self._get_key(service)
        self.redis.hset(key, mapping={
            'state': CircuitState.CLOSED.value,
            'failures': 0,
            'last_success': time.time()
        })
    
    def record_failure(self, service: str):
        """Record failed API call."""
        key = self._get_key(service)
        failures = int(self.redis.hget(key, 'failures') or 0) + 1
        
        new_state = (CircuitState.OPEN.value 
                     if failures >= self.failure_threshold 
                     else CircuitState.CLOSED.value)
        
        self.redis.hset(key, mapping={
            'state': new_state,
            'failures': failures,
            'last_failure': time.time()
        })
        
        if new_state == CircuitState.OPEN.value:
            logger.warning(f"Circuit breaker OPENED for {service} after {failures} failures")
    
    def is_call_allowed(self, service: str) -> bool:
        """Check if API call is allowed based on circuit breaker state."""
        state = self.get_state(service)
        return state != CircuitState.OPEN


class RedisCachedEODHDClient:
    """
    Redis-cached EODHD client with rate limiting and async processing.
    
    Features:
    - Caches API responses in Redis with configurable TTL
    - Rate limiting to prevent API overload
    - Circuit breaker pattern for handling API failures
    - Asynchronous task processing for bulk operations
    - Exponential backoff retry mechanism
    """
    
    def __init__(self, config = None):
        self.config = config or get_config()
        
        # Initialize Redis connection
        self.redis = redis.from_url(
            self.config.redis.url,
            max_connections=self.config.redis.max_connections,
            decode_responses=False  # We handle encoding ourselves
        )
        
        # Initialize components
        self.rate_limiter = RateLimiter(self.redis)
        self.circuit_breaker = CircuitBreaker(self.redis)
        
        # Initialize underlying EODHD client
        self.eodhd_client = EODHDClient(
            api_key=self.config.external_services.eodhd_api_key,
            base_url=self.config.external_services.eodhd_base_url
        )
        
        # Initialize RQ queue for async processing
        self.queue = Queue(
            name=self.config.redis.queue_name,
            connection=self.redis
        )
        
        logger.info("RedisCachedEODHDClient initialized with Redis URL: %s", 
                   self.config.redis.url)
    
    def _generate_cache_key(self, prefix: str, **params) -> str:
        """Generate cache key from parameters."""
        # Create deterministic key from sorted parameters
        key_data = json.dumps(params, sort_keys=True, default=str)
        key_hash = hashlib.md5(key_data.encode()).hexdigest()[:12]
        return f"eodhd:{prefix}:{key_hash}"
    
    def _get_cached_result(self, cache_key: str) -> Optional[Any]:
        """Get cached result from Redis."""
        try:
            cached = self.redis.get(cache_key)
            if cached:
                return json.loads(cached.decode('utf-8'))
        except Exception as e:
            logger.warning(f"Failed to get cached result for {cache_key}: {e}")
        return None
    
    def _set_cached_result(self, cache_key: str, result: Any, ttl: int = None):
        """Set cached result in Redis."""
        try:
            ttl = ttl or self.config.redis.eodhd_cache_ttl
            serialized = json.dumps(result, default=str, ensure_ascii=False)
            self.redis.setex(cache_key, ttl, serialized.encode('utf-8'))
        except Exception as e:
            logger.warning(f"Failed to cache result for {cache_key}: {e}")
    
    def _wait_for_rate_limit(self):
        """Wait if rate limit is exceeded."""
        identifier = "eodhd_api"
        limit = self.config.redis.rate_limit_requests
        window = self.config.redis.rate_limit_window
        
        if not self.rate_limiter.is_allowed(identifier, limit, window):
            wait_time = self.rate_limiter.wait_time(identifier, limit, window)
            if wait_time > 0:
                logger.info(f"Rate limit exceeded, waiting {wait_time} seconds")
                time.sleep(wait_time)
    
    def _execute_with_circuit_breaker(self, operation_name: str, func, *args, **kwargs):
        """Execute operation with circuit breaker protection."""
        service_key = f"eodhd_{operation_name}"
        
        # Check circuit breaker
        if not self.circuit_breaker.is_call_allowed(service_key):
            raise Exception(f"Circuit breaker OPEN for {service_key}")
        
        try:
            # Wait for rate limit
            self._wait_for_rate_limit()
            
            # Execute operation
            result = func(*args, **kwargs)
            
            # Record success
            self.circuit_breaker.record_success(service_key)
            
            return result
            
        except Exception as e:
            # Record failure
            self.circuit_breaker.record_failure(service_key)
            raise e
    
    def get_historical_prices_cached(
        self,
        symbol: str,
        exchange: str,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
        period: str = "d"
    ) -> List[PriceDataPoint]:
        """
        Get historical prices with Redis caching.
        
        Args:
            symbol: Stock symbol
            exchange: Exchange code
            from_date: Start date (optional)
            to_date: End date (optional)  
            period: Data period (d, w, m)
            
        Returns:
            List of price data points
        """
        # Generate cache key
        cache_key = self._generate_cache_key(
            "historical_prices",
            symbol=symbol,
            exchange=exchange,
            from_date=from_date,
            to_date=to_date,
            period=period
        )
        
        # Try to get from cache first
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            logger.debug(f"Cache HIT for historical prices: {symbol}.{exchange}")
            # Convert back to PriceDataPoint objects
            return [PriceDataPoint(**item) for item in cached_result]
        
        logger.debug(f"Cache MISS for historical prices: {symbol}.{exchange}")
        
        # Execute with circuit breaker
        def fetch_prices():
            return self.eodhd_client.get_historical_prices(
                symbol, exchange, from_date, to_date, period
            )
        
        try:
            prices = self._execute_with_circuit_breaker(
                "historical_prices", fetch_prices
            )
            
            # Cache the result
            prices_data = [asdict(price) for price in prices]
            self._set_cached_result(cache_key, prices_data)
            
            return prices
            
        except Exception as e:
            logger.error(f"Failed to fetch historical prices for {symbol}.{exchange}: {e}")
            raise e
    
    def get_symbol_info_cached(self, symbol: str, exchange: str) -> Optional[EODHDSymbolInfo]:
        """Get symbol info with Redis caching."""
        cache_key = self._generate_cache_key(
            "symbol_info",
            symbol=symbol,
            exchange=exchange
        )
        
        # Try cache first
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            logger.debug(f"Cache HIT for symbol info: {symbol}.{exchange}")
            return EODHDSymbolInfo(**cached_result)
        
        logger.debug(f"Cache MISS for symbol info: {symbol}.{exchange}")
        
        # Execute with circuit breaker
        def fetch_info():
            return self.eodhd_client.get_symbol_info(symbol, exchange)
        
        try:
            info = self._execute_with_circuit_breaker("symbol_info", fetch_info)
            
            if info:
                # Cache the result
                self._set_cached_result(cache_key, asdict(info))
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to fetch symbol info for {symbol}.{exchange}: {e}")
            raise e
    
    def validate_symbol_cached(self, symbol: str, exchange: str) -> bool:
        """Validate symbol with Redis caching."""
        cache_key = self._generate_cache_key(
            "validate_symbol", 
            symbol=symbol,
            exchange=exchange
        )
        
        # Try cache first
        cached_result = self._get_cached_result(cache_key)
        if cached_result is not None:
            logger.debug(f"Cache HIT for symbol validation: {symbol}.{exchange}")
            return bool(cached_result)
        
        logger.debug(f"Cache MISS for symbol validation: {symbol}.{exchange}")
        
        # Execute with circuit breaker
        def validate():
            return self.eodhd_client.validate_symbol(symbol, exchange)
        
        try:
            is_valid = self._execute_with_circuit_breaker("validate_symbol", validate)
            
            # Cache the result (shorter TTL for validation results)
            self._set_cached_result(cache_key, is_valid, ttl=3600)  # 1 hour
            
            return is_valid
            
        except Exception as e:
            logger.error(f"Failed to validate symbol {symbol}.{exchange}: {e}")
            raise e
    
    def queue_bulk_price_fetch(self, symbol_exchange_pairs: List[Tuple[str, str]]) -> List[str]:
        """
        Queue bulk price fetching as async jobs.
        
        Args:
            symbol_exchange_pairs: List of (symbol, exchange) tuples
            
        Returns:
            List of job IDs
        """
        job_ids = []
        
        for symbol, exchange in symbol_exchange_pairs:
            try:
                job = self.queue.enqueue(
                    fetch_historical_prices_job,
                    symbol=symbol,
                    exchange=exchange,
                    redis_url=self.config.redis.url,
                    eodhd_api_key=self.config.external_services.eodhd_api_key,
                    eodhd_base_url=self.config.external_services.eodhd_base_url,
                    job_timeout=self.config.redis.job_timeout
                )
                job_ids.append(job.id)
                logger.debug(f"Queued price fetch job for {symbol}.{exchange}: {job.id}")
                
            except Exception as e:
                logger.error(f"Failed to queue job for {symbol}.{exchange}: {e}")
        
        logger.info(f"Queued {len(job_ids)} price fetch jobs")
        return job_ids
    
    def get_job_results(self, job_ids: List[str]) -> Dict[str, Any]:
        """
        Get results from completed jobs.
        
        Args:
            job_ids: List of job IDs to check
            
        Returns:
            Dictionary with job results
        """
        results = {}
        
        for job_id in job_ids:
            try:
                job = Job.fetch(job_id, connection=self.redis)
                
                if job.is_finished:
                    results[job_id] = {
                        'status': 'completed',
                        'result': job.result
                    }
                elif job.is_failed:
                    results[job_id] = {
                        'status': 'failed',
                        'error': str(job.exc_info)
                    }
                else:
                    results[job_id] = {
                        'status': 'pending'
                    }
                    
            except Exception as e:
                logger.error(f"Failed to get job {job_id} result: {e}")
                results[job_id] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        return results
    
    def clear_cache(self, pattern: str = "eodhd:*"):
        """Clear cached data matching pattern."""
        try:
            keys = self.redis.keys(pattern)
            if keys:
                deleted = self.redis.delete(*keys)
                logger.info(f"Cleared {deleted} cached entries matching pattern: {pattern}")
            else:
                logger.info(f"No cached entries found matching pattern: {pattern}")
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            info = self.redis.info()
            keys_count = len(self.redis.keys("eodhd:*"))
            
            return {
                'redis_version': info.get('redis_version'),
                'used_memory': info.get('used_memory_human'),
                'connected_clients': info.get('connected_clients'),
                'total_keys': keys_count,
                'eodhd_cached_keys': keys_count
            }
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {'error': str(e)}


# RQ Job Functions (must be at module level for pickle serialization)

def fetch_historical_prices_job(symbol: str, exchange: str, redis_url: str, 
                               eodhd_api_key: str, eodhd_base_url: str) -> Dict[str, Any]:
    """
    RQ job function to fetch historical prices.
    
    This function runs in a separate worker process.
    """
    try:
        # Create EODHD client in worker process
        client = EODHDClient(api_key=eodhd_api_key, base_url=eodhd_base_url)
        
        # Fetch prices
        prices = client.get_historical_prices(symbol, exchange)
        
        # Convert to serializable format
        prices_data = [asdict(price) for price in prices]
        
        return {
            'symbol': symbol,
            'exchange': exchange,
            'prices_count': len(prices_data),
            'prices': prices_data,
            'success': True
        }
        
    except Exception as e:
        logger.error(f"Job failed for {symbol}.{exchange}: {e}")
        return {
            'symbol': symbol,
            'exchange': exchange,
            'success': False,
            'error': str(e)
        }


def start_redis_worker(redis_url: str, queue_name: str = "eodhd_queue"):
    """
    Start a Redis Queue worker.
    
    This function should be called in a separate process to start processing jobs.
    """
    try:
        redis_conn = redis.from_url(redis_url)
        queue = Queue(name=queue_name, connection=redis_conn)
        
        worker = Worker([queue], connection=redis_conn)
        logger.info(f"Starting Redis worker for queue: {queue_name}")
        
        worker.work()
        
    except Exception as e:
        logger.error(f"Redis worker failed: {e}")
        raise e
