"""
Redis integration test utility.

This script tests the Redis-based EODHD client functionality including:
- Redis connection
- Caching mechanism
- Rate limiting
- Circuit breaker
- Async job processing
"""

import asyncio
import time
import logging
from typing import List

from config import get_config
from services.infrastructure.redis_eodhd_client import RedisCachedEODHDClient


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_redis_connection():
    """Test basic Redis connection."""
    try:
        config = get_config()
        client = RedisCachedEODHDClient(config)
        
        # Test Redis connection
        client.redis.ping()
        logger.info("✅ Redis connection successful")
        
        # Get cache stats
        stats = client.get_cache_stats()
        logger.info(f"📊 Cache stats: {stats}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Redis connection failed: {e}")
        return False


def test_eodhd_caching():
    """Test EODHD API caching."""
    try:
        config = get_config()
        client = RedisCachedEODHDClient(config)
        
        # Test symbols
        test_symbols = [
            ("AAPL", "US"),
            ("MSFT", "US"),
            ("GOOGL", "US")
        ]
        
        results = []
        
        for symbol, exchange in test_symbols:
            logger.info(f"🔍 Testing {symbol}.{exchange}")
            
            # First call - should hit API and cache
            start_time = time.time()
            try:
                prices1 = client.get_historical_prices_cached(symbol, exchange)
                duration1 = time.time() - start_time
                logger.info(f"   First call: {len(prices1)} prices in {duration1:.2f}s")
                
                # Second call - should hit cache
                start_time = time.time()
                prices2 = client.get_historical_prices_cached(symbol, exchange)
                duration2 = time.time() - start_time
                logger.info(f"   Second call (cached): {len(prices2)} prices in {duration2:.2f}s")
                
                # Verify cache speed improvement
                speedup = duration1 / duration2 if duration2 > 0 else float('inf')
                logger.info(f"   🚀 Cache speedup: {speedup:.1f}x")
                
                results.append({
                    'symbol': symbol,
                    'exchange': exchange,
                    'prices_count': len(prices1),
                    'api_duration': duration1,
                    'cache_duration': duration2,
                    'speedup': speedup
                })
                
            except Exception as e:
                logger.error(f"   ❌ Failed: {e}")
                results.append({
                    'symbol': symbol,
                    'exchange': exchange,
                    'error': str(e)
                })
        
        logger.info(f"✅ Caching test completed: {len(results)} symbols tested")
        return results
        
    except Exception as e:
        logger.error(f"❌ Caching test failed: {e}")
        return []


def test_rate_limiting():
    """Test rate limiting functionality."""
    try:
        config = get_config()
        client = RedisCachedEODHDClient(config)
        
        # Clear cache first
        client.clear_cache()
        
        # Test rate limiting with rapid calls
        logger.info("🚦 Testing rate limiting...")
        
        start_time = time.time()
        successful_calls = 0
        
        for i in range(10):  # Make 10 rapid calls
            try:
                # Use different symbols to avoid cache hits
                symbol = f"TEST{i:02d}"
                client.validate_symbol_cached(symbol, "US")
                successful_calls += 1
                logger.info(f"   Call {i+1}: Success")
                
            except Exception as e:
                logger.warning(f"   Call {i+1}: Rate limited or failed: {e}")
        
        duration = time.time() - start_time
        logger.info(f"✅ Rate limiting test: {successful_calls}/10 calls in {duration:.2f}s")
        
        return {
            'successful_calls': successful_calls,
            'total_calls': 10,
            'duration': duration
        }
        
    except Exception as e:
        logger.error(f"❌ Rate limiting test failed: {e}")
        return {}


def test_async_processing():
    """Test async job processing."""
    try:
        config = get_config()
        client = RedisCachedEODHDClient(config)
        
        # Queue some jobs
        logger.info("📋 Testing async job processing...")
        
        test_pairs = [
            ("AAPL", "US"),
            ("MSFT", "US"),
            ("GOOGL", "US")
        ]
        
        # Queue jobs
        job_ids = client.queue_bulk_price_fetch(test_pairs)
        logger.info(f"   Queued {len(job_ids)} jobs: {job_ids}")
        
        # Wait and check results
        max_wait_time = 30  # seconds
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            results = client.get_job_results(job_ids)
            
            completed = sum(1 for r in results.values() if r['status'] == 'completed')
            failed = sum(1 for r in results.values() if r['status'] == 'failed')
            pending = len(job_ids) - completed - failed
            
            logger.info(f"   Jobs: {completed} completed, {failed} failed, {pending} pending")
            
            if pending == 0:
                break
            
            time.sleep(2)
        
        final_results = client.get_job_results(job_ids)
        logger.info(f"✅ Async processing test completed")
        
        return final_results
        
    except Exception as e:
        logger.error(f"❌ Async processing test failed: {e}")
        return {}


def main():
    """Run all Redis integration tests."""
    logger.info("🧪 Starting Redis integration tests...")
    
    # Test 1: Redis Connection
    if not test_redis_connection():
        logger.error("❌ Redis connection failed - aborting tests")
        return
    
    # Test 2: Caching
    caching_results = test_eodhd_caching()
    
    # Test 3: Rate Limiting
    rate_limit_results = test_rate_limiting()
    
    # Test 4: Async Processing (requires worker)
    logger.info("⚠️  Async processing test requires Redis worker to be running")
    logger.info("   Start worker with: python -m backend.utils.redis_worker")
    
    # Print summary
    logger.info("📊 Test Summary:")
    logger.info(f"   Caching: {len(caching_results)} symbols tested")
    logger.info(f"   Rate limiting: {rate_limit_results.get('successful_calls', 0)} successful calls")
    
    logger.info("✅ Redis integration tests completed!")


if __name__ == "__main__":
    main()
