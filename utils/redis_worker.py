"""
Redis Queue worker for processing EODHD API jobs.

This script starts a Redis worker that processes EODHD API calls asynchronously.
Use this to handle bulk operations without blocking the main application.

Usage:
    python -m backend.utils.redis_worker
"""

import os
import sys
import logging
import signal
from pathlib import Path

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_config
from services.infrastructure.redis_eodhd_client import start_redis_worker


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GracefulWorkerShutdown:
    """Handle graceful shutdown of Redis worker."""
    
    def __init__(self):
        self.shutdown = False
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)
    
    def exit_gracefully(self, signum, frame):
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.shutdown = True


def main():
    """Start Redis worker."""
    logger.info("🚀 Starting Redis EODHD worker...")
    
    try:
        config = get_config()
        
        logger.info(f"Redis URL: {config.redis.url}")
        logger.info(f"Queue: {config.redis.queue_name}")
        logger.info(f"Worker timeout: {config.redis.worker_timeout}s")
        
        # Setup graceful shutdown
        shutdown_handler = GracefulWorkerShutdown()
        
        # Start worker
        start_redis_worker(
            redis_url=config.redis.url,
            queue_name=config.redis.queue_name
        )
        
    except KeyboardInterrupt:
        logger.info("Worker stopped by user")
    except Exception as e:
        logger.error(f"Worker failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
