"""Infrastructure bootstrap - DB, RAG, external services."""

from __future__ import annotations

import logging
import os
import time
from typing import Dict, Any

from core.db import init_db_engine
from startup.rag_bootstrap import initialize_rag_system as _rag_initialize

logger = logging.getLogger(__name__)


def reset_circuit_breakers() -> Dict[str, Any]:
    """Reset all circuit breakers on startup."""
    try:
        from config import get_config
        import redis
        
        config = get_config()
        redis_client = redis.from_url(config.redis.url)
        
        # Get all circuit breaker keys
        keys = redis_client.keys("circuit:*")
        reset_services = {}
        
        for key in keys:
            service_name = key.decode('utf-8').split(':', 1)[1]
            # Get current state before resetting
            state_data = redis_client.hgetall(key)
            if b'state' in state_data:
                old_state = state_data[b'state'].decode('utf-8')
                if old_state == "open":
                    # This service was in OPEN state
                    reset_services[service_name] = "was_open"
            
            # Reset circuit breaker
            redis_client.hset(key, mapping={
                'state': 'closed',
                'failures': 0,
                'last_success': int(time.time())
            })
        
        logger.info(f"Reset {len(keys)} circuit breakers on startup")
        if reset_services:
            logger.warning(f"Circuit breakers in OPEN state were reset: {', '.join(reset_services.keys())}")
        
        return {"success": True, "reset_count": len(keys), "previously_open": reset_services}
        
    except Exception as e:
        logger.error(f"Failed to reset circuit breakers: {e}")
        return {"success": False, "error": str(e)}


def verify_external_apis() -> Dict[str, Any]:
    """Verify connection to critical external APIs."""
    results = {}
    
    # Check EODHD API
    try:
        from config import get_config
        import requests
        
        config = get_config()
        api_key = config.external_services.eodhd_api_key
        
        if not api_key:
            results["eodhd"] = {
                "success": False,
                "error": "API key not configured"
            }
        else:
            # Try a simple API call
            url = "https://eodhistoricaldata.com/api/exchange-symbol-list/US"
            params = {
                "api_token": api_key,
                "fmt": "json",
                "limit": 5  # Just get a few symbols to test
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if isinstance(data, list) and len(data) > 0:
                results["eodhd"] = {
                    "success": True,
                    "data_sample": data[:2]  # First 2 items
                }
            else:
                results["eodhd"] = {
                    "success": False,
                    "error": "Unexpected response format"
                }
                
    except Exception as e:
        logger.error(f"Failed to verify EODHD API: {e}")
        results["eodhd"] = {
            "success": False,
            "error": str(e)
        }
    
    return results


def ensure_db_ready() -> bool:
    """Ensure database is ready for bootstrap."""
    try:
        # Initialize database engine
        init_db_engine()
        
        # Ensure DB is ready
        db_ready = True
        try:
            from core.db import get_db_session
            session = get_db_session()
            if session is None:
                db_ready = False
            else:
                try:
                    session.close()
                except Exception:
                    pass
        except Exception:
            db_ready = False
            
        # Try to create tables if needed
        if db_ready:
            try:
                from core.persistence import init_db_if_configured
                init_db_if_configured()
            except Exception as e:
                logger.warning(f"Table creation may have failed: {e}")
                # Continue anyway - tables might already exist
            
        return db_ready
    except Exception:
        logger.exception("Database initialization failed")
        return False


def initialize_rag_system() -> None:
    """Initialize RAG system if configured."""
    try:
        from config import get_config
        config = get_config()
        
        # Skip if disabled
        rag_enabled = os.getenv("STARTUP_RAG", "1").lower() in ("1", "true", "yes")
        if not rag_enabled:
            logger.info("RAG initialization skipped (STARTUP_RAG=0)")
            return
        
        # Skip if embeddings path not configured
        embeddings_path = config.rag.embeddings_path
        if not embeddings_path:
            logger.info("RAG initialization skipped (no embeddings path)")
            return
            
        # Initialize RAG
        _rag_initialize()
        logger.info("RAG system initialized")
        
    except Exception:
        logger.exception("RAG initialization failed")


def run_infrastructure_bootstrap() -> Dict[str, Any]:
    """Run all infrastructure bootstrap tasks."""
    results = {}
    
    # Database initialization
    db_ready = ensure_db_ready()
    results['database'] = db_ready
    
    # Exit early if database isn't ready
    if not db_ready:
        logger.error("Database initialization failed - skipping other infrastructure tasks")
        return results
    
    # Reset circuit breakers
    try:
        circuit_breaker_flag = os.getenv("STARTUP_RESET_CIRCUIT_BREAKERS", "1").lower()
        if circuit_breaker_flag in ("1", "true", "yes"):
            circuit_breaker_result = reset_circuit_breakers()
            results['circuit_breakers'] = circuit_breaker_result.get('success', False)
            
            if circuit_breaker_result.get('previously_open'):
                logger.warning(
                    "Circuit breakers were in OPEN state: %s", 
                    circuit_breaker_result.get('previously_open')
                )
        else:
            logger.info("Circuit breaker reset skipped (STARTUP_RESET_CIRCUIT_BREAKERS=%s)", circuit_breaker_flag)
            results['circuit_breakers'] = 'skipped'
    except Exception:
        logger.exception("Circuit breaker reset failed")
        results['circuit_breakers'] = 'error'
    
    # External APIs check
    try:
        apis_flag = os.getenv("STARTUP_CHECK_APIS", "1").lower()
        if apis_flag in ("1", "true", "yes"):
            apis_result = verify_external_apis()
            # Check if any API has success = False
            apis_success = all(api.get('success', False) for api in apis_result.values())
            results['external_apis'] = apis_success
            
            # Log warnings for failed APIs
            failed_apis = [name for name, data in apis_result.items() if not data.get('success', False)]
            if failed_apis:
                logger.warning("Some external APIs failed verification: %s", ", ".join(failed_apis))
        else:
            logger.info("External APIs check skipped (STARTUP_CHECK_APIS=%s)", apis_flag)
            results['external_apis'] = 'skipped'
    except Exception:
        logger.exception("External APIs check failed")
        results['external_apis'] = 'error'
    
    # RAG initialization
    try:
        rag_flag = os.getenv("STARTUP_RAG", "1").lower()
        if rag_flag in ("1", "true", "yes"):
            initialize_rag_system()
            results['rag'] = True
        else:
            logger.info("RAG initialization skipped (STARTUP_RAG=%s)", rag_flag)
            results['rag'] = 'skipped'
    except Exception:
        logger.exception("RAG initialization failed")
        results['rag'] = 'error'
    
    return results