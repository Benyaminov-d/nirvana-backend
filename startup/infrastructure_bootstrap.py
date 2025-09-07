"""Infrastructure bootstrap - databases, external services, RAG."""

from __future__ import annotations

import logging
import os
from typing import Dict

from core.persistence import init_db_if_configured
from services.rag.pgvector_service import PgVectorRAG

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Ensure detailed bootstrap logs are visible


def ensure_db_ready() -> bool:
    """Initialize database if configured."""
    try:
        logger.info("Infrastructure bootstrap: initializing database...")
        result = init_db_if_configured()
        if result:
            logger.info("Database ready - continuing with other infrastructure")
        return result
    except Exception as e:
        logger.error("Database initialization failed: %s", str(e))
        return False


def initialize_rag_system() -> None:
    """Initialize RAG system on application startup."""
    try:
        logger.info("Initializing RAG system...")
        # Only initialize if enabled
        if os.getenv("RAG_ENABLED", "true").lower() != "true":
            logger.info(
                "RAG system disabled via RAG_ENABLED environment variable"
            )
            return
        
        logger.info("Initializing RAG system...")
        
        rag_service = PgVectorRAG()
        
        # Initialize database
        if not rag_service.initialize_database():
            logger.error("Failed to initialize RAG database")
            return
        
        # Load documents (but don't force reload on startup)
        if not rag_service.load_documents(force_reload=False):
            logger.warning(
                "No documents loaded in RAG system - "
                "run /api/rag/initialize to load documents"
            )
        else:
            stats = rag_service.get_stats()
            logger.info(
                f"RAG system initialized: {stats.get('total_documents', 0)} "
                f"documents loaded"
            )
        
    except ImportError:
        logger.info("RAG dependencies not available - RAG system disabled")
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        # Don't fail the entire app startup
        pass


def run_infrastructure_bootstrap() -> Dict[str, bool]:
    """Run all infrastructure bootstrap tasks."""
    results = {}
    
    # Database initialization
    print("  - Initializing database...")  # Force visibility
    logger.info("Infrastructure bootstrap: initializing database...")
    db_ready = ensure_db_ready()
    results['database'] = db_ready
    
    if db_ready:
        print("  - Database ready - continuing with other infrastructure")  # Force visibility
        logger.info("Database ready - continuing with other infrastructure")
        
        # RAG system initialization
        try:
            print("  - Initializing RAG system...")  # Force visibility
            initialize_rag_system()
            results['rag'] = True
            print("  - RAG system initialized successfully")  # Force visibility
        except Exception:
            print("  - RAG initialization failed")  # Force visibility
            logger.exception("RAG initialization failed")
            results['rag'] = False
    else:
        print("  - Database not ready - skipping dependent infrastructure")  # Force visibility
        logger.warning("Database not ready - skipping dependent infrastructure")
        results['rag'] = False
    
    return results
