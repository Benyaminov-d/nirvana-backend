"""
RAG system bootstrap - initialize on startup.
"""

import logging
import os

logger = logging.getLogger(__name__)


def initialize_rag_system():
    """Initialize RAG system on application startup."""
    try:
        # Only initialize if enabled
        if os.getenv("RAG_ENABLED", "true").lower() != "true":
            logger.info("RAG system disabled via RAG_ENABLED environment variable")
            return
        
        from services.rag.pgvector_service import PgVectorRAG
        
        logger.info("Initializing RAG system...")
        
        rag_service = PgVectorRAG()
        
        # Initialize database
        if not rag_service.initialize_database():
            logger.error("Failed to initialize RAG database")
            return
        
        # Load documents (but don't force reload on startup)
        if not rag_service.load_documents(force_reload=False):
            logger.warning("No documents loaded in RAG system - run /api/rag/initialize to load documents")
        else:
            stats = rag_service.get_stats()
            logger.info(f"RAG system initialized: {stats.get('total_documents', 0)} documents loaded")
        
    except ImportError:
        logger.info("RAG dependencies not available - RAG system disabled")
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        # Don't fail the entire app startup
        pass


if __name__ == "__main__":
    initialize_rag_system()
