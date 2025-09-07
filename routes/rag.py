"""
Simple RAG API endpoints using pgvector.
"""

from typing import Optional, List, Dict, Any
import logging

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field
from starlette.requests import Request

from utils.auth import require_pub_or_basic as _require_pub_or_basic

try:
    from services.rag.pgvector_service import PgVectorRAG
    RAG_AVAILABLE = True
except ImportError:
    PgVectorRAG = None  # type: ignore
    RAG_AVAILABLE = False

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/rag", tags=["rag"])

# Global RAG service instance
_rag_service: Optional[PgVectorRAG] = None


def get_rag_service() -> PgVectorRAG:
    """Get or initialize the RAG service."""
    global _rag_service
    if not RAG_AVAILABLE:
        raise HTTPException(503, "RAG service not available - pgvector dependencies missing")
    
    if _rag_service is None:
        _rag_service = PgVectorRAG()
    return _rag_service


class SearchRequest(BaseModel):
    """Request model for document search."""
    query: str = Field(..., description="Search query")
    limit: int = Field(5, ge=1, le=20, description="Maximum results")
    source_file: Optional[str] = Field(None, description="Filter by source file")


class SearchResponse(BaseModel):
    """Response model for search results."""
    query: str
    results: List[Dict[str, Any]]
    total_found: int


@router.get("/health")
def rag_health_check(
    request: Request,
    _auth: None = Depends(_require_pub_or_basic)
) -> Dict[str, Any]:
    """Health check for RAG service."""
    try:
        if not RAG_AVAILABLE:
            return {
                "status": "unavailable",
                "service": "RAG",
                "error": "pgvector dependencies missing"
            }
        
        rag_service = get_rag_service()
        stats = rag_service.get_stats()
        
        return {
            "status": "healthy",
            "service": "RAG with pgvector",
            "stats": stats
        }
        
    except Exception as e:
        logger.error(f"RAG health check failed: {e}")
        return {
            "status": "unhealthy",
            "service": "RAG",
            "error": str(e)
        }


@router.get("/stats")
def get_rag_stats(
    request: Request,
    _auth: None = Depends(_require_pub_or_basic)
) -> Dict[str, Any]:
    """Get RAG database statistics."""
    try:
        rag_service = get_rag_service()
        stats = rag_service.get_stats()
        
        return {
            "success": True,
            "stats": stats
        }
        
    except Exception as e:
        logger.error(f"Error getting RAG stats: {e}")
        raise HTTPException(500, f"Failed to get stats: {str(e)}")


@router.post("/search", response_model=SearchResponse)
def search_documents(
    search_request: SearchRequest,
    request: Request,
    _auth: None = Depends(_require_pub_or_basic)
) -> SearchResponse:
    """Search for similar documents."""
    try:
        rag_service = get_rag_service()
        
        results = rag_service.search_similar(
            query=search_request.query,
            limit=search_request.limit,
            source_file_filter=search_request.source_file
        )
        
        return SearchResponse(
            query=search_request.query,
            results=results,
            total_found=len(results)
        )
        
    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        raise HTTPException(500, f"Search failed: {str(e)}")


@router.post("/initialize")
def initialize_rag(
    request: Request,
    force_reload: bool = Query(False, description="Force reload even if documents exist"),
    _auth: None = Depends(_require_pub_or_basic)
) -> Dict[str, Any]:
    """Initialize RAG database and load documents."""
    try:
        rag_service = get_rag_service()
        
        # Initialize database
        if not rag_service.initialize_database():
            raise HTTPException(500, "Failed to initialize database")
        
        # Load documents
        if not rag_service.load_documents(force_reload=force_reload):
            raise HTTPException(500, "Failed to load documents")
        
        # Get final stats
        stats = rag_service.get_stats()
        
        return {
            "success": True,
            "message": "RAG system initialized successfully",
            "stats": stats
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error initializing RAG: {e}")
        raise HTTPException(500, f"Initialization failed: {str(e)}")


@router.delete("/clear")
def clear_documents(
    request: Request,
    confirm: bool = Query(False, description="Must be True to confirm deletion"),
    _auth: None = Depends(_require_pub_or_basic)
) -> Dict[str, Any]:
    """Clear all documents from RAG database."""
    if not confirm:
        raise HTTPException(400, "This operation requires confirmation. Set confirm=true")
    
    try:
        rag_service = get_rag_service()
        
        if not rag_service.clear_documents():
            raise HTTPException(500, "Failed to clear documents")
        
        return {
            "success": True,
            "message": "All documents cleared from RAG database"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error clearing documents: {e}")
        raise HTTPException(500, f"Clear operation failed: {str(e)}")


@router.get("/context")
def get_context_for_query(
    request: Request,
    q: str = Query(..., description="Query to find context for"),
    max_chunks: int = Query(3, ge=1, le=10, description="Maximum context chunks"),
    _auth: None = Depends(_require_pub_or_basic)
) -> Dict[str, Any]:
    """Generate context for a query (for testing RAG)."""
    try:
        rag_service = get_rag_service()
        
        context_prompt, used_docs = rag_service.get_context_for_query(
            query=q,
            max_chunks=max_chunks
        )
        
        return {
            "query": q,
            "context": context_prompt,
            "chunks_used": len(used_docs),
            "source_files": list(set(doc['source_file'] for doc in used_docs))
        }
        
    except Exception as e:
        logger.error(f"Error generating context: {e}")
        raise HTTPException(500, f"Context generation failed: {str(e)}")
