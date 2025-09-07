"""
RAG database models for pgvector.
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, JSON  # type: ignore
from sqlalchemy.sql import func  # type: ignore
from sqlalchemy.dialects.postgresql import UUID  # type: ignore
from sqlalchemy.ext.declarative import declarative_base  # type: ignore
import uuid

try:
    from pgvector.sqlalchemy import Vector  # type: ignore
    PGVECTOR_AVAILABLE = True
except ImportError:
    # Fallback type for when pgvector is not installed
    Vector = lambda dim: Text  # type: ignore
    PGVECTOR_AVAILABLE = False

Base = declarative_base()


class RagDocument(Base):
    """Document chunks with embeddings for RAG."""
    __tablename__ = 'rag_documents'
    
    id = Column(Integer, primary_key=True)
    chunk_id = Column(String(255), unique=True, nullable=False, index=True)
    text = Column(Text, nullable=False)
    
    # Vector embedding - 1536 dimensions for OpenAI text-embedding-ada-002
    embedding = Column(Vector(1536) if PGVECTOR_AVAILABLE else Text, nullable=True)
    
    source_file = Column(String(255), nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)
    
    # JSON metadata
    meta_data = Column(JSON, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    def __repr__(self):
        return f"<RagDocument(chunk_id='{self.chunk_id}', source='{self.source_file}')>"
    
    def to_dict(self):
        """Convert to dictionary for API responses."""
        return {
            'id': self.id,
            'chunk_id': self.chunk_id,
            'text': self.text,
            'source_file': self.source_file,
            'chunk_index': self.chunk_index,
            'metadata': self.meta_data,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
