"""
Simple RAG service using pgvector for document search.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from sqlalchemy.orm import Session  # type: ignore
from sqlalchemy import text  # type: ignore

from services.rag.document_processor import DocumentProcessor, DocumentChunk
import sys
from pathlib import Path
# Add backend to Python path for imports
backend_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_path))

from core.db import get_db_session
from core.db_models.rag import RagDocument

try:
    from openai import OpenAI  # type: ignore
    OPENAI_AVAILABLE = True
except ImportError:
    OpenAI = None  # type: ignore
    OPENAI_AVAILABLE = False

logger = logging.getLogger(__name__)


class PgVectorRAG:
    """Simple RAG service using PostgreSQL + pgvector."""
    
    def __init__(
        self,
        documents_directory: Optional[str] = None,
        embedding_model: str = "text-embedding-ada-002",
        chunk_size: int = 800,
        chunk_overlap: int = 150
    ):
        # Set default documents directory
        if documents_directory is None:
            backend_path = Path(__file__).parent.parent.parent
            documents_directory = str(backend_path / "rag")
        
        self.documents_directory = documents_directory
        self.embedding_model_name = embedding_model
        
        # Initialize document processor
        self.processor = DocumentProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Initialize OpenAI client for embeddings
        self.openai_client = None
        if OPENAI_AVAILABLE:
            try:
                import os
                api_key = os.getenv("OPENAI_API_KEY")
                if api_key:
                    self.openai_client = OpenAI(api_key=api_key)
                    logger.info(f"OpenAI client initialized for embeddings: {embedding_model}")
                else:
                    logger.warning("OPENAI_API_KEY not found, embeddings disabled")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
                self.openai_client = None
        else:
            logger.warning("OpenAI not available, embeddings disabled")
    
    def _ensure_pgvector_extension(self, session: Session) -> bool:
        """Ensure pgvector extension is installed."""
        try:
            # Check if extension exists
            result = session.execute(text(
                "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')"
            )).scalar()
            
            if not result:
                logger.info("Installing pgvector extension...")
                session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                session.commit()
                logger.info("pgvector extension installed")
            
            return True
        except Exception as e:
            logger.error(f"Failed to install pgvector extension: {e}")
            return False
    
    def initialize_database(self) -> bool:
        """Initialize database tables and extensions."""
        try:
            session = get_db_session()
            if session is None:
                logger.error("No database session available")
                return False
            
            try:
                # Ensure pgvector extension
                if not self._ensure_pgvector_extension(session):
                    return False
                
                # Create tables
                from core.db_models.rag import Base
                from core.db import engine
                Base.metadata.create_all(engine)
                
                # Create index for vector similarity search
                try:
                    session.execute(text("""
                        CREATE INDEX IF NOT EXISTS rag_documents_embedding_idx 
                        ON rag_documents USING hnsw (embedding vector_cosine_ops)
                    """))
                    session.commit()
                    logger.info("Created vector similarity index")
                except Exception as e:
                    logger.warning(f"Could not create HNSW index, using default: {e}")
                    # Fallback to basic index
                    try:
                        session.execute(text("""
                            CREATE INDEX IF NOT EXISTS rag_documents_embedding_basic_idx 
                            ON rag_documents (embedding)
                        """))
                        session.commit()
                    except Exception as e2:
                        logger.warning(f"Could not create basic index either: {e2}")
                
                logger.info("Database initialized successfully")
                return True
                
            finally:
                session.close()
                
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            return False
    
    def load_documents(self, force_reload: bool = False) -> bool:
        """Load documents from the rag directory into database."""
        try:
            session = get_db_session()
            if session is None:
                return False
            
            try:
                # Check if we already have documents
                if not force_reload:
                    count = session.query(RagDocument).count()
                    if count > 0:
                        logger.info(f"Database already has {count} document chunks")
                        return True
                
                # Clear existing documents if force reload
                if force_reload:
                    session.query(RagDocument).delete()
                    session.commit()
                    logger.info("Cleared existing documents for reload")
                
                # Process documents
                logger.info(f"Loading documents from: {self.documents_directory}")
                chunks = self.processor.load_documents_from_directory(self.documents_directory)
                
                if not chunks:
                    logger.warning("No document chunks found to process")
                    return False
                
                logger.info(f"Processing {len(chunks)} document chunks")
                
                # Generate embeddings via OpenAI
                embeddings = None
                if self.openai_client is not None:
                    texts = [chunk.text for chunk in chunks]
                    logger.info("Generating embeddings via OpenAI...")
                    embeddings = self._generate_openai_embeddings(texts)
                    logger.info("OpenAI embeddings generated successfully")
                
                # Insert into database
                for i, chunk in enumerate(chunks):
                    embedding_vector = embeddings[i] if embeddings is not None else None
                    
                    doc = RagDocument(
                        chunk_id=chunk.id,
                        text=chunk.text,
                        embedding=embedding_vector,
                        source_file=chunk.source_file,
                        chunk_index=chunk.chunk_index,
                        meta_data=chunk.metadata
                    )
                    session.add(doc)
                
                session.commit()
                logger.info(f"Successfully loaded {len(chunks)} chunks into database")
                return True
                
            finally:
                session.close()
                
        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            return False
    
    def _generate_openai_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI API."""
        try:
            if not self.openai_client:
                return []
                
            # OpenAI has rate limits, process in batches
            batch_size = 100  # Conservative batch size
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                logger.info(f"Processing embedding batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
                
                response = self.openai_client.embeddings.create(
                    model=self.embedding_model_name,
                    input=batch
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
            
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Error generating OpenAI embeddings: {e}")
            return []

    def search_similar(
        self, 
        query: str, 
        limit: int = 5,
        similarity_threshold: float = 0.3,
        source_file_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar document chunks."""
        try:
            if self.openai_client is None:
                logger.warning("No OpenAI client available for search")
                return []
            
            session = get_db_session()
            if session is None:
                return []
            
            try:
                # Generate query embedding via OpenAI
                query_embeddings = self._generate_openai_embeddings([query])
                if not query_embeddings:
                    logger.warning("Failed to generate query embedding")
                    return []
                query_embedding = query_embeddings[0]
                
                # Convert embedding to string format for PostgreSQL
                embedding_str = str(query_embedding)
                
                # Build SQL query with direct embedding substitution
                sql_query_parts = [
                    f"SELECT *, (embedding <=> '{embedding_str}'::vector) as distance",
                    "FROM rag_documents",
                    "WHERE embedding IS NOT NULL"
                ]
                
                # Add source file filter
                if source_file_filter:
                    sql_query_parts.append("AND source_file = %(source_file)s")
                
                # Add similarity threshold (distance threshold)  
                distance_threshold = 1.0 - similarity_threshold
                sql_query_parts.append(f"AND (embedding <=> '{embedding_str}'::vector) < {distance_threshold}")
                
                # Order and limit
                sql_query_parts.extend([
                    f"ORDER BY embedding <=> '{embedding_str}'::vector",
                    f"LIMIT {limit}"
                ])
                
                sql_query = " ".join(sql_query_parts)
                
                # Prepare parameters (only for source_file filter if needed)
                params = {}
                if source_file_filter:
                    params['source_file'] = source_file_filter
                
                # Execute query  
                logger.debug(f"Executing vector similarity query with {len(query_embedding)} dimensions")
                result = session.execute(text(sql_query), params)
                rows = result.fetchall()
                
                logger.info(f"Vector search returned {len(rows)} similar documents")
                
                # Format results
                documents = []
                for row in rows:
                    similarity = 1.0 - float(row.distance)  # Convert distance to similarity
                    doc_dict = {
                        'id': row.id,
                        'chunk_id': row.chunk_id,
                        'text': row.text,
                        'source_file': row.source_file,
                        'chunk_index': row.chunk_index,
                        'metadata': row.meta_data,
                        'similarity': similarity,
                        'distance': float(row.distance)
                    }
                    documents.append(doc_dict)
                
                logger.info(f"Found {len(documents)} similar documents for query")
                return documents
                
            finally:
                session.close()
                
        except Exception as e:
            logger.error(f"Error searching similar documents: {e}")
            return []
    
    def get_context_for_query(
        self, 
        query: str, 
        max_chunks: int = 3,
        max_context_length: int = 2000
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Generate context string from relevant documents."""
        # Search for relevant chunks
        relevant_docs = self.search_similar(
            query=query,
            limit=max_chunks * 2,  # Get more to select from
            similarity_threshold=0.1  # Low threshold for broader search
        )
        
        if not relevant_docs:
            return "", []
        
        # Build context from top chunks
        context_parts = []
        used_docs = []
        current_length = 0
        
        for doc in relevant_docs[:max_chunks]:
            source_info = f"From {doc['source_file']}"
            chunk_text = doc['text'].strip()
            
            formatted_chunk = f"{source_info}:\n{chunk_text}\n"
            
            # Check if adding this chunk would exceed length limit
            if current_length + len(formatted_chunk) > max_context_length:
                # Try to fit a truncated version if there's reasonable space
                remaining_space = max_context_length - current_length
                if remaining_space > 200:
                    truncated_text = chunk_text[:remaining_space-100] + "..."
                    formatted_chunk = f"{source_info}:\n{truncated_text}\n"
                    context_parts.append(formatted_chunk)
                    used_docs.append(doc)
                break
            
            context_parts.append(formatted_chunk)
            used_docs.append(doc)
            current_length += len(formatted_chunk)
        
        context_prompt = ""
        if context_parts:
            context_prompt = (
                "Relevant information from Nirvana documentation:\n\n" +
                "\n".join(context_parts).strip()
            )
        
        logger.info(f"Generated context with {len(used_docs)} chunks ({current_length} chars)")
        return context_prompt, used_docs
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the RAG database."""
        try:
            session = get_db_session()
            if session is None:
                return {"error": "No database connection"}
            
            try:
                # Count total documents
                total_docs = session.query(RagDocument).count()
                
                # Count by source file
                source_counts = session.execute(text("""
                    SELECT source_file, COUNT(*) as count 
                    FROM rag_documents 
                    GROUP BY source_file
                    ORDER BY count DESC
                """)).fetchall()
                
                # Count documents with embeddings
                embedded_count = session.execute(text("""
                    SELECT COUNT(*) FROM rag_documents WHERE embedding IS NOT NULL
                """)).scalar()
                
                return {
                    "total_documents": total_docs,
                    "documents_with_embeddings": embedded_count,
                    "embedding_model": self.embedding_model_name,
                    "source_files": {row.source_file: row.count for row in source_counts},
                    "embedding_available": self.openai_client is not None
                }
                
            finally:
                session.close()
                
        except Exception as e:
            logger.error(f"Error getting RAG stats: {e}")
            return {"error": str(e)}
    
    def clear_documents(self) -> bool:
        """Clear all documents from the database."""
        try:
            session = get_db_session()
            if session is None:
                return False
            
            try:
                deleted_count = session.query(RagDocument).delete()
                session.commit()
                logger.info(f"Cleared {deleted_count} documents from database")
                return True
                
            finally:
                session.close()
                
        except Exception as e:
            logger.error(f"Error clearing documents: {e}")
            return False
