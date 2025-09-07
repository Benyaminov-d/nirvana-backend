"""
Document processing service for chunking and preprocessing documents.
"""

import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import hashlib
import logging

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """Represents a chunk of document text with metadata."""
    id: str
    text: str
    source_file: str
    chunk_index: int
    metadata: Dict[str, Any]


class DocumentProcessor:
    """Handles document chunking and preprocessing for RAG."""
    
    def __init__(
        self, 
        chunk_size: int = 1000, 
        chunk_overlap: int = 200,
        min_chunk_size: int = 50
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
    
    def load_documents_from_directory(self, directory: str) -> List[DocumentChunk]:
        """Load and process all documents from a directory."""
        chunks = []
        directory_path = Path(directory)
        
        if not directory_path.exists():
            logger.error(f"Directory does not exist: {directory}")
            return chunks
        
        # Process markdown files
        for md_file in directory_path.glob("*.md"):
            try:
                file_chunks = self.process_markdown_file(md_file)
                chunks.extend(file_chunks)
                logger.info(f"Processed {len(file_chunks)} chunks from {md_file.name}")
            except Exception as e:
                logger.error(f"Error processing {md_file}: {e}")
        
        return chunks
    
    def process_markdown_file(self, file_path: Path) -> List[DocumentChunk]:
        """Process a markdown file and split into chunks."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Could not read file {file_path}: {e}")
            return []
        
        # Extract title from first line if it's a header
        title = self._extract_title(content)
        
        # Clean and preprocess text
        cleaned_content = self._clean_text(content)
        
        # Split into chunks
        text_chunks = self._split_text(cleaned_content)
        
        # Create DocumentChunk objects
        chunks = []
        for i, chunk_text in enumerate(text_chunks):
            if len(chunk_text.strip()) < self.min_chunk_size:
                continue
                
            chunk_id = self._generate_chunk_id(file_path.name, i, chunk_text)
            
            chunk = DocumentChunk(
                id=chunk_id,
                text=chunk_text,
                source_file=str(file_path.name),
                chunk_index=i,
                metadata={
                    "title": title,
                    "file_path": str(file_path),
                    "file_type": "markdown",
                    "chunk_size": len(chunk_text),
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def _extract_title(self, content: str) -> str:
        """Extract title from document content."""
        lines = content.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('# '):
                return line[2:].strip()
            elif line.startswith('**') and line.endswith('**'):
                return line[2:-2].strip()
        
        return "Document"
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        
        # Clean up markdown formatting while preserving structure
        # Remove reference-style links
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        
        # Clean up bold/italic markers but keep the text
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*([^*]+)\*', r'\1', text)      # Italic
        
        # Remove email links markup
        text = re.sub(r'\[([^\]]+@[^\]]+)\]\([^)]+\)', r'\1', text)
        
        # Clean up excessive periods and special characters
        text = re.sub(r'\.{3,}', '...', text)
        
        # Normalize spaces
        text = re.sub(r'[ \t]+', ' ', text)
        
        return text.strip()
    
    def _split_text(self, text: str) -> List[str]:
        """Split text into chunks with overlap."""
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at sentence boundaries
            if end < len(text):
                # Look for sentence endings within the last 200 characters
                last_chunk = text[max(0, end - 200):end]
                sentence_break = max(
                    last_chunk.rfind('. '),
                    last_chunk.rfind('.\n'),
                    last_chunk.rfind('! '),
                    last_chunk.rfind('?\n'),
                    last_chunk.rfind('\n\n')
                )
                
                if sentence_break != -1:
                    # Adjust end to the sentence break
                    end = max(0, end - 200) + sentence_break + 1
            
            chunk = text[start:end].strip()
            
            if len(chunk) >= self.min_chunk_size:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = max(start + 1, end - self.chunk_overlap)
            
            # Break if we're not making progress
            if start >= end:
                break
        
        return chunks
    
    def _generate_chunk_id(self, filename: str, chunk_index: int, text: str) -> str:
        """Generate a unique ID for a document chunk."""
        content_hash = hashlib.md5(text.encode('utf-8')).hexdigest()[:8]
        return f"{filename}_{chunk_index}_{content_hash}"
    
    def get_chunk_by_id(self, chunks: List[DocumentChunk], chunk_id: str) -> Optional[DocumentChunk]:
        """Retrieve a chunk by its ID."""
        for chunk in chunks:
            if chunk.id == chunk_id:
                return chunk
        return None
    
    def filter_chunks_by_source(self, chunks: List[DocumentChunk], source_file: str) -> List[DocumentChunk]:
        """Filter chunks by source file."""
        return [chunk for chunk in chunks if chunk.source_file == source_file]
