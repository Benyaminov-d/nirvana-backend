"""
Base Repository class providing common database operations.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Type, TypeVar, Generic
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from core.db import get_db_session
from core.db import Base
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=Base)


class BaseRepository(Generic[T]):
    """Base repository class with common CRUD operations."""
    
    def __init__(self, model: Type[T]):
        self.model = model
    
    def get_session(self) -> Optional[Session]:
        """Get database session."""
        return get_db_session()
    
    def get_by_id(self, id: Any) -> Optional[T]:
        """Get entity by ID."""
        session = self.get_session()
        if session is None:
            return None
        
        try:
            return session.query(self.model).filter(self.model.id == id).first()
        except SQLAlchemyError as e:
            logger.error(f"Error getting {self.model.__name__} by id {id}: {e}")
            return None
        finally:
            session.close()
    
    def get_all(self, limit: Optional[int] = None) -> List[T]:
        """Get all entities with optional limit."""
        session = self.get_session()
        if session is None:
            return []
        
        try:
            query = session.query(self.model)
            if limit:
                query = query.limit(limit)
            return query.all()
        except SQLAlchemyError as e:
            logger.error(f"Error getting all {self.model.__name__}: {e}")
            return []
        finally:
            session.close()
    
    def create(self, **kwargs) -> Optional[T]:
        """Create new entity."""
        session = self.get_session()
        if session is None:
            return None
        
        try:
            entity = self.model(**kwargs)
            session.add(entity)
            session.commit()
            session.refresh(entity)
            return entity
        except SQLAlchemyError as e:
            logger.error(f"Error creating {self.model.__name__}: {e}")
            session.rollback()
            return None
        finally:
            session.close()
    
    def update(self, id: Any, **kwargs) -> Optional[T]:
        """Update entity by ID."""
        session = self.get_session()
        if session is None:
            return None
        
        try:
            entity = session.query(self.model).filter(self.model.id == id).first()
            if entity:
                for key, value in kwargs.items():
                    if hasattr(entity, key):
                        setattr(entity, key, value)
                session.commit()
                session.refresh(entity)
            return entity
        except SQLAlchemyError as e:
            logger.error(f"Error updating {self.model.__name__} with id {id}: {e}")
            session.rollback()
            return None
        finally:
            session.close()
    
    def delete(self, id: Any) -> bool:
        """Delete entity by ID."""
        session = self.get_session()
        if session is None:
            return False
        
        try:
            entity = session.query(self.model).filter(self.model.id == id).first()
            if entity:
                session.delete(entity)
                session.commit()
                return True
            return False
        except SQLAlchemyError as e:
            logger.error(f"Error deleting {self.model.__name__} with id {id}: {e}")
            session.rollback()
            return False
        finally:
            session.close()
    
    def count(self) -> int:
        """Count total entities."""
        session = self.get_session()
        if session is None:
            return 0
        
        try:
            return session.query(self.model).count()
        except SQLAlchemyError as e:
            logger.error(f"Error counting {self.model.__name__}: {e}")
            return 0
        finally:
            session.close()
    
    def execute_query(self, query_func, *args, **kwargs) -> Any:
        """Execute custom query with session management."""
        session = self.get_session()
        if session is None:
            return None
        
        try:
            return query_func(session, *args, **kwargs)
        except SQLAlchemyError as e:
            logger.error(f"Error executing query: {e}")
            session.rollback()
            return None
        finally:
            session.close()
