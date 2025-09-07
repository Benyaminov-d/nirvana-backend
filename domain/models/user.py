"""
User domain model.

Pure business entity for user management
independent of authentication framework details.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from datetime import datetime
from uuid import UUID, uuid4

from shared.exceptions import DataValidationError


@dataclass
class User:
    """Domain model for application users."""
    
    id: UUID
    email: str
    is_active: bool = True
    is_premium: bool = False
    region: Optional[str] = None
    created_at: datetime = None
    
    def __post_init__(self):
        """Validate user data."""
        if self.id is None:
            object.__setattr__(self, 'id', uuid4())
        
        if self.created_at is None:
            object.__setattr__(self, 'created_at', datetime.utcnow())
        
        # Email validation
        if not self.email or '@' not in self.email:
            raise DataValidationError("Valid email address required")
    
    @classmethod
    def create(cls, email: str, **kwargs) -> User:
        """Factory method to create user."""
        return cls(
            id=uuid4(),
            email=email.lower().strip(),
            **kwargs
        )
