"""
User Repository for handling user-related database operations.
"""

from __future__ import annotations
from typing import List, Optional, Dict, Any
from datetime import datetime
from sqlalchemy.orm import Session

from repositories.base_repository import BaseRepository
from core.models import User, AuthAttempt, DigestSubscription
import logging

logger = logging.getLogger(__name__)


class UserRepository(BaseRepository[User]):
    """Repository for User operations."""
    
    def __init__(self):
        super().__init__(User)
    
    def get_by_email(self, email: str) -> Optional[User]:
        """Get user by email address."""
        def query_func(session: Session) -> Optional[User]:
            return (
                session.query(User)
                .filter(User.email == email)
                .first()
            )
        
        return self.execute_query(query_func)
    
    def get_by_username(self, username: str) -> Optional[User]:
        """Get user by username."""
        def query_func(session: Session) -> Optional[User]:
            return (
                session.query(User)
                .filter(User.username == username)
                .first()
            )
        
        return self.execute_query(query_func)
    
    def create_user(
        self,
        username: str,
        email: str,
        password_hash: str,
        **kwargs
    ) -> Optional[User]:
        """Create a new user."""
        def query_func(session: Session) -> Optional[User]:
            user = User(
                username=username,
                email=email,
                password_hash=password_hash,
                created_at=datetime.utcnow(),
                **kwargs
            )
            session.add(user)
            session.commit()
            session.refresh(user)
            return user
        
        return self.execute_query(query_func)
    
    def update_last_login(self, user_id: int) -> bool:
        """Update user's last login timestamp."""
        def query_func(session: Session) -> bool:
            user = session.query(User).filter(User.id == user_id).first()
            if user:
                user.last_login = datetime.utcnow()
                session.commit()
                return True
            return False
        
        return self.execute_query(query_func) or False
    
    def log_auth_attempt(
        self,
        email: str,
        success: bool,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> bool:
        """Log authentication attempt."""
        def query_func(session: Session) -> bool:
            attempt = AuthAttempt(
                email=email,
                success=success,
                ip_address=ip_address,
                user_agent=user_agent,
                attempted_at=datetime.utcnow()
            )
            session.add(attempt)
            session.commit()
            return True
        
        return self.execute_query(query_func) or False
    
    def get_recent_failed_attempts(self, email: str, minutes: int = 15) -> int:
        """Get count of recent failed auth attempts."""
        def query_func(session: Session) -> int:
            from datetime import timedelta
            cutoff = datetime.utcnow() - timedelta(minutes=minutes)
            
            return (
                session.query(AuthAttempt)
                .filter(
                    AuthAttempt.email == email,
                    AuthAttempt.success == False,
                    AuthAttempt.attempted_at >= cutoff
                )
                .count()
            )
        
        return self.execute_query(query_func) or 0
    
    def create_digest_subscription(
        self,
        email: str,
        frequency: str = "weekly",
        preferences: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Create digest subscription."""
        def query_func(session: Session) -> bool:
            subscription = DigestSubscription(
                email=email,
                frequency=frequency,
                preferences=preferences or {},
                is_active=True,
                created_at=datetime.utcnow()
            )
            session.add(subscription)
            session.commit()
            return True
        
        return self.execute_query(query_func) or False
    
    def get_active_subscriptions(self, frequency: Optional[str] = None) -> List[DigestSubscription]:
        """Get active digest subscriptions."""
        def query_func(session: Session) -> List[DigestSubscription]:
            query = (
                session.query(DigestSubscription)
                .filter(DigestSubscription.is_active == True)
            )
            
            if frequency:
                query = query.filter(DigestSubscription.frequency == frequency)
            
            return query.all()
        
        return self.execute_query(query_func) or []
    
    def unsubscribe_digest(self, email: str) -> bool:
        """Unsubscribe user from digest."""
        def query_func(session: Session) -> bool:
            subscription = (
                session.query(DigestSubscription)
                .filter(DigestSubscription.email == email)
                .first()
            )
            
            if subscription:
                subscription.is_active = False
                subscription.unsubscribed_at = datetime.utcnow()
                session.commit()
                return True
            return False
        
        return self.execute_query(query_func) or False
