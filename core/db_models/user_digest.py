from datetime import datetime
from sqlalchemy import (  # type: ignore
    Column,
    Integer,
    String,
    Date,
    DateTime,
    JSON,
    UniqueConstraint,
    ForeignKey,
)
from sqlalchemy.orm import relationship  # type: ignore

from core.db import Base


class User(Base):
    __tablename__ = "user"

    id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String(320), nullable=False, unique=True, index=True)
    is_paid = Column(Integer, nullable=False, default=0)
    region = Column(String(32), nullable=True)
    consents_json = Column(JSON, nullable=True)
    created_at_utc = Column(DateTime, nullable=False, default=datetime.utcnow)
    password_hash = Column(String(128), nullable=True)
    email_verified = Column(Integer, nullable=False, default=0)
    verification_token = Column(String(64), nullable=True, index=True)
    last_login_utc = Column(DateTime, nullable=True)
    password_reset_token = Column(String(64), nullable=True, index=True)
    password_reset_expires_utc = Column(DateTime, nullable=True)
    digest_subscriptions = relationship(
        "DigestSubscription",
        back_populates="user",
        cascade="all, delete-orphan",
    )


class AuthAttempt(Base):
    __tablename__ = "auth_attempt"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp_utc = Column(DateTime, nullable=False, default=datetime.utcnow)
    email = Column(String(320), nullable=True, index=True)
    ip = Column(String(64), nullable=True, index=True)
    purpose = Column(String(32), nullable=False, default="signin")
    success = Column(Integer, nullable=False, default=0)


class DigestSubscription(Base):
    __tablename__ = "digest_subscription"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("user.id"), nullable=False, index=True)
    parameter_bucket_id = Column(
        Integer, ForeignKey("parameter_bucket.id"), nullable=False, index=True
    )
    active = Column(Integer, nullable=False, default=1)
    created_at_utc = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("user_id", "parameter_bucket_id", name="uq_digest_subscription_unique"),
    )
    user = relationship("User", back_populates="digest_subscriptions")
    parameter_bucket = relationship("ParameterBucket")


class DigestIssue(Base):
    __tablename__ = "digest_issue"

    id = Column(Integer, primary_key=True, autoincrement=True)
    parameter_bucket_id = Column(
        Integer, ForeignKey("parameter_bucket.id"), nullable=False, index=True
    )
    snapshot_id = Column(Integer, ForeignKey("catalogue_snapshot.id"), nullable=True, index=True)
    publish_date_utc = Column(Date, nullable=False, index=True)
    content_hash = Column(String(128), nullable=True)
    deliveries = relationship("DigestDelivery", back_populates="issue", cascade="all, delete-orphan")


class DigestDelivery(Base):
    __tablename__ = "digest_delivery"

    id = Column(Integer, primary_key=True, autoincrement=True)
    issue_id = Column(Integer, ForeignKey("digest_issue.id"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("user.id"), nullable=False, index=True)
    status = Column(String(32), nullable=True)
    sent_at_utc = Column(DateTime, nullable=True)
    unsubscribe_token = Column(String(64), nullable=True, index=True)
    issue = relationship("DigestIssue", back_populates="deliveries")
    user = relationship("User")


__all__ = [
    "User",
    "DigestSubscription",
    "DigestIssue",
    "DigestDelivery",
    "AuthAttempt",
]


