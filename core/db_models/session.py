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


class Session(Base):
    __tablename__ = "session"

    id = Column(String(64), primary_key=True)
    created_at_utc = Column(DateTime, nullable=False, default=datetime.utcnow)
    last_seen_utc = Column(DateTime, nullable=True)
    country = Column(String(32), nullable=True)
    device_fingerprint_hash = Column(String(128), nullable=True)
    quotas = relationship(
        "SessionQuota", back_populates="session", cascade="all, delete-orphan"
    )
    preview_caches = relationship(
        "SessionPreviewCache",
        back_populates="session",
        cascade="all, delete-orphan",
    )


class SessionQuota(Base):
    __tablename__ = "session_quota"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), ForeignKey("session.id"), nullable=False, index=True)
    date_utc = Column(Date, nullable=False)
    free_ticker_cvar_lookups_used = Column(Integer, nullable=False, default=0)

    __table_args__ = (
        UniqueConstraint("session_id", "date_utc", name="uq_session_quota_day"),
    )
    session = relationship("Session", back_populates="quotas")


class SessionPreviewCache(Base):
    __tablename__ = "session_preview_cache"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), ForeignKey("session.id"), nullable=False, index=True)
    parameter_bucket_id = Column(
        Integer, ForeignKey("parameter_bucket.id"), nullable=False, index=True
    )
    snapshot_id = Column(
        Integer, ForeignKey("catalogue_snapshot.id"), nullable=False, index=True
    )
    top3_scores = Column(JSON, nullable=True)
    cached_at_utc = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint(
            "session_id",
            "parameter_bucket_id",
            "snapshot_id",
            name="uq_session_preview_cache_key",
        ),
    )
    session = relationship("Session", back_populates="preview_caches")
    parameter_bucket = relationship("ParameterBucket")
    snapshot = relationship("CatalogueSnapshot")


__all__ = ["Session", "SessionQuota", "SessionPreviewCache"]


