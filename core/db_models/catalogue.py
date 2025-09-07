from datetime import datetime
from sqlalchemy import (  # type: ignore
    Column,
    Integer,
    String,
    Text,
    DateTime,
    Float,
    JSON,
    UniqueConstraint,
    ForeignKey,
)
from sqlalchemy.orm import relationship  # type: ignore

from core.db import Base


class ParameterBucket(Base):
    __tablename__ = "parameter_bucket"

    id = Column(Integer, primary_key=True, autoincrement=True)
    confidence_level = Column(Float, nullable=False)
    is_annualized = Column(Integer, nullable=False, default=1)
    horizon_years = Column(Integer, nullable=False, default=1)
    cvar_cutoff = Column(Float, nullable=True)
    universe_id = Column(String(128), nullable=True)
    jurisdiction = Column(String(32), nullable=True)
    method_version = Column(String(64), nullable=True)
    model_id = Column(String(64), nullable=True)

    __table_args__ = (
        UniqueConstraint(
            "confidence_level",
            "is_annualized",
            "horizon_years",
            "cvar_cutoff",
            "universe_id",
            "jurisdiction",
            "method_version",
            "model_id",
            name="uq_parameter_bucket_key",
        ),
    )
    catalogue_snapshots = relationship(
        "CatalogueSnapshot", back_populates="parameter_bucket"
    )


class CatalogueSnapshot(Base):
    __tablename__ = "catalogue_snapshot"

    id = Column(Integer, primary_key=True, autoincrement=True)
    parameter_bucket_id = Column(
        Integer, ForeignKey("parameter_bucket.id"), nullable=False, index=True
    )
    as_of_utc = Column(DateTime, nullable=False, index=True)
    method_version = Column(String(64), nullable=True)
    model_id = Column(String(64), nullable=True)
    universe_id = Column(String(128), nullable=True)
    notes = Column(Text, nullable=True)

    __table_args__ = (
        UniqueConstraint(
            "parameter_bucket_id",
            "as_of_utc",
            name="uq_catalogue_snapshot_key",
        ),
    )
    parameter_bucket = relationship(
        "ParameterBucket", back_populates="catalogue_snapshots"
    )
    entries = relationship(
        "CatalogueSnapshotEntry",
        back_populates="snapshot",
        cascade="all, delete-orphan",
    )


class CatalogueSnapshotEntry(Base):
    __tablename__ = "catalogue_snapshot_entry"

    id = Column(Integer, primary_key=True, autoincrement=True)
    snapshot_id = Column(
        Integer, ForeignKey("catalogue_snapshot.id"), nullable=False, index=True
    )
    rank = Column(Integer, nullable=False)
    instrument_id = Column(
        Integer, ForeignKey("symbols.id"), nullable=False, index=True
    )

    compass_score = Column(Float, nullable=True)
    passes_standard = Column(Integer, nullable=False, default=0)
    expected_return_annualized = Column(Float, nullable=True)
    volatility_annualized = Column(Float, nullable=True)
    cvar_95_annualized = Column(Float, nullable=True)
    cvar_99_annualized = Column(Float, nullable=True)
    alpha_star_annualized = Column(Float, nullable=True)
    fees = Column(Float, nullable=True)
    score_breakdown_json = Column(Text, nullable=True)

    __table_args__ = (
        UniqueConstraint("snapshot_id", "rank", name="uq_catalogue_entry_rank"),
    )
    snapshot = relationship("CatalogueSnapshot", back_populates="entries")
    instrument = relationship("Symbols")


class CompassAnchor(Base):
    __tablename__ = "compass_anchor"

    id = Column(Integer, primary_key=True, autoincrement=True)
    category = Column(String(128), nullable=False, index=True)
    version = Column(String(64), nullable=False, index=True)
    mu_low = Column(Float, nullable=False)
    mu_high = Column(Float, nullable=False)
    median_mu = Column(Float, nullable=True)
    p1 = Column(Float, nullable=True)
    p99 = Column(Float, nullable=True)
    metadata_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("category", "version", name="uq_compass_anchor_cat_ver"),
    )


__all__ = [
    "ParameterBucket",
    "CatalogueSnapshot",
    "CatalogueSnapshotEntry",
    "CompassAnchor",
]

