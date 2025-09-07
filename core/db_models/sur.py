from datetime import datetime
from sqlalchemy import (  # type: ignore
    Column,
    Integer,
    String,
    DateTime,
    Float,
    UniqueConstraint,
    ForeignKey,
)
from sqlalchemy.orm import relationship  # type: ignore

from core.db import Base


class SurModel(Base):
    __tablename__ = "sur_model"

    id = Column(Integer, primary_key=True, autoincrement=True)
    parameter_bucket_id = Column(
        Integer, ForeignKey("parameter_bucket.id"), nullable=True, index=True
    )
    snapshot_id = Column(
        Integer, ForeignKey("catalogue_snapshot.id"), nullable=True, index=True
    )
    type = Column(String(32), nullable=False)
    as_of_utc = Column(DateTime, nullable=False, index=True)
    method_version = Column(String(64), nullable=True)
    model_id = Column(String(64), nullable=True)
    portfolio_cvar_annualized = Column(Float, nullable=True)
    expected_return_annualized = Column(Float, nullable=True)
    volatility_annualized = Column(Float, nullable=True)
    compass_score = Column(Float, nullable=True)
    passes_standard = Column(Integer, nullable=False, default=0)

    __table_args__ = (
        UniqueConstraint("snapshot_id", "type", name="uq_sur_model_snapshot_type"),
    )
    parameter_bucket = relationship("ParameterBucket")
    snapshot = relationship("CatalogueSnapshot")
    constituents = relationship(
        "SurModelConstituent",
        back_populates="model",
        cascade="all, delete-orphan",
    )


class SurModelConstituent(Base):
    __tablename__ = "sur_model_constituent"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(Integer, ForeignKey("sur_model.id"), nullable=False, index=True)
    instrument_id = Column(
        Integer, ForeignKey("symbols.id"), nullable=False, index=True
    )
    weight = Column(Float, nullable=False)

    __table_args__ = (
        UniqueConstraint("model_id", "instrument_id", name="uq_sur_model_constituent_unique"),
    )
    model = relationship("SurModel", back_populates="constituents")
    instrument = relationship("Symbols")


__all__ = ["SurModel", "SurModelConstituent"]


