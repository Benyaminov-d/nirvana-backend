from datetime import datetime
from sqlalchemy import (  # type: ignore
    Column,
    Integer,
    String,
    DateTime,
    Float,
    ForeignKey,
)
from sqlalchemy.orm import relationship  # type: ignore

from core.db import Base


class PortfolioRequest(Base):
    __tablename__ = "portfolio_request"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(
        String(64), ForeignKey("session.id"), nullable=True, index=True
    )
    session = relationship("Session")
    parameter_bucket = relationship("ParameterBucket")
    positions = relationship(
        "PortfolioPosition",
        back_populates="portfolio",
        cascade="all, delete-orphan",
    )
    user_id = Column(Integer, nullable=True, index=True)
    jurisdiction = Column(String(32), nullable=True)
    parameter_bucket_id = Column(
        Integer, ForeignKey("parameter_bucket.id"), nullable=True, index=True
    )
    submitted_at_utc = Column(
        DateTime, nullable=False, default=datetime.utcnow
    )


class PortfolioPosition(Base):
    __tablename__ = "portfolio_position"

    id = Column(Integer, primary_key=True, autoincrement=True)
    portfolio_id = Column(
        Integer, ForeignKey("portfolio_request.id"), nullable=False, index=True
    )
    portfolio = relationship("PortfolioRequest", back_populates="positions")
    instrument = relationship("Symbols")
    instrument_id = Column(
        Integer, ForeignKey("symbols.id"), nullable=True, index=True
    )
    raw_symbol = Column(String(128), nullable=True)
    weight = Column(Float, nullable=True)
    quantity = Column(Float, nullable=True)
    price_as_of_utc = Column(DateTime, nullable=True)


class PortfolioMetrics(Base):
    __tablename__ = "portfolio_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    portfolio_id = Column(
        Integer, ForeignKey("portfolio_request.id"), nullable=False, index=True
    )
    portfolio = relationship("PortfolioRequest")
    matched_sur_model_id = Column(
        Integer, ForeignKey("sur_model.id"), nullable=True, index=True
    )
    proposed_sur_model_id = Column(
        Integer, ForeignKey("sur_model.id"), nullable=True, index=True
    )
    matched_sur_model = relationship(
        "SurModel", foreign_keys=[matched_sur_model_id]
    )
    proposed_sur_model = relationship(
        "SurModel", foreign_keys=[proposed_sur_model_id]
    )
    as_of_utc = Column(DateTime, nullable=False, index=True)
    method_version = Column(String(64), nullable=True)
    model_id = Column(String(64), nullable=True)
    portfolio_cvar_annualized = Column(Float, nullable=True)
    expected_return_annualized = Column(Float, nullable=True)
    volatility_annualized = Column(Float, nullable=True)
    compass_score = Column(Float, nullable=True)
    passes_standard = Column(Integer, nullable=False, default=0)


__all__ = ["PortfolioRequest", "PortfolioPosition", "PortfolioMetrics"]
