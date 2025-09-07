from datetime import datetime
from sqlalchemy import (  # type: ignore
    Column,
    Integer,
    String,
    Text,
    Date,
    DateTime,
    Float,
    UniqueConstraint,
    JSON,
    ForeignKey,
)
from sqlalchemy.orm import relationship  # type: ignore

from core.db import Base


class PriceLast(Base):
    __tablename__ = "price_last"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(128), nullable=False, index=True)
    as_of_date = Column(Date, nullable=False, index=True)
    price_close = Column(Float, nullable=False)
    currency = Column(String(16), nullable=True)
    source = Column(String(64), nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("symbol", "as_of_date", name="uq_price_last_symbol_asof"),
    )


class CvarSnapshot(Base):
    __tablename__ = "cvar_snapshot"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(64), nullable=False, index=True)
    instrument_id = Column(Integer, ForeignKey("symbols.id"), nullable=True, index=True)
    as_of_date = Column(Date, nullable=False, index=True)
    start_date = Column(Date, nullable=True)
    years = Column(Integer, nullable=False, default=1)
    alpha_label = Column(Integer, nullable=False)
    alpha = Column(Float, nullable=True)
    cvar_nig = Column(Float, nullable=True)
    cvar_ghst = Column(Float, nullable=True)
    cvar_evar = Column(Float, nullable=True)
    return_as_of = Column(Float, nullable=True)
    return_annual = Column(Float, nullable=True)
    extra = Column(JSON, nullable=True)
    cached = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint(
            "symbol",
            "as_of_date",
            "alpha_label",
            "instrument_id",
            name="uq_cvar_snapshot_by_instrument",
        ),
    )
    instrument = relationship("Symbols", primaryjoin="Symbols.id==CvarSnapshot.instrument_id", viewonly=True)


class AnnualCvarViolation(Base):
    __tablename__ = "annual_cvar_violation"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(64), nullable=False, index=True)
    year = Column(Integer, nullable=False, index=True)
    as_of_date = Column(Date, nullable=False)
    next_year = Column(Integer, nullable=False)
    next_return = Column(Float, nullable=True)
    cvar99_nig = Column(Float, nullable=True)
    cvar99_ghst = Column(Float, nullable=True)
    cvar99_evar = Column(Float, nullable=True)
    cvar99_worst = Column(Float, nullable=True)
    violated99 = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("symbol", "year", name="uq_annual_violation_key"),
    )


class AnomalyReport(Base):
    __tablename__ = "anomaly_report"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(64), nullable=False, index=True)
    as_of_date = Column(Date, nullable=False, index=True)
    policy = Column(String(32), nullable=True)
    asset_class = Column(String(64), nullable=True)
    report = Column(JSON, nullable=True)
    summary = Column(JSON, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("symbol", "as_of_date", name="uq_anomaly_report_key"),
    )


class InsufficientDataEvent(Base):
    __tablename__ = "insufficient_data_event"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(64), nullable=False, index=True)
    as_of_date = Column(Date, nullable=True, index=True)
    code = Column(String(64), nullable=False, default="insufficient_data")
    error = Column(Text, nullable=True)
    source = Column(String(64), nullable=True)
    correlation_id = Column(String(128), nullable=True)
    raw = Column(JSON, nullable=True)
    diag = Column(JSON, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


__all__ = [
    "PriceLast",
    "CvarSnapshot",
    "AnnualCvarViolation",
    "AnomalyReport",
    "InsufficientDataEvent",
]


