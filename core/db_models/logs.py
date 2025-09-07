from datetime import datetime
from sqlalchemy import (  # type: ignore
    Column,
    Integer,
    String,
    DateTime,
    JSON,
    ForeignKey,
)
from sqlalchemy.orm import relationship  # type: ignore

from core.db import Base


class TickerLookupLog(Base):
    __tablename__ = "ticker_lookup_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp_utc = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    session = relationship("Session")
    snapshot = relationship("CatalogueSnapshot")
    instrument = relationship("Symbols")
    session_id = Column(String(64), ForeignKey("session.id"), nullable=True, index=True)
    user_id = Column(Integer, nullable=True, index=True)
    symbol_input = Column(String(256), nullable=True)
    resolved_instrument_id = Column(Integer, ForeignKey("symbols.id"), nullable=True, index=True)
    parameters_json = Column(JSON, nullable=True)
    snapshot_id = Column(Integer, ForeignKey("catalogue_snapshot.id"), nullable=True, index=True)


class ProximitySearchLog(Base):
    __tablename__ = "proximity_search_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp_utc = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    session = relationship("Session")
    parameter_bucket = relationship("ParameterBucket")
    snapshot = relationship("CatalogueSnapshot")
    session_id = Column(String(64), ForeignKey("session.id"), nullable=True, index=True)
    parameter_bucket_id = Column(Integer, ForeignKey("parameter_bucket.id"), nullable=True, index=True)
    snapshot_id = Column(Integer, ForeignKey("catalogue_snapshot.id"), nullable=True, index=True)
    top_n_returned = Column(Integer, nullable=True)


__all__ = ["TickerLookupLog", "ProximitySearchLog"]


