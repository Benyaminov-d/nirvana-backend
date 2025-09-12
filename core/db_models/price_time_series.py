from datetime import datetime, date
from sqlalchemy import (  # type: ignore
    Column,
    Integer,
    Date,
    DateTime,
    String,
    BigInteger,
    Numeric,
    ForeignKey,
    Index,
)
from sqlalchemy.orm import relationship  # type: ignore

from core.db import Base


class PriceTimeSeries(Base):
    __tablename__ = "price_time_series"

    id = Column(Integer, primary_key=True, autoincrement=True)

    symbol_id = Column(Integer, ForeignKey("symbols.id"), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    price = Column(Numeric(20, 6), nullable=False)
    volume = Column(BigInteger, nullable=True)
    source_type = Column(String(20), nullable=False)  # 'raw' | 'computed'
    version_id = Column(String(128), nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationships
    instrument = relationship("Symbols", backref="price_time_series")

    __table_args__ = (
        Index("ix_pts_symbol_date", "symbol_id", "date"),
        Index("uq_pts_symbol_date_version", "symbol_id", "date", "version_id", unique=True),
    )

    def __repr__(self) -> str:
        return (
            f"<PriceTimeSeries(symbol_id={self.symbol_id}, date={self.date}, "
            f"price={self.price})>"
        )


