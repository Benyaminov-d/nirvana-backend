from datetime import datetime
from sqlalchemy import (  # type: ignore
    Column,
    Integer,
    String,
    DateTime,
    UniqueConstraint,
)

from core.db import Base


class Exchange(Base):
    """EODHD Exchange information model.

    Stores exchange metadata from EODHD API exchanges-list endpoint.
    """
    __tablename__ = "exchange"

    id = Column(Integer, primary_key=True, autoincrement=True)
    code = Column(String(32), nullable=False, unique=True, index=True)
    name = Column(String(256), nullable=True)
    operating_mic = Column(String(128), nullable=True)
    country = Column(String(128), nullable=True)
    currency = Column(String(16), nullable=True)
    country_iso2 = Column(String(8), nullable=True)
    country_iso3 = Column(String(8), nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow
    )

    __table_args__ = (
        UniqueConstraint("code", name="uq_exchange_code"),
    )

    def __repr__(self) -> str:
        return (
            f"<Exchange(code='{self.code}', "
            f"name='{self.name}', country='{self.country}')>"
        )
