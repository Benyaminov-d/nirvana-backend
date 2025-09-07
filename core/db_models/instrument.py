from datetime import datetime
from sqlalchemy import (  # type: ignore
    Column,
    Integer,
    String,
    Text,
    DateTime,
    JSON,
    UniqueConstraint,
    ForeignKey,
)
from sqlalchemy.orm import relationship  # type: ignore

from core.db import Base


class InstrumentAlias(Base):
    __tablename__ = "instrument_alias"

    id = Column(Integer, primary_key=True, autoincrement=True)
    instrument_id = Column(
        Integer, ForeignKey("symbols.id"), nullable=False, index=True
    )
    alias = Column(String(256), nullable=False, index=True)
    id_type = Column(String(32), nullable=False, default="name")

    __table_args__ = (
        UniqueConstraint("instrument_id", "alias", "id_type", name="uq_instrument_alias_key"),
    )
    instrument = relationship("Symbols", back_populates="aliases")


__all__ = ["Symbols", "InstrumentAlias"]


