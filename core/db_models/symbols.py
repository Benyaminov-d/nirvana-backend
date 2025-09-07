"""
Symbols model - Financial instrument symbols and metadata.

This is the renamed and enhanced version of the former PriceSeries model,
with added support for categorical flags based on folder structure.
"""

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


class Symbols(Base):
    """
    Financial instrument symbols and metadata.
    
    Supports categorical flags for various indices and classifications.
    Previously known as PriceSeries.
    """
    __tablename__ = "symbols"

    # Core identification fields
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(128), nullable=False, index=True)
    name = Column(Text, nullable=True)
    alternative_names = Column(JSON, nullable=True)
    country = Column(String(64), nullable=True)
    exchange = Column(String(64), nullable=True)
    currency = Column(String(32), nullable=True)
    instrument_type = Column("type", String(64), nullable=True)
    isin = Column(String(64), nullable=True)
    
    # Timestamp fields
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(
        DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow
    )
    
    # Data quality fields
    insufficient_history = Column(Integer, nullable=True, default=None)  # 1=insufficient history, 0=sufficient history, NULL=not checked
    valid = Column(Integer, nullable=True, default=None)  # 1=valid (can be used), 0=invalid (any reason), NULL=not checked
    dropped_points_recent = Column(Integer, nullable=True, default=None)
    has_dropped_points_recent = Column(Integer, nullable=False, default=0)
    
    # ==========================================
    # CATEGORICAL FLAGS - Based on folder names
    # ==========================================
    
    # Five Stars (Morningstar 5-star rated)
    five_stars = Column(Integer, nullable=False, default=0)
    
    # US Index constituents
    sp_500 = Column(Integer, nullable=False, default=0)  # S&P 500
    nasdaq_100 = Column(Integer, nullable=False, default=0)  # NASDAQ 100
    dow_jones_industrial_average = Column(Integer, nullable=False, default=0)  # Dow Jones
    russell_1000 = Column(Integer, nullable=False, default=0)  # Russell 1000
    sp_midcap_400 = Column(Integer, nullable=False, default=0)  # S&P MidCap 400
    sp_smallcap_600 = Column(Integer, nullable=False, default=0)  # S&P SmallCap 600
    
    # UK Index constituents
    ftse_100 = Column(Integer, nullable=False, default=0)  # FTSE 100
    ftse_350 = Column(Integer, nullable=False, default=0)  # FTSE 350
    
    # Canada Index constituents
    tsx_60 = Column(Integer, nullable=False, default=0)  # TSX 60
    sp_60_constituents = Column(
        Integer, nullable=False, default=0
    )  # S&P/TSX 60 constituents
    sp_composite = Column(
        Integer, nullable=False, default=0
    )  # S&P/TSX Composite
    
    # Harvard Universe classification flags
    etf_flag = Column(
        Integer, nullable=False, default=0
    )  # Exchange-traded fund
    mutual_fund_flag = Column(
        Integer, nullable=False, default=0
    )  # Mutual fund
    common_stock_flag = Column(
        Integer, nullable=False, default=0
    )  # Common stock

    # Unique constraint
    __table_args__ = (
        UniqueConstraint(
            "symbol", "country", name="uq_symbols_symbol_country"
        ),
    )
    
    # Relationships
    aliases = relationship(
        "InstrumentAlias",
        back_populates="instrument",
        cascade="all, delete-orphan",
    )
    snapshots = relationship(
        "CvarSnapshot",
        primaryjoin="Symbols.id==CvarSnapshot.instrument_id",
        viewonly=True,
    )
    compass_inputs = relationship(
        "CompassInputs",
        back_populates="symbols",  # Updated relationship name
        cascade="all, delete-orphan",
    )
    
    def __repr__(self) -> str:
        return (
            f"<Symbols(id={self.id}, symbol='{self.symbol}', "
            f"country='{self.country}')>"
        )
    
    # ==========================================
    # UTILITY METHODS
    # ==========================================
    
    def get_active_flags(self) -> list[str]:
        """Return list of active flag names (where flag=1)."""
        flag_columns = [
            'five_stars', 'sp_500', 'nasdaq_100',
            'dow_jones_industrial_average', 'russell_1000', 'sp_midcap_400',
            'sp_smallcap_600', 'ftse_100', 'ftse_350', 'tsx_60',
            'sp_60_constituents', 'sp_composite', 'etf_flag',
            'mutual_fund_flag', 'common_stock_flag'
        ]
        active_flags = []
        for flag in flag_columns:
            if getattr(self, flag, 0) == 1:
                active_flags.append(flag)
        return active_flags
    
    def set_flag(self, flag_name: str, value: int = 1) -> bool:
        """Set a flag value.
        
        Returns True if successful, False if flag doesn't exist.
        """
        if hasattr(self, flag_name):
            setattr(self, flag_name, value)
            return True
        return False
    
    def is_in_index(self, index_name: str) -> bool:
        """Check if symbol is in a specific index."""
        flag_mapping = {
            'sp500': 'sp_500',
            's&p500': 'sp_500', 
            'sp_500': 'sp_500',
            'nasdaq100': 'nasdaq_100',
            'nasdaq_100': 'nasdaq_100',
            'dow': 'dow_jones_industrial_average',
            'dow_jones': 'dow_jones_industrial_average',
            'russell1000': 'russell_1000',
            'russell_1000': 'russell_1000',
            'ftse100': 'ftse_100',
            'ftse_100': 'ftse_100',
            'ftse350': 'ftse_350',
            'ftse_350': 'ftse_350',
            'tsx60': 'tsx_60',
            'tsx_60': 'tsx_60',
            'sp_60_constituents': 'sp_60_constituents',
            'sp60': 'sp_60_constituents',
            'sp_composite': 'sp_composite',
            'tsx_composite': 'sp_composite',
        }
        
        normalized_name = index_name.lower().replace(' ', '').replace('-', '_')
        flag_name = flag_mapping.get(normalized_name, normalized_name)
        
        return getattr(self, flag_name, 0) == 1
    
    @classmethod
    def get_flag_columns(cls) -> list[str]:
        """Return list of all flag column names."""
        return [
            'five_stars', 'sp_500', 'nasdaq_100',
            'dow_jones_industrial_average', 'russell_1000', 'sp_midcap_400',
            'sp_smallcap_600', 'ftse_100', 'ftse_350', 'tsx_60',
            'sp_60_constituents', 'sp_composite', 'etf_flag',
            'mutual_fund_flag', 'common_stock_flag'
        ]


# Backward compatibility alias
PriceSeries = Symbols


class InstrumentAlias(Base):
    __tablename__ = "instrument_alias"

    id = Column(Integer, primary_key=True, autoincrement=True)
    instrument_id = Column(Integer, ForeignKey("symbols.id"), nullable=False)
    alias_symbol = Column(String(128), nullable=False, index=True)
    alias_exchange = Column(String(64), nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationships
    instrument = relationship("Symbols", back_populates="aliases")
