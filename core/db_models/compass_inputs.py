from datetime import datetime
from sqlalchemy import (  # type: ignore
    Column,
    Integer,
    String,
    DateTime,
    Float,
    ForeignKey,
    Index,
)
from sqlalchemy.orm import relationship  # type: ignore
from core.db import Base


class CompassInputs(Base):
    """
    Core financial parameters per instrument x category x version.
    
    Stores computed μ_i (expected annual return) and L_i_99 (-CVaR^99%) 
    for Compass Score calculation and anchor calibration.
    
    Based on Compass Score v1.0 specification:
    - μ_i: expected simple 12-month total return, net of fees, pre-tax, base currency
    - L_i_99: tail-loss = -CVaR^99%, used for Safety component S_i
    
    This is the Single Source of Truth (SSoT) for Compass calculations.
    """
    __tablename__ = "compass_inputs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # === INSTRUMENT × CATEGORY × VERSION ===
    instrument_id = Column(Integer, ForeignKey("symbols.id"), nullable=False)
    category_id = Column(String(64), nullable=False)  # "US", "UK", "CA"  
    version_id = Column(String(128), nullable=False)  # "US-Equity-Returns--v2025Q4"
    
    # === CORE PARAMETERS ===
    # μ_i - Expected annual return (percentage points per year)
    # Net of expense ratio, pre-tax, in base currency
    # Used for: R_i normalization and anchor calibration
    mu_i = Column(Float, nullable=False)
    
    # L_i_99 - Tail loss = -CVaR^99% (percentage points per year)  
    # Used for: S_i = clip(1 - L_i_99/LT, 0, 1) and Nirvana Standard gate
    L_i_99 = Column(Float, nullable=False)
    
    # === TECHNICAL METADATA ===
    # Policy and model identifiers for traceability
    mu_policy_id = Column(String(64), nullable=True)    # Expected Return Estimation Policy
    cvar_model_id = Column(String(64), nullable=True)   # CVaR methodology version
    
    # Processing metadata
    calc_timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    data_vendor = Column(String(32), nullable=False, default="EODHD")  # EODHD, Bloomberg, etc.
    run_id = Column(String(128), nullable=True)         # ETL batch identifier
    
    # Audit fields
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # === RELATIONSHIPS ===
    symbols = relationship("Symbols", back_populates="compass_inputs")
    
    __table_args__ = (
        # Primary queries: get all μ_i for a category x version (anchor calibration)
        Index("ix_compass_inputs_category_version", "category_id", "version_id"),
        
        # Runtime queries: get μ_i, L_i_99 for specific instrument x version  
        Index("ix_compass_inputs_instrument_version", "instrument_id", "version_id"),
        
        # Anchor calibration: efficient μ_i queries
        Index("ix_compass_inputs_mu_i", "mu_i"),
        
        # Unique constraint: one record per instrument x category x version
        Index("uq_compass_inputs", "instrument_id", "category_id", "version_id", unique=True),
    )

    def __repr__(self) -> str:
        return (
            f"<CompassInputs(instrument_id={self.instrument_id}, "
            f"category='{self.category_id}', version='{self.version_id}', "
            f"mu_i={self.mu_i:.4f}, L_i_99={self.L_i_99:.4f})>"
        )
