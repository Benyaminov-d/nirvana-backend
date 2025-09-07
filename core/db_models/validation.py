from datetime import datetime
from sqlalchemy import (  # type: ignore
    Column,
    Integer,
    String,
    Date,
    DateTime,
    JSON,
    UniqueConstraint,
    ForeignKey,
)
from sqlalchemy.orm import relationship  # type: ignore

from core.db import Base


class ValidationFlags(Base):
    """
    Detailed validation flags for price series data.
    
    Tracks specific reasons why data was rejected or accepted with warnings.
    Based on Data Knockout Policy documentation.
    """
    __tablename__ = "validation_flags"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(128), nullable=False, index=True)
    country = Column(String(64), nullable=True, index=True)
    as_of_date = Column(Date, nullable=False, index=True)
    
    # Overall validity flag (syncs with price_series.insufficient_history)
    valid = Column(Integer, nullable=False, default=1)  # 1 = valid, 0 = invalid
    
    # === HISTORY CRITERIA ===
    # InsufficientHistoryError: actual_years + 0.08 < min_years
    insufficient_total_history = Column(Integer, nullable=False, default=0)
    
    # InsufficientDataError: len(finite_returns) < 2
    insufficient_data_after_cleanup = Column(Integer, nullable=False, default=0)
    
    # === STRUCTURAL CRITERIA ===
    # ValidationResult flags from nirvana_risk/timeseries/validation.py
    backward_dates = Column(Integer, nullable=False, default=0)
    zero_or_negative_prices = Column(Integer, nullable=False, default=0) 
    extreme_price_jumps = Column(Integer, nullable=False, default=0)
    
    # === LIQUIDITY CRITERIA ===
    # LiquidityDecision flags from coverage.py
    critical_years = Column(Integer, nullable=False, default=0)  # Any year < 150 obs
    multiple_violations_last252 = Column(Integer, nullable=False, default=0)  # ≥2 violations
    multiple_weak_years = Column(Integer, nullable=False, default=0)  # ≥2 years with 150-199 obs
    low_liquidity_warning = Column(Integer, nullable=False, default=0)  # 1 violation or 1 weak year
    
    # === ANOMALY CRITERIA ===
    # From validator_core.pyx and anomalies.py
    robust_outliers = Column(Integer, nullable=False, default=0)  # Robust Z-score violations
    price_discontinuities = Column(Integer, nullable=False, default=0)  # Split detection patterns
    long_plateaus = Column(Integer, nullable=False, default=0)  # ≥20 consecutive equal prices
    illiquid_spikes = Column(Integer, nullable=False, default=0)  # Large moves with zero volume
    
    # === ANALYTICS DATA ===
    # Detailed metrics for dashboard analytics
    liquidity_metrics = Column(JSON, nullable=True)  # last252 stats, per-year counts
    anomaly_details = Column(JSON, nullable=True)    # robust z-scores, detected patterns
    validation_summary = Column(JSON, nullable=True) # summary stats for quick reporting
    
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint(
            "symbol", "country", "as_of_date", 
            name="uq_validation_flags_symbol_country_date"
        ),
    )

    @property
    def has_critical_issues(self) -> bool:
        """Check if symbol has any critical issues that block processing."""
        return bool(
            self.insufficient_total_history or 
            self.insufficient_data_after_cleanup or
            self.backward_dates or 
            self.zero_or_negative_prices or
            self.extreme_price_jumps or
            self.critical_years or
            self.multiple_violations_last252 or
            self.multiple_weak_years
        )
    
    @property 
    def has_warnings(self) -> bool:
        """Check if symbol has warnings but can still be processed."""
        return bool(
            self.low_liquidity_warning or
            self.long_plateaus or 
            self.illiquid_spikes
        )
    
    @property
    def rejection_reasons(self) -> list[str]:
        """Get list of rejection reasons for this symbol."""
        reasons = []
        if self.insufficient_total_history:
            reasons.append("insufficient_total_history")
        if self.insufficient_data_after_cleanup:
            reasons.append("insufficient_data_after_cleanup")
        if self.backward_dates:
            reasons.append("backward_dates")
        if self.zero_or_negative_prices:
            reasons.append("zero_or_negative_prices")
        if self.extreme_price_jumps:
            reasons.append("extreme_price_jumps")
        if self.critical_years:
            reasons.append("critical_years")
        if self.multiple_violations_last252:
            reasons.append("multiple_violations_last252")
        if self.multiple_weak_years:
            reasons.append("multiple_weak_years")
        if self.robust_outliers:
            reasons.append("robust_outliers")
        if self.price_discontinuities:
            reasons.append("price_discontinuities")
        return reasons


__all__ = [
    "ValidationFlags",
]
