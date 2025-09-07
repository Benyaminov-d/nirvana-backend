from datetime import datetime
from sqlalchemy import (  # type: ignore
    Column,
    Integer,
    String,
    DateTime,
    Float,
    Boolean,
    Index,
)
from core.db import Base


class CompassAnchorVersions(Base):
    """
    Quarterly anchor calibration versions per category.
    
    Stores μ_low, μ_high anchors with full provenance for replay/audit.
    Based on Compass Score v1.0 specification - Section 4 (Quarterly Calibration).
    
    Example version_id: "US-Equity-Returns--v2025Q4"
    """
    __tablename__ = "compass_anchor_versions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # === VERSION IDENTIFICATION ===
    category_id = Column(String(64), nullable=False, index=True)  # "US", "UK", "CA"
    version_id = Column(String(128), nullable=False, unique=True, index=True)  # "US-Equity-Returns--v2025Q4"
    as_of_date = Column(DateTime, nullable=False, index=True)  # Calibration date
    
    # === COMPUTED ANCHORS ===
    # Return anchors from Harrell-Davis 5%/95% quantiles
    mu_low = Column(Float, nullable=False)   # μ_low (percentage points per year)
    mu_high = Column(Float, nullable=False)  # μ_high (percentage points per year)
    
    # === PROVENANCE (for replay/audit) ===
    # Winsorization bounds (applied to μ_i before anchor computation)
    winsor_p1 = Column(Float, nullable=False, default=0.01)    # 1% (type-7)
    winsor_p99 = Column(Float, nullable=False, default=0.99)   # 99% (type-7)
    
    # Quantile estimator method and levels
    quantile_method = Column(String(32), nullable=False, default="Harrell-Davis")
    p_low = Column(Float, nullable=False, default=0.05)   # 5th percentile
    p_high = Column(Float, nullable=False, default=0.95)  # 95th percentile
    
    # Spreads and guardrails
    spread_before = Column(Float, nullable=False)  # μ_high - μ_low before guardrails
    spread_after = Column(Float, nullable=False)   # μ_high - μ_low after guardrails
    guardrails_applied = Column(Boolean, nullable=False, default=False)
    
    # Universe statistics 
    sample_size_n = Column(Integer, nullable=False)  # Number of μ_i values used
    clip_rate_low = Column(Float, nullable=True)     # % of R_i clipped at 0
    clip_rate_high = Column(Float, nullable=True)    # % of R_i clipped at 1
    
    # === OPTIONAL: 3-MONTH SMOOTHING ===
    # Raw anchors from each of the three preceding month-ends
    mu_low_m1 = Column(Float, nullable=True)   # Month -1
    mu_low_m2 = Column(Float, nullable=True)   # Month -2  
    mu_low_m3 = Column(Float, nullable=True)   # Month -3
    mu_high_m1 = Column(Float, nullable=True)  # Month -1
    mu_high_m2 = Column(Float, nullable=True)  # Month -2
    mu_high_m3 = Column(Float, nullable=True)  # Month -3
    smoothed = Column(Boolean, nullable=False, default=False)  # Used median smoothing
    
    # === METADATA ===
    code_version = Column(String(64), nullable=True)  # Git commit/build version
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    __table_args__ = (
        Index("ix_compass_anchor_category_date", "category_id", "as_of_date"),
    )

    def __repr__(self) -> str:
        return (
            f"<CompassAnchorVersions(version_id='{self.version_id}', "
            f"mu_low={self.mu_low:.4f}, mu_high={self.mu_high:.4f})>"
        )
