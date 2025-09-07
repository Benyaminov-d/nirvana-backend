from datetime import datetime
from sqlalchemy import (  # type: ignore
    Column,
    Integer,
    String,
    DateTime,
    Text,
    Index,
)
from core.db import Base


class RiskModels(Base):
    """
    Reference table for CVaR model configurations.
    
    Defines methodology versions for L_i_99 = -CVaR^99% computation.
    Used for traceability and replay of risk calculations.
    """
    __tablename__ = "risk_models"

    cvar_model_id = Column(String(64), primary_key=True)  # "parametric-v1.2", "historical-v2.0"
    
    # Model specification
    distribution = Column(String(32), nullable=False)      # "normal", "t-student", "historical"
    horizon_days = Column(Integer, nullable=False)         # 252 (1 year)
    lookback_days = Column(Integer, nullable=False)        # 1260 (5 years)
    confidence = Column(String(16), nullable=False)        # "99%"
    
    # Technical details
    annualization_method = Column(String(32), nullable=False)  # "sqrt252", "business_days"
    code_version = Column(String(64), nullable=True)       # Git commit/library version
    definition = Column(Text, nullable=True)               # Full methodology description
    
    # Metadata
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    created_by = Column(String(128), nullable=True)        # User/system that created
    
    def __repr__(self) -> str:
        return (
            f"<RiskModels(id='{self.cvar_model_id}', "
            f"distribution='{self.distribution}', confidence='{self.confidence}')>"
        )


class MuPolicies(Base):
    """
    Reference table for Expected Return Estimation Policies.
    
    Defines methodology versions for μ_i computation.
    Used for traceability and replay of return calculations.
    """
    __tablename__ = "mu_policies"

    mu_policy_id = Column(String(64), primary_key=True)  # "simple-12mo-v1.0", "ema-weighted-v1.1"
    
    # Policy specification
    definition = Column(Text, nullable=False)            # "Simple 12-month total return, net of fees, base currency"
    window_days = Column(Integer, nullable=False)        # 365 (1 year lookback)
    weighting_scheme = Column(String(32), nullable=False)  # "equal", "exponential", "linear"
    
    # Technical details
    code_version = Column(String(64), nullable=True)     # Git commit/library version
    base_currency_handling = Column(String(64), nullable=True)  # Currency conversion method
    
    # Metadata  
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    created_by = Column(String(128), nullable=True)      # User/system that created
    
    def __repr__(self) -> str:
        return (
            f"<MuPolicies(id='{self.mu_policy_id}', "
            f"window_days={self.window_days}, weighting='{self.weighting_scheme}')>"
        )
