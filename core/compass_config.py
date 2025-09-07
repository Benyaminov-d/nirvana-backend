"""
Safe configuration for anchor calibration only.

This module contains ONLY non-proprietary configuration for statistical
calibration procedures. All proprietary parameters are in the compiled library.
"""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class AnchorCalibConfig:
    """Configuration for statistical anchor calibration (non-proprietary)."""

    # Standard statistical winsorization percentiles
    p_winsor_low: float = 0.01
    p_winsor_high: float = 0.99

    # Harrell-Davis quantile estimation (standard statistical method)
    p_hd_low: float = 0.05   # 5th percentile
    p_hd_high: float = 0.95  # 95th percentile

    # Spread constraints (risk management)
    min_spread: float = 0.02  # 2% minimum spread
    max_spread: float = 0.60  # 60% maximum spread

    @classmethod
    def from_environment(cls) -> 'AnchorCalibConfig':
        """Create configuration from environment variables."""
        return cls(
            p_winsor_low=float(os.getenv('ANCHOR_WINSOR_LOW', '0.01')),
            p_winsor_high=float(os.getenv('ANCHOR_WINSOR_HIGH', '0.99')),
            p_hd_low=float(os.getenv('ANCHOR_HD_LOW', '0.05')),
            p_hd_high=float(os.getenv('ANCHOR_HD_HIGH', '0.95')),
            min_spread=float(os.getenv('ANCHOR_MIN_SPREAD', '0.02')),
            max_spread=float(os.getenv('ANCHOR_MAX_SPREAD', '0.60')),
        )


def get_anchor_config() -> AnchorCalibConfig:
    """Get anchor calibration configuration."""
    return AnchorCalibConfig.from_environment()
