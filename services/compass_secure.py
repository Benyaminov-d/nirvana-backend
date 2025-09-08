"""
Secure Compass Score interface.

This module provides a secure interface to the proprietary Compass Score
calculation library without exposing any formulas, parameters, or algorithms.
"""

import logging
from typing import Any, Dict, Optional

_LOG = logging.getLogger(__name__)

# Set up detailed logging
_LOG.setLevel(logging.DEBUG)
if not _LOG.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(
        logging.Formatter("%(asctime)s COMPASS_SECURE %(levelname)s: %(message)s")
    )
    _LOG.addHandler(_h)


class CompassSecureInterface:
    """Secure interface to proprietary Compass Score calculation."""

    def __init__(self):
        self._score_func = None
        self._score_breakdown_func = None
        self._initialized = False

    def _ensure_initialized(self):
        """Initialize the compiled library interface."""
        if self._initialized:
            return

        try:
            # Import from compiled binary library only
            _LOG.debug("Initializing Compass Score library (compiled mode)")
            from nirvana_risk_core import compass as _c  # type: ignore
            self._score_func = _c.score
            self._score_breakdown_func = _c.score_with_breakdown
            self._initialized = True
            _LOG.info("Successfully initialized Compass Score compiled library")
        except ImportError:
            try:
                # Fallback to binary module
                _LOG.debug("Compiled module not found, trying fallback import")
                from importlib import import_module
                mod = import_module("nirvana_risk_core.compass_core")
                self._score_func = getattr(mod, "score")
                breakdown_func = getattr(mod, "score_with_breakdown")
                self._score_breakdown_func = breakdown_func
                self._initialized = True
                _LOG.info("Successfully initialized Compass Score fallback library")
            except Exception as exc:
                _LOG.error(f"Failed to initialize Compass library: {exc}")
                msg = f"Compass library not available: {exc}"
                raise RuntimeError(msg)

    def compute_score(
        self,
        mu: float,
        L: float,
        anchor_low: float,
        anchor_high: float,
        tolerance: float
    ) -> int:
        """
        Compute proprietary risk-adjusted score.

        All parameters and algorithms are proprietary to Nirvana.

        Args:
            mu: Expected annual return
            L: Tail loss measure
            anchor_low: Lower anchor (from calibration)
            anchor_high: Upper anchor (from calibration)
            tolerance: Risk tolerance

        Returns:
            Proprietary score (integer)
        """
        self._ensure_initialized()

        try:
            # Call proprietary algorithm (parameters are internal to library)
            _LOG.debug(
                f"Computing score with params: mu={mu:.4f} L={L:.4f} anchors=[{anchor_low:.4f},{anchor_high:.4f}] tol={tolerance:.4f}"
            )
            score = self._score_func(mu, L, anchor_low, anchor_high, tolerance)
            _LOG.debug(f"Score calculation result: {int(score)}")
            return int(score)
        except Exception as exc:
            _LOG.warning("Score calculation failed: %s", exc)
            return 0

    def compute_score_with_breakdown(
        self,
        mu: float,
        L: float,
        anchor_low: float,
        anchor_high: float,
        tolerance: float
    ) -> Dict[str, Any]:
        """
        Compute proprietary score with internal components.

        Returns breakdown for debugging/analysis. All computation
        details are proprietary to Nirvana.
        """
        self._ensure_initialized()

        try:
            # Call proprietary algorithm with breakdown
            _LOG.debug(
                f"Computing score breakdown with params: mu={mu:.4f} L={L:.4f} anchors=[{anchor_low:.4f},{anchor_high:.4f}] tol={tolerance:.4f}"
            )
            breakdown_args = (mu, L, anchor_low, anchor_high, tolerance)
            result = self._score_breakdown_func(*breakdown_args)
            _LOG.debug(f"Score breakdown R={result.get('R'):.4f} S={result.get('S'):.4f} score={result.get('score')}")
            return result
        except Exception as exc:
            _LOG.warning("Score breakdown calculation failed: %s", exc)
            return {
                "score": 0,
                "error": str(exc)
            }


# Global secure interface instance
_secure_interface: Optional[CompassSecureInterface] = None


def get_secure_compass() -> CompassSecureInterface:
    """Get the secure Compass Score interface singleton."""
    global _secure_interface
    if _secure_interface is None:
        _secure_interface = CompassSecureInterface()
    return _secure_interface
