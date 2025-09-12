"""
Validation Integration Service.

Integrates the detailed ValidationFlags from nirvana_risk library
with the backend database storage and symbols updates.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any, Dict, Optional

from sqlalchemy.orm import Session  # type: ignore

from core.db import get_db_session
from core.models import Symbols, ValidationFlags as DBValidationFlags

# Import the library ValidationFlags and aggregator
try:
    from nirvana_risk.timeseries import (
        ValidationFlags as LibValidationFlags,
        aggregate_validation_flags
    )
    # Types for coercion from event payloads
    from nirvana_risk.timeseries.types import (
        LiquidityDecision as LibLiquidityDecision,
        Last252Metrics as LibLast252Metrics,
    )
    from nirvana_risk.pipeline.errors import InsufficientHistoryError, InsufficientDataError
except ImportError:
    # Fallback if library is not available
    LibValidationFlags = None
    aggregate_validation_flags = None
    InsufficientHistoryError = Exception
    InsufficientDataError = Exception
    LibLiquidityDecision = None  # type: ignore
    LibLast252Metrics = None  # type: ignore

_logger = logging.getLogger("nirvana.validation_integration")


class ValidationIntegrationService:
    """Service for integrating ValidationFlags with database storage."""
    
    def __init__(self, db_session: Optional[Session] = None):
        """Initialize with optional database session."""
        self._session = db_session
        self._own_session = db_session is None
    
    def _get_session(self) -> Session:
        """Get database session (create if needed)."""
        if self._session is None:
            self._session = get_db_session()
        return self._session
    
    def _close_session(self) -> None:
        """Close session if we own it."""
        if self._own_session and self._session:
            try:
                self._session.close()
            except Exception:
                pass
            self._session = None

    @staticmethod
    def _normalize_country(country: Optional[str]) -> Optional[str]:
        """Normalize country names to standard codes: US, UK, CA.

        Falls back to uppercased 2-3 letter code when provided, otherwise None.
        """
        if not country:
            return None
        try:
            s = str(country).strip().lower()
        except Exception:
            return None
        if not s:
            return None
        if s in ("united states", "united states of america", "usa", "us"):
            return "US"
        if s in ("united kingdom", "great britain", "uk", "gb"):
            return "UK"
        if s in ("canada", "ca", "can"):
            return "CA"
        # If looks like a short code, return uppercased
        if len(s) in (2, 3) and s.isalpha():
            return s.upper()
        # Unknown mapping, keep original trimmed form
        return str(country).strip()

    def process_validation_result(
        self,
        symbol: str,
        validation_data: Dict[str, Any],
        country: Optional[str] = None,
        as_of_date: Optional[date] = None
    ) -> Optional[DBValidationFlags]:
        """
        Process validation results from load_prices() and related functions.
        
        Args:
            symbol: Symbol being validated
            validation_data: Results from load_prices(), anomaly detection, etc.
            country: Country code 
            as_of_date: Validation date (defaults to today)
        
        Returns:
            Database ValidationFlags record or None if library not available
        """
        if LibValidationFlags is None or aggregate_validation_flags is None:
            _logger.warning(
                "nirvana_risk library not available, skipping detailed validation"
            )
            return None
            
        session = self._get_session()
        
        if as_of_date is None:
            as_of_date = date.today()
        
        try:
            # Extract validation components from the data
            validation_result = None
            liquidity_decision = None
            anomaly_report = None
            history_error = None
            years_actual = None
            years_required = None
            returns_count = None
            
            # Parse validation_data from various sources
            success = validation_data.get("success", True)
            error_code = validation_data.get("code", "")
            
            # History errors
            if not success and error_code == "insufficient_history":
                years_actual = validation_data.get("years", 0.0)
                years_required = validation_data.get("min_years", 10.0)  # Default to 10 years
                history_error = InsufficientHistoryError(years=years_actual, min_years=years_required)
            elif not success and error_code == "insufficient_data":
                history_error = InsufficientDataError("insufficient data")
                returns_count = validation_data.get("returns_count")
            
            # Get anomaly report if available
            anomaly_report = validation_data.get("anomalies_report")

            # Get liquidity decision if available (coerce dict -> lib types)
            liquidity_decision = self._coerce_liquidity_decision(
                validation_data.get("liquidity_decision")
            )
            
            # Create ValidationFlags using library aggregator
            lib_flags = aggregate_validation_flags(
                symbol=symbol,
                validation_result=validation_result,
                liquidity_decision=liquidity_decision,
                anomaly_report=anomaly_report,
                history_error=history_error,
                years_actual=years_actual,
                years_required=years_required,
                returns_count=returns_count
            )
            
            # Resolve/normalize country to avoid ambiguous DB updates
            resolved_country = self._resolve_country(session, symbol, country)

            # Convert library ValidationFlags to database model
            db_flags = self._convert_to_db_model(
                lib_flags, 
                symbol=symbol, 
                country=resolved_country, 
                as_of_date=as_of_date
            )
            
            # Save to database
            db_record = self._save_validation_flags(session, db_flags)
            
            # Update both flags in symbols
            has_history_issues = (
                lib_flags.insufficient_total_history or 
                lib_flags.insufficient_data_after_cleanup
            )
            self._sync_symbols_flags(session, symbol, resolved_country, has_history_issues, lib_flags.valid)
            
            session.commit()
            return db_record
            
        except Exception as e:
            session.rollback()
            _logger.error(f"Failed to process validation result for {symbol}: {e}")
            raise
        finally:
            if self._own_session:
                self._close_session()

    def _coerce_liquidity_decision(self, obj: Any):
        """Convert incoming event payload to library LiquidityDecision.

        Accepts None, already-typed LibLiquidityDecision, or a plain dict with
        keys: status, reasons, per_year, last252{...}.
        """
        if obj is None or LibLiquidityDecision is None or LibLast252Metrics is None:
            return None
        # Already correct type
        if isinstance(obj, LibLiquidityDecision):
            return obj
        # Expecting dict structure
        if not isinstance(obj, dict):
            return None
        status = obj.get("status")
        reasons = obj.get("reasons") or []
        per_year = obj.get("per_year") or {}
        last = obj.get("last252") or None
        last252 = None
        if isinstance(last, dict):
            try:
                last252 = LibLast252Metrics(
                    n_obs=int(last.get("n_obs", 0)),
                    n_nonzero=int(last.get("n_nonzero", 0)),
                    zero_share=float(last.get("zero_share", 0.0)),
                    dropped_points_recent_total=int(last.get("dropped_points_recent_total", 0)),
                    dropped_points_recent_with_flag=int(last.get("dropped_points_recent_with_flag", 0)),
                    resampled=bool(last.get("resampled", False)),
                    unique_prices=last.get("unique_prices"),
                    plateau_share=last.get("plateau_share"),
                )
            except Exception:
                last252 = None
        try:
            return LibLiquidityDecision(
                status=str(status) if status is not None else "sufficient",
                reasons=list(reasons) if isinstance(reasons, list) else [],
                per_year=dict(per_year) if isinstance(per_year, dict) else {},
                last252=last252,
                as_of_effective=None,
            )
        except Exception:
            return None

    def _resolve_country(self, session: Session, symbol: str, country: Optional[str]) -> Optional[str]:
        """Resolve country deterministically to avoid ambiguous queries.

        Priority:
        1) Provided country (normalized)
        2) Unique country found in Symbols for this symbol
        3) Preferred among multiple: US > UK > CA
        4) None (leave unspecified)
        """
        norm = self._normalize_country(country)
        if norm:
            return norm
        try:
            rows = (
                session.query(Symbols.country)
                .filter(Symbols.symbol == symbol)
                .distinct()
                .all()
            )
            countries = [r[0] for r in rows if r and r[0]]
            if not countries:
                return None
            if len(countries) == 1:
                return str(countries[0]).strip()
            # deterministic preference
            for pref in ("US", "UK", "CA"):
                if pref in countries:
                    return pref
            # fallback to first sorted for stability
            return sorted(set(str(c).strip() for c in countries))[0]
        except Exception:
            return None
    
    def _convert_to_db_model(
        self, 
        lib_flags: LibValidationFlags, 
        symbol: str, 
        country: Optional[str], 
        as_of_date: date
    ) -> DBValidationFlags:
        """Convert library ValidationFlags to database model."""
        norm_country = self._normalize_country(country)
        return DBValidationFlags(
            symbol=symbol,
            country=norm_country,
            as_of_date=as_of_date,
            valid=1 if lib_flags.valid else 0,
            
            # History criteria
            insufficient_total_history=1 if lib_flags.insufficient_total_history else 0,
            insufficient_data_after_cleanup=1 if lib_flags.insufficient_data_after_cleanup else 0,
            
            # Structural criteria
            backward_dates=1 if lib_flags.backward_dates else 0,
            zero_or_negative_prices=1 if lib_flags.zero_or_negative_prices else 0,
            extreme_price_jumps=1 if lib_flags.extreme_price_jumps else 0,
            
            # Liquidity criteria
            critical_years=1 if lib_flags.critical_years else 0,
            multiple_violations_last252=1 if lib_flags.multiple_violations_last252 else 0,
            multiple_weak_years=1 if lib_flags.multiple_weak_years else 0,
            low_liquidity_warning=1 if lib_flags.low_liquidity_warning else 0,
            
            # Anomaly criteria  
            robust_outliers=1 if lib_flags.robust_outliers else 0,
            price_discontinuities=1 if lib_flags.price_discontinuities else 0,
            long_plateaus=1 if lib_flags.long_plateaus else 0,
            illiquid_spikes=1 if lib_flags.illiquid_spikes else 0,
            
            # Analytics data
            liquidity_metrics=lib_flags.liquidity_metrics,
            anomaly_details=lib_flags.anomaly_details,
            validation_summary=lib_flags.validation_summary,
        )
    
    def _save_validation_flags(
        self, 
        session: Session, 
        db_flags: DBValidationFlags
    ) -> DBValidationFlags:
        """Save or update ValidationFlags in database."""
        # Try to find existing record (prefer exact country match, then any country for that date)
        existing = session.query(DBValidationFlags).filter(
            DBValidationFlags.symbol == db_flags.symbol,
            DBValidationFlags.country == db_flags.country,
            DBValidationFlags.as_of_date == db_flags.as_of_date
        ).one_or_none()
        if existing is None:
            # When country is not specified, there may be multiple rows for different countries.
            # Use first() to avoid MultipleResultsFound while still updating a stable record
            existing = (
                session.query(DBValidationFlags)
                .filter(
                    DBValidationFlags.symbol == db_flags.symbol,
                    DBValidationFlags.as_of_date == db_flags.as_of_date,
                )
                .order_by(DBValidationFlags.country.asc())
                .first()
            )
        
        if existing:
            # Update existing record
            for attr in [
                'valid', 'insufficient_total_history', 'insufficient_data_after_cleanup',
                'backward_dates', 'zero_or_negative_prices', 'extreme_price_jumps',
                'critical_years', 'multiple_violations_last252', 'multiple_weak_years', 'low_liquidity_warning',
                'robust_outliers', 'price_discontinuities', 'long_plateaus', 'illiquid_spikes',
                'liquidity_metrics', 'anomaly_details', 'validation_summary'
            ]:
                setattr(existing, attr, getattr(db_flags, attr))
            existing.updated_at = datetime.utcnow()
            return existing
        else:
            # Create new record
            session.add(db_flags)
            return db_flags
    
    def _sync_symbols_flags(
        self, 
        session: Session, 
        symbol: str, 
        country: Optional[str], 
        has_history_issues: bool,
        is_valid: bool
    ) -> None:
        """Sync symbols flags with ValidationFlags."""
        norm_country = self._normalize_country(country)
        # Try match by symbol and normalized country first
        symbols = None
        if norm_country:
            symbols = session.query(Symbols).filter(
                Symbols.symbol == symbol,
                Symbols.country == norm_country
            ).one_or_none()
        # Fallback: match by symbol only
        if symbols is None:
            symbols = session.query(Symbols).filter(
                Symbols.symbol == symbol
            ).one_or_none()
        
        if symbols:
            # insufficient_history: only for data history problems
            symbols.insufficient_history = 1 if has_history_issues else 0
            
            # valid: general validity flag (any rejection reasons)
            symbols.valid = 1 if is_valid else 0
            
            _logger.debug(
                f"Updated {symbol}: insufficient_history={symbols.insufficient_history}, "
                f"valid={symbols.valid}"
            )


# Convenience functions
def process_ticker_validation(
    symbol: str,
    validation_data: Dict[str, Any],
    country: Optional[str] = None,
    as_of_date: Optional[date] = None,
    db_session: Optional[Session] = None
) -> Optional[DBValidationFlags]:
    """
    Process validation result for a single ticker.
    
    This is the main integration point for ticker processing pipelines.
    """
    service = ValidationIntegrationService(db_session)
    return service.process_validation_result(
        symbol=symbol,
        validation_data=validation_data,
        country=country,
        as_of_date=as_of_date
    )


def sync_insufficient_history_flags(limit: int = 1000) -> int:
    """
    Sync existing insufficient_history flags from symbols to validation_flags.
    
    This is useful for migrating existing data.
    
    Returns:
        Number of records processed
    """
    if LibValidationFlags is None:
        _logger.warning("nirvana_risk library not available, skipping migration")
        return 0
        
    session = get_db_session()
    count = 0
    
    try:
        # Get symbols records that have insufficient_history set but no validation_flags
        symbols_records = session.query(Symbols).filter(
            Symbols.insufficient_history.isnot(None)
        ).limit(limit).all()
        
        for ps in symbols_records:
            # Check if validation flags already exist
            existing = session.query(DBValidationFlags).filter(
                DBValidationFlags.symbol == ps.symbol,
                DBValidationFlags.country == ps.country
            ).first()
            
            if not existing:
                # Create basic validation flags from insufficient_history
                validation_data = {
                    "success": ps.insufficient_history != 1,
                    "code": "insufficient_history" if ps.insufficient_history == 1 else "success"
                }
                
                service = ValidationIntegrationService(session)
                service.process_validation_result(
                    symbol=ps.symbol,
                    validation_data=validation_data,
                    country=ps.country,
                    as_of_date=date.today()
                )
                count += 1
                
        session.commit()
        return count
        
    except Exception as e:
        session.rollback()
        _logger.error(f"Failed to sync insufficient_history flags: {e}")
        raise
    finally:
        session.close()
