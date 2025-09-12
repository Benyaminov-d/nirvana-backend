"""
Compass Parameters Service v2.0 - Parameters Only Architecture

Computes and stores financial parameters (μ_i, L_i_99) for Compass Score calculation.

Workflow:
1. Fetch EODHD time series (temporary, in-memory)
2. Compute μ_i (expected annual return) from time series
3. Retrieve L_i_99 from existing CvarSnapshot (alpha_label=99)
4. Store parameters in compass_inputs table
5. Discard time series data

Used for:
- Compass Score calculation (R_i, S_i normalization)
- Anchor calibration (winsorization of μ_i cross-section)

Eliminates massive time series storage while maintaining deterministic computation.
"""

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, date
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import or_, and_, func

from core.db import get_db_session
from core.models import Symbols, ValidationFlags, CompassInputs, CvarSnapshot, RiskModels, MuPolicies
from utils.common import _eodhd_suffix_for
from services.infrastructure.redis_eodhd_client import RedisCachedEODHDClient
# Import only the constant to avoid circular dependency
from core.universe_config import ACTIVE_UNIVERSE

# Use external library for EODHD data
from nirvana_risk.pipeline.io_eodhd import get_price_series
from nirvana_risk.pipeline.metrics import compute_log_returns


_LOG = logging.getLogger(__name__)


class CompassParametersService:
    """
    Service for computing and storing financial parameters (μ, L).
    
    Computes expected annual return from EODHD data and retrieves CVaR from existing snapshots.
    Eliminates time series storage for improved performance.
    """

    def __init__(self):
        self.session = get_db_session()
        if not self.session:
            raise RuntimeError("Failed to create database session")
            
        # Initialize Redis-cached EODHD client
        self.eodhd_client = RedisCachedEODHDClient()
        
        # Processing configuration
        self.workers = int(os.environ.get("EXP_REPROCESS_WORKERS", "32"))
        
        # Model configuration  
        self.mu_policy_id = "simple-12mo-net-fees-v2.0"
        
        # Version for this processing run
        self.version_id = f"compass-params-{datetime.now().strftime('%Y%m%d')}"
        
        _LOG.info("CompassParametersService initialized with %d workers and Redis EODHD client", self.workers)

    def create_parameters_for_validated_universe(self, universe_type: str = ACTIVE_UNIVERSE) -> None:
        """
        Main entry point: compute parameters for validated universe.
        
        Workflow:
        1. Get eligible symbols from UniverseManager (same filtering as Harvard Universe)
        2. Process symbols in parallel: fetch EODHD → compute μ → retrieve L → store
        3. Store results in compass_inputs table
        
        Args:
            universe_type: Type of universe to use (default: ACTIVE_UNIVERSE)
        """
        try:
            _LOG.info("Starting Compass Parameters processing v2.0 for %s universe", universe_type)
            
            # Delayed import to avoid circular dependency
            from services.universe_manager import get_universe_manager
            
            # Get universe products using the same manager as elsewhere
            universe_manager = get_universe_manager(universe_type)
            universe_products = universe_manager.get_universe_products()
            
            _LOG.info("Found %d products in %s universe", len(universe_products), universe_type)
            
            # Convert to format expected by processing functions
            all_symbols = []
            for product in universe_products:
                if not product.symbol or not product.id:
                    continue
                
                # Normalize country codes for consistency
                country_code = self._normalize_country_code(product.country)
                
                # Create symbol data dictionary
                symbol_data = {
                    'id': product.id,
                    'symbol': product.symbol,
                    'exchange': product.exchange,
                    'instrument_type': product.instrument_type,
                    'country': product.country,
                    'category_id': country_code,  # Use normalized country code
                    'eodhd_symbol': product.symbol + _eodhd_suffix_for(product.exchange, product.country)
                }
                all_symbols.append(symbol_data)
            
            _LOG.info("Total symbols to process: %d", len(all_symbols))
            
            if not all_symbols:
                _LOG.warning("No eligible symbols found, skipping parameter computation")
                return
                
            # Process symbols in parallel
            self._process_symbols_parallel(all_symbols)
            
            _LOG.info("Compass Parameters processing completed successfully")
            
        except Exception as exc:
            _LOG.error("Compass Parameters processing failed: %s", exc)
            raise
        finally:
            if self.session:
                try:
                    self.session.close()
                except Exception:
                    pass

    # Removed _get_eligible_symbols method as we now use UniverseManager directly
    
    def _normalize_country_code(self, country: str) -> str:
        """
        Normalize country code to standard format.
        
        Args:
            country: Country name or code
            
        Returns:
            Normalized country code
        """
        # Standard country code mapping
        country_map = {
            # United Kingdom variants
            "UK": "UK",
            "United Kingdom": "UK", 
            "GB": "UK",
            "Great Britain": "UK",
            
            # United States variants
            "US": "US",
            "USA": "US",
            "United States": "US",
            "United States of America": "US",
            
            # Canada variants
            "CA": "CA",
            "Canada": "CA",
            
            # Other countries - add as needed
        }
        
        # Return normalized code or original if not found
        return country_map.get(country, country)

    def _process_symbols_parallel(self, symbols: List[Dict[str, Any]]) -> None:
        """
        Process symbols in parallel using ThreadPoolExecutor.
        
        Each worker:
        1. Fetches EODHD time series (temporary)
        2. Computes μ, L, volatility
        3. Stores parameters in compass_inputs
        4. Discards time series
        """
        processed_count = 0
        failed_count = 0
        
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            # Submit all jobs
            future_to_symbol = {
                executor.submit(self._process_single_symbol, symbol_data): symbol_data
                for symbol_data in symbols
            }
            
            # Process results as they complete
            for future in as_completed(future_to_symbol):
                symbol_data = future_to_symbol[future]
                
                try:
                    result = future.result()
                    if result:
                        processed_count += 1
                        
                        # Log progress periodically
                        if processed_count % 100 == 0:
                            _LOG.info("Progress: %d/%d processed, %d failed", 
                                     processed_count, len(symbols), failed_count)
                    else:
                        failed_count += 1
                        
                except Exception as exc:
                    failed_count += 1
                    _LOG.error("Processing failed for %s: %s", 
                              symbol_data.get('symbol', 'UNKNOWN'), exc)
        
        _LOG.info("Processing completed: %d success, %d failed", processed_count, failed_count)
        _LOG.info("Total symbols processed: %d out of %d (%.1f%%)", 
                 processed_count, len(symbols), 
                 (processed_count / len(symbols) * 100) if symbols else 0)

    def _process_single_symbol(self, symbol_data: Dict[str, Any], max_retries: int = 3) -> Optional[Dict[str, Any]]:
        """
        Process a single symbol: fetch EODHD → compute μ → retrieve L → store.
        
        Args:
            symbol_data: Symbol metadata with id, symbol, exchange, etc.
            max_retries: Maximum number of retry attempts for transient errors
            
        Returns:
            Result dictionary or None if processing failed
        """
        symbol = symbol_data.get('symbol')
        eodhd_symbol = symbol_data.get('eodhd_symbol', symbol)
        instrument_id = symbol_data.get('id')
        category_id = symbol_data.get('category_id', 'US')  # Default to US
        
        if not symbol or not instrument_id:
            _LOG.warning("Invalid symbol data: %s", symbol_data)
            return None
            
        # Create new session for thread safety
        session = get_db_session()
        if not session:
            _LOG.error("Failed to create session for %s", symbol)
            return None
            
        # Check if symbol is valid in validation_flags
        try:
            from core.models import ValidationFlags
            
            # Check if symbol has a valid=1 entry in validation_flags
            is_valid = session.query(ValidationFlags)\
                .filter(ValidationFlags.symbol == symbol, 
                        ValidationFlags.valid == 1)\
                .first()
                
            if not is_valid:
                _LOG.debug("Skipping %s: Not marked as valid in validation_flags", symbol)
                return None
                
        except Exception as e:
            _LOG.warning("Failed to check validation status for %s: %s", symbol, e)
            # Continue processing even if validation check fails
        
        try:
            # Retry loop for transient errors
            retry_count = 0
            while retry_count <= max_retries:
                try:
                    # 1. FETCH EODHD DATA
                    _LOG.debug("Fetching EODHD data for %s", eodhd_symbol)
                    # Calculate from_date as a proper date object (3 years ago)
                    from_date = datetime.now().date()
                    from_date = date(from_date.year - 3, from_date.month, from_date.day)
                    
                    # Check if we already have CompassInputs for this symbol
                    existing = session.query(CompassInputs).filter(
                        CompassInputs.instrument_id == instrument_id,
                        CompassInputs.category_id == category_id,
                        CompassInputs.version_id == self.version_id
                    ).first()
                    
                    if existing and existing.mu_i is not None and existing.L_i_99 is not None:
                        _LOG.debug("Using existing CompassInputs for %s", symbol)
                        return {
                            'mu_annual': existing.mu_i,
                            'L_i_99': existing.L_i_99,
                            'symbol': symbol,
                            'from_cache': True
                        }
                    
                    # If circuit breaker is open, try to use CvarSnapshot data directly
                    try:
                        raw_data = self.eodhd_client.get_historical_prices_cached(
                            eodhd_symbol, 
                            exchange="",  # Exchange is part of the symbol for EODHD
                            period="d",  # Daily data
                            from_date=from_date,  # 3 years ago
                            to_date=None  # Latest available
                        )
                    except Exception as e:
                        if "Circuit breaker OPEN" in str(e):
                            _LOG.warning("Circuit breaker open for %s, trying CvarSnapshot fallback", symbol)
                            # Use CvarSnapshot directly
                            cvar_data = self._get_cvar_snapshot_data(session, symbol)
                            if cvar_data and cvar_data.get('return_annual') is not None:
                                mu_annual = cvar_data.get('return_annual')
                                L_i_99 = cvar_data.get('cvar_nig', 0.20)
                                
                                # Store parameters from CvarSnapshot
                                compass_input = CompassInputs(
                                    instrument_id=instrument_id,
                                    category_id=category_id,
                                    version_id=self.version_id,
                                    mu_i=mu_annual,
                                    L_i_99=L_i_99,
                                    mu_policy_id=self.mu_policy_id,
                                    data_vendor="CvarSnapshot-Fallback",
                                    run_id=f"compass-params-{datetime.now().strftime('%Y%m%d-%H%M')}"
                                )
                                
                                if existing:
                                    existing.mu_i = mu_annual
                                    existing.L_i_99 = L_i_99
                                    existing.data_vendor = "CvarSnapshot-Fallback"
                                    existing.updated_at = datetime.utcnow()
                                else:
                                    session.add(compass_input)
                                    
                                session.commit()
                                
                                return {
                                    'mu_annual': mu_annual,
                                    'L_i_99': L_i_99,
                                    'symbol': symbol,
                                    'from_fallback': True
                                }
                        # Re-raise if not circuit breaker or no fallback data
                        raise
                    
                    if not raw_data or len(raw_data) < 252:  # Need at least 1 year
                        _LOG.warning("Insufficient data for %s: %d points", symbol, len(raw_data) if raw_data else 0)
                        return None
                        
                    # 2. COMPUTE μ (expected annual return)
                    mu_annual = self._compute_expected_annual_return(raw_data)
                    if mu_annual is None:
                        return None
                        
                    # 3. RETRIEVE L_i_99 from existing CvarSnapshot
                    L_i_99, reason = self._get_cvar99_from_snapshot(session, symbol, instrument_id)
                    if L_i_99 is None:
                        _LOG.info("Skipping %s: %s", symbol, reason)
                        return None  # Skip symbols without CVaR data
                        
                    # 4. STORE ONLY PARAMETERS
                    compass_input = CompassInputs(
                        instrument_id=instrument_id,
                        category_id=category_id,
                        version_id=self.version_id,
                        mu_i=mu_annual,
                        L_i_99=L_i_99,
                        mu_policy_id=self.mu_policy_id,
                        data_vendor="EODHD-Redis",  # Redis-cached EODHD
                        run_id=f"compass-params-{datetime.now().strftime('%Y%m%d-%H%M')}"
                    )
                    
                    if existing:
                        existing.mu_i = mu_annual
                        existing.L_i_99 = L_i_99
                        existing.updated_at = datetime.utcnow()
                    else:
                        session.add(compass_input)
                        
                    session.commit()
                    
                    return {
                        'mu_annual': mu_annual,
                        'L_i_99': L_i_99,
                        'symbol': symbol
                    }
                    
                except Exception as exc:
                    retry_count += 1
                    if retry_count <= max_retries:
                        _LOG.warning("Retry %d/%d for %s: %s", retry_count, max_retries, symbol, exc)
                        time.sleep(1)  # Wait before retry
                        try:
                            session.rollback()
                        except Exception:
                            pass
                    else:
                        _LOG.error("Failed to process %s after %d retries: %s", 
                                  symbol_data.get('symbol', 'UNKNOWN'), max_retries, exc)
                        try:
                            session.rollback()
                        except Exception:
                            pass
                        return None
            
            # Should never reach here
            return None
            
        finally:
            # Always close the session
            try:
                session.close()
            except Exception:
                pass
                
    def _get_cvar_snapshot_data(self, session, symbol: str) -> Optional[Dict[str, Any]]:
        """Get latest CvarSnapshot data for a symbol."""
        try:
            from sqlalchemy import func
            
            # Get latest snapshot
            latest_date = session.query(func.max(CvarSnapshot.as_of_date))\
                .filter(CvarSnapshot.symbol == symbol)\
                .scalar()
                
            if not latest_date:
                return None
                
            snapshot = session.query(CvarSnapshot)\
                .filter(
                    CvarSnapshot.symbol == symbol,
                    CvarSnapshot.as_of_date == latest_date,
                    CvarSnapshot.alpha_label == 99  # Use 99% CVaR
                )\
                .first()
                
            if snapshot:
                return {
                    'return_annual': snapshot.return_annual,
                    'cvar_nig': snapshot.cvar_nig,
                    'as_of_date': snapshot.as_of_date
                }
                
            return None
            
        except Exception as e:
            _LOG.warning("Failed to get CvarSnapshot data for %s: %s", symbol, e)
            return None

    def _compute_expected_annual_return(self, raw_data) -> Optional[float]:
        """
        Compute μ (expected annual return) from EODHD time series.
        
        Applies winsorization as per Compass Score specification.
        Returns annualized expected return in percentage points.
        """
        try:
            # Clean and prepare price data
            clean_prices = []
            for point in raw_data:
                try:
                    # Handle both PriceDataPoint objects and dictionaries
                    if hasattr(point, 'adjusted_close'):
                        adj_close = float(point.adjusted_close)
                    else:
                        adj_close = float(point.get("adjusted_close", point.get("close", 0)))
                    
                    if adj_close > 0:
                        clean_prices.append(adj_close)
                except (ValueError, TypeError, AttributeError):
                    continue
            
            if len(clean_prices) < 252:  # Minimum 1 year
                return None
                
            # Sort by date (data should already be sorted by EODHD)
            clean_prices = sorted(clean_prices)
            
            # Compute simple returns (as per Compass Score v1.0 specification)
            # Document requires "simple 12-month total return" not log returns
            simple_returns = []
            for i in range(1, len(clean_prices)):
                if clean_prices[i] > 0 and clean_prices[i-1] > 0:
                    simple_return = (clean_prices[i] / clean_prices[i-1]) - 1.0
                    simple_returns.append(simple_return)
            
            if len(simple_returns) < 252:
                return None
            
            # Compute annualized expected return (μ) - NO winsorization at time series level
            # per Compass Score v1.0 specification (winsorization only at anchor calibration level)
            returns_array = np.array(simple_returns)
            mu_annual = float(np.mean(returns_array) * 252)  # Annualized
            
            return mu_annual
            
        except Exception as exc:
            _LOG.error("Expected return computation failed: %s", exc)
            return None

    def _get_cvar99_from_snapshot(self, session: Session, symbol: str, instrument_id: int) -> Tuple[Optional[float], str]:
        """
        Retrieve L_i_99 (tail loss) from existing CvarSnapshot.
        
        Looks for CVaR^99% data in the most recent snapshot.
        Returns:
            - positive tail loss value (L = -CVaR) or None
            - reason string explaining why data is missing if None
        """
        try:
            # Query most recent CvarSnapshot with alpha_label=99 (99% confidence)
            snapshot = (
                session.query(CvarSnapshot)
                .filter(
                    CvarSnapshot.symbol == symbol,
                    CvarSnapshot.instrument_id == instrument_id,
                    CvarSnapshot.alpha_label == 99
                )
                .order_by(CvarSnapshot.as_of_date.desc())
                .first()
            )
            
            if not snapshot:
                return None, "No CVaR snapshot found"
                
            # Extract CVaR value (prefer GHST > NIG > EVaR for robustness)
            cvar_value = None
            reason = "No valid CVaR values in snapshot"
            
            if snapshot.cvar_ghst is not None and np.isfinite(snapshot.cvar_ghst):
                cvar_value = snapshot.cvar_ghst
                reason = "Using GHST model"
            elif snapshot.cvar_nig is not None and np.isfinite(snapshot.cvar_nig):
                cvar_value = snapshot.cvar_nig
                reason = "Using NIG model"
            elif snapshot.cvar_evar is not None and np.isfinite(snapshot.cvar_evar):
                cvar_value = snapshot.cvar_evar
                reason = "Using EVaR model"
                
            if cvar_value is not None:
                # Return positive tail loss value (L = -CVaR)
                return float(abs(cvar_value)), reason
                
            return None, reason
            
        except Exception as exc:
            error_msg = f"Failed to retrieve CVaR: {exc}"
            _LOG.error("Failed to retrieve CVaR for %s: %s", symbol, exc)
            return None, error_msg


def create_compass_parameters_for_validated_universe() -> None:
    """
    Entry point function for startup integration.
    
    Creates financial parameters for validated universe symbols.
    Computes μ from EODHD and retrieves L from existing CVaR snapshots.
    """
    try:
        service = CompassParametersService()
        service.create_parameters_for_validated_universe()
    except Exception as e:
        _LOG.error(f"Failed to create compass parameters: {e}")


def setup_reference_data() -> None:
    """
    Setup minimal reference data for mu policies.
    """
    session = get_db_session()
    if not session:
        _LOG.error("Failed to create session for reference data setup")
        return
        
    try:
        # Expected return policy
        mu_policy = MuPolicies(
            mu_policy_id="simple-12mo-net-fees-v2.0",
            definition="Simple 12-month total return, net of expense ratio, pre-tax, base currency",
            window_days=365,
            weighting_scheme="equal",
            code_version="v2.0.0",
            created_by="compass-parameters-service"
        )
        
        # Insert if not exists
        existing_mu = session.query(MuPolicies).filter(
            MuPolicies.mu_policy_id == mu_policy.mu_policy_id).first()
        if not existing_mu:
            session.add(mu_policy)
            session.commit()
            _LOG.info("Reference data setup completed")
        
    except Exception as exc:
        _LOG.error("Reference data setup failed: %s", exc)
        try:
            session.rollback()
        except Exception:
            pass
    finally:
        try:
            session.close()
        except Exception:
            pass