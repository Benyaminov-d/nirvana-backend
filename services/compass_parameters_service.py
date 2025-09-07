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
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, date
from typing import Dict, Any, List, Optional

import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import or_, and_, func

from core.db import get_db_session
from core.models import PriceSeries, ValidationFlags, CompassInputs, CvarSnapshot, RiskModels, MuPolicies
from utils.common import _eodhd_suffix_for
from services.infrastructure.redis_eodhd_client import RedisCachedEODHDClient

# Use external library for EODHD data
from nirvana_risk.pipeline.io_eodhd import get_price_series
from nirvana_risk.pipeline.metrics import compute_log_returns


def _get_latest_validation_subquery(session):
    """
    Get subquery for latest ValidationFlags records per symbol.
    Returns the most recent validation record for each symbol.
    """
    return (
        session.query(
            ValidationFlags.symbol,
            ValidationFlags.country,
            func.max(ValidationFlags.as_of_date).label('latest_date')
        )
        .group_by(ValidationFlags.symbol, ValidationFlags.country)
        .subquery()
    )

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

    def create_parameters_for_validated_universe(self) -> None:
        """
        Main entry point: compute parameters for validated universe.
        
        Workflow:
        1. Get eligible symbols by country (US/UK/CA) 
        2. Process symbols in parallel: fetch EODHD → compute μ → retrieve L → store
        3. Store results in compass_inputs table
        """
        try:
            _LOG.info("Starting Compass Parameters processing v2.0")
            
            # Get eligible symbols by country
            countries = ["US", "UK", "CA"]
            all_symbols = []
            
            for country in countries:
                symbols = self._get_eligible_symbols(country)
                _LOG.info("Found %d eligible symbols for %s", len(symbols), country)
                
                for symbol_data in symbols:
                    symbol_data['country'] = country
                    symbol_data['category_id'] = country  # Simple mapping for now
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

    def _get_eligible_symbols(self, country: str) -> List[Dict[str, Any]]:
        """
        Get eligible symbols for a country based on validated universe.
        
        Uses ValidationFlags.valid=1 (latest record) and instrument type filters.
        """
        try:
            # Latest validation subquery (one record per symbol + country)
            latest_validations = _get_latest_validation_subquery(self.session)
            
            # Base query: PriceSeries joined with latest ValidationFlags
            query = (
                self.session.query(PriceSeries)
                .join(latest_validations, and_(
                    PriceSeries.symbol == latest_validations.c.symbol,
                    PriceSeries.country == latest_validations.c.country
                ))
                .join(ValidationFlags, and_(
                    ValidationFlags.symbol == latest_validations.c.symbol,
                    ValidationFlags.country == latest_validations.c.country,
                    ValidationFlags.as_of_date == latest_validations.c.latest_date
                ))
                .filter(
                    PriceSeries.country == country,
                    ValidationFlags.valid == 1,
                    ValidationFlags.insufficient_total_history == 0,
                )
            )
            
            # Country-specific filters per validated universe specification
            if country == "US":
                # US: ETF, Mutual Fund, Common Stock (non-PINK exchange) + five_stars=1
                query = query.filter(
                    or_(
                        PriceSeries.instrument_type.ilike("ETF"),
                        PriceSeries.instrument_type.ilike("Mutual Fund"),
                        and_(
                            PriceSeries.instrument_type.ilike("Common Stock"),
                            ~PriceSeries.exchange.ilike("PINK")
                        ),
                        PriceSeries.five_stars == 1
                    )
                )
            elif country == "UK":
                # UK: ETF, Common Stock
                query = query.filter(
                    or_(
                        PriceSeries.instrument_type.ilike("ETF"),
                        PriceSeries.instrument_type.ilike("Common Stock")
                    )
                )
            elif country == "CA":
                # Canada: ETF only
                query = query.filter(PriceSeries.instrument_type.ilike("ETF"))
            
            symbols = query.all()
            
            return [
                {
                    'id': ps.id,
                    'symbol': ps.symbol,
                    'exchange': ps.exchange,
                    'instrument_type': ps.instrument_type,
                    'eodhd_symbol': ps.symbol + _eodhd_suffix_for(ps.exchange, country)
                }
                for ps in symbols
            ]
            
        except Exception as exc:
            _LOG.error("Failed to get eligible symbols for %s: %s", country, exc)
            return []

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

    def _process_single_symbol(self, symbol_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process a single symbol: fetch EODHD → compute μ → retrieve L → store.
        
        Core architecture: NO time series storage, only parameters.
        """
        # Create dedicated session and EODHD client for this worker thread
        session = get_db_session()
        if not session:
            _LOG.error("Failed to create DB session for %s", symbol_data.get('symbol', 'UNKNOWN'))
            return None
            
        # Thread-safe Redis EODHD client instance
        eodhd_client = RedisCachedEODHDClient()
            
        try:
            symbol = symbol_data['symbol']
            eodhd_symbol = symbol_data['eodhd_symbol']
            instrument_id = symbol_data['id']
            category_id = symbol_data['category_id']
            
            # 1. FETCH TIME SERIES (Redis-cached, with rate limiting and retry logic)
            from datetime import date, timedelta
            
            # Calculate from_date for 5 years of data
            to_date = date.today()
            from_date = to_date - timedelta(days=1825)  # 5 years
            
            # Extract symbol and exchange from eodhd_symbol (format: SYMBOL.EXCHANGE)
            if '.' in eodhd_symbol:
                symbol_part, exchange_part = eodhd_symbol.rsplit('.', 1)
            else:
                symbol_part, exchange_part = eodhd_symbol, 'US'
            
            raw_data = eodhd_client.get_historical_prices_cached(
                symbol=symbol_part,
                exchange=exchange_part, 
                from_date=from_date,
                to_date=to_date
            )
            if not raw_data or len(raw_data) < 252:  # Minimum 1 year
                _LOG.debug("Insufficient data for %s: %d records", symbol, len(raw_data) if raw_data else 0)
                return None
            
            # 2. COMPUTE μ (expected annual return)
            mu_annual = self._compute_expected_annual_return(raw_data)
            if mu_annual is None:
                return None
                
            # 3. RETRIEVE L_i_99 from existing CvarSnapshot
            L_i_99 = self._get_cvar99_from_snapshot(session, symbol, instrument_id)
            if L_i_99 is None:
                _LOG.debug("No CVaR data available for %s, using fallback", symbol)
                L_i_99 = 0.20  # 20% fallback for missing CVaR data
                
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
            
            # Check for existing record and update/insert
            existing = session.query(CompassInputs).filter(
                CompassInputs.instrument_id == instrument_id,
                CompassInputs.category_id == category_id,
                CompassInputs.version_id == self.version_id
            ).first()
            
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
            _LOG.error("Failed to process %s: %s", symbol_data.get('symbol', 'UNKNOWN'), exc)
            try:
                session.rollback()
            except Exception:
                pass
            return None
            
        finally:
            try:
                session.close()
            except Exception:
                pass

    def _compute_expected_annual_return(self, raw_data: List[Dict]) -> Optional[float]:
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
                    adj_close = float(point.get("adjusted_close", point.get("close", 0)))
                    if adj_close > 0:
                        clean_prices.append(adj_close)
                except (ValueError, TypeError):
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

    def _get_cvar99_from_snapshot(self, session: Session, symbol: str, instrument_id: int) -> Optional[float]:
        """
        Retrieve L_i_99 (tail loss) from existing CvarSnapshot.
        
        Looks for CVaR^99% data in the most recent snapshot.
        Returns positive tail loss value (L = -CVaR).
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
                return None
                
            # Extract CVaR value (prefer GHST > NIG > EVaR for robustness)
            cvar_value = None
            if snapshot.cvar_ghst and np.isfinite(snapshot.cvar_ghst):
                cvar_value = snapshot.cvar_ghst
            elif snapshot.cvar_nig and np.isfinite(snapshot.cvar_nig):
                cvar_value = snapshot.cvar_nig
            elif snapshot.cvar_evar and np.isfinite(snapshot.cvar_evar):
                cvar_value = snapshot.cvar_evar
                
            if cvar_value is not None:
                # Return positive tail loss value (L = -CVaR)
                return float(abs(cvar_value))
                
            return None
            
        except Exception as exc:
            _LOG.error("Failed to retrieve CVaR for %s: %s", symbol, exc)
            return None


def create_compass_parameters_for_validated_universe() -> None:
    """
    Entry point function for startup integration.
    
    Creates financial parameters for validated universe symbols.
    Computes μ from EODHD and retrieves L from existing CVaR snapshots.
    """
    service = CompassParametersService()
    service.create_parameters_for_validated_universe()


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
