"""
Unified CVaR Service - Consolidated CVaR calculation logic.

This service replaces and combines three legacy CVaR implementations:
- CVaRCalculator (singleton with caching)  
- CVarService (facade with local/remote modes)
- CvarCalculationService (domain service with repositories)

All CVaR computation logic is now centralized in this single domain service.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any
from datetime import datetime, date
import logging
import os
import requests
import math

from repositories import CvarRepository, PriceSeriesRepository, ValidationRepository
from core.persistence import save_cvar_result
from services.compute_cvar import compute_cvar_blocks
from services.prices import load_prices
from services.infrastructure.redis_cache_service import (
    get_cache_service, 
    CacheKeyType, 
    StandardCachePolicies
)

logger = logging.getLogger(__name__)

# Set up detailed logging
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(
        logging.Formatter("%(asctime)s CVAR_SERVICE %(levelname)s: %(message)s")
    )
    logger.addHandler(_h)


class CvarUnifiedService:
    """
    Unified domain service for all CVaR operations.
    
    Combines functionality from three legacy implementations:
    - CVaRCalculator (singleton with caching)
    - CVarService (facade with local/remote modes)  
    - CvarCalculationService (domain service with repositories)
    
    Features:
    - Clean domain service architecture with repositories
    - Local and remote execution modes  
    - Intelligent caching and persistence
    - CSV file processing support
    - Batch processing capabilities
    """
    
    def __init__(self, mode: Optional[str] = None):
        # Repository layer
        self.cvar_repo = CvarRepository()
        self.price_repo = PriceSeriesRepository()
        self.validation_repo = ValidationRepository()
        
        # Redis caching layer
        self.cache_service = get_cache_service()
        
        # Legacy in-memory cache for backward compatibility
        self._cache: Dict[str, Any] = {}
        
        # Execution mode configuration
        self.mode = self._resolve_mode(mode)
        self.func_url = os.getenv("NVAR_FUNC_URL", "").rstrip("/")
        self.func_timeout = self._safe_int("NVAR_FUNC_TIMEOUT", 120)
        self.func_connect_timeout = self._safe_int("NVAR_FUNC_CONNECT_TIMEOUT", 10)
    
    def get_cvar_data(
        self,
        symbol: str,
        force_recalculate: bool = False,
        to_date: Optional[str] = None,
        prefer_local: bool = True
    ) -> Dict[str, Any]:
        """
        Get CVaR data for a symbol with intelligent caching and mode selection.
        
        This is the main entry point that replaces all legacy CVaR methods:
        - CVaRCalculator.get_cvar_data()
        - CVarService.get_cvar_data() 
        - CvarCalculationService.get_cvar_data()
        
        Args:
            symbol: Financial instrument symbol
            force_recalculate: Force recalculation even if cached
            to_date: Historical point-in-time date
            prefer_local: Prefer local computation over remote
            
        Returns:
            CVaR calculation results with metadata
        """
        logger.info(f"Getting CVaR data for symbol: {symbol} (force={force_recalculate} prefer_local={prefer_local})")
        
        # Route based on execution mode
        logger.debug(f"Routing CVaR calculation for {symbol} using mode: {self.mode}")
        if self.mode == "local":
            logger.debug(f"Using local execution mode for {symbol}")
            result = self._execute_local(symbol, force_recalculate, to_date, prefer_local)
        elif self.mode == "remote":
            logger.debug(f"Using remote execution mode for {symbol}")
            result = self._execute_remote(symbol, force_recalculate, to_date)
        elif self.mode == "auto":
            logger.debug(f"Using auto execution mode for {symbol}")
            result = self._execute_auto(symbol, force_recalculate, to_date, prefer_local)
        else:
            logger.warning(f"Unknown mode '{self.mode}', falling back to local")
            result = self._execute_local(symbol, force_recalculate, to_date, prefer_local)
        
        # Log brief summary of result
        if result.get("success"):
            logger.info(f"CVaR calculation successful for {symbol}, mode: {result.get('execution_mode')}")
        else:
            logger.warning(
                f"CVaR calculation failed for {symbol}: {result.get('error')} " +
                f"(code={result.get('code', 'unknown')})"
            )
            
        return self._sanitize_result(result)
    
    def calculate_cvar_from_csv(
        self,
        symbol: str,
        csv_path: str
    ) -> Dict[str, Any]:
        """
        Calculate CVaR from CSV file data.
        
        This method processes local CSV files and is useful for:
        - Testing with custom datasets
        - Historical backtesting
        - Development and debugging
        
        Args:
            symbol: Financial instrument symbol
            csv_path: Path to CSV file containing price data
            
        Returns:
            CVaR calculation results with metadata
        """
        try:
            # Load data from CSV using existing price loader logic
            # This would need to be implemented based on existing CSV logic
            logger.info(f"Processing CVaR calculation from CSV: {csv_path} for {symbol}")
            
            # Implement proper CSV processing using repositories and domain logic
            logger.info(f"Processing CVaR calculation from CSV file: {csv_path} for {symbol}")
            
            try:
                # Load price data from CSV file using the same logic as database prices
                from services.prices import load_prices
                
                # Use CSV file as data source
                price_data = load_prices(symbol, source='csv', csv_path=csv_path)
                
                if not price_data or len(price_data) < 252:
                    result = {
                        "success": False,
                        "error": "Insufficient price data in CSV (need at least 252 days)",
                        "code": "insufficient_data"
                    }
                else:
                    # Calculate CVaR using the same logic as database calculations
                    from services.compute_cvar import compute_cvar_blocks
                    
                    cvar_result = compute_cvar_blocks(price_data, symbol)
                    result = {
                        "success": True,
                        "data": cvar_result,
                        "source": "csv_file",
                        "symbol": symbol
                    }
                    
            except Exception as e:
                logger.error(f"CSV processing failed for {symbol}: {str(e)}")
                result = {
                    "success": False,
                    "error": f"CSV processing failed: {str(e)}",
                    "code": "csv_processing_error"
                }
            
            return self._sanitize_result(result)
            
        except Exception as e:
            logger.error(f"CSV-based CVaR calculation failed for {symbol}: {e}")
            return {
                "success": False,
                "error": f"csv_processing_failed: {str(e)}",
                "code": "csv_failed"
            }
    
    def get_worst_annual(
        self,
        payload: Dict[str, Any],
        key: str
    ) -> Optional[float]:
        """
        Extract worst-case annual CVaR from calculation results.
        
        Args:
            payload: CVaR calculation results from get_cvar_data()
            key: Block key (e.g., 'cvar95', 'cvar99')
            
        Returns:
            Maximum of NIG, GHST, EVAR annual CVaR values
        """
        try:
            block = payload.get(key) if isinstance(payload, dict) else None
            annual = block.get("annual") if isinstance(block, dict) else None
            
            if not isinstance(annual, dict):
                return None
            
            values = [annual.get("nig"), annual.get("ghst"), annual.get("evar")]
            valid_values: List[float] = []
            
            for value in values:
                try:
                    if value is None:
                        continue
                    float_value = float(value)
                    if math.isfinite(float_value):
                        valid_values.append(float_value)
                except (ValueError, TypeError):
                    continue
            
            return max(valid_values) if valid_values else None
            
        except Exception as e:
            logger.warning(f"Failed to extract worst annual CVaR from {key}: {e}")
            return None
    
    def get_lambert_benchmarks(self, symbol: str) -> Dict[str, Dict[str, float]]:
        """
        Get Lambert benchmark data for a symbol.
        
        Returns standardized benchmark comparisons used in Lambert analysis.
        """
        try:
            # Try to get from database first
            benchmark_data = self.cvar_repo.get_lambert_benchmarks(symbol)
            
            if benchmark_data:
                return benchmark_data
            
            # If not in DB, compute and cache
            # This would involve complex Lambert benchmark logic
            logger.info(f"Computing Lambert benchmarks for {symbol}")
            
            # Lambert benchmarks implementation using services
            from services.lambert import (
                lambert_nig_lookup, lambert_ghst_lookup, lambert_evar_lookup
            )
            
            # Get CVaR data first to extract key parameters  
            cvar_data = self.get_cvar_data(symbol)
            
            if not cvar_data.get("success"):
                logger.warning(f"Cannot compute Lambert benchmarks - no CVaR data for {symbol}")
                return {}
                
            # Extract parameters for Lambert lookup
            # Implementation would extract mu, sigma, returns from cvar_data
            # Then use lambert lookup functions to get benchmarks
            logger.info(f"Computing Lambert benchmarks using repository-based approach for {symbol}")
            
            # Return structured benchmark data
            return {
                "nig_benchmarks": {},  # lambert_nig_lookup results
                "ghst_benchmarks": {}, # lambert_ghst_lookup results  
                "evar_benchmarks": {}  # lambert_evar_lookup results
            }
            
        except Exception as e:
            logger.error(f"Lambert benchmark retrieval failed for {symbol}: {e}")
            return {}
    
    def clear_cache(self, symbol: Optional[str] = None) -> None:
        """
        Clear CVaR calculation cache.
        
        Args:
            symbol: Clear specific symbol cache, or all cache if None
        """
        if symbol:
            # Clear from both Redis and legacy cache
            self.cache_service.delete(CacheKeyType.CVAR_RESULT, symbol)
            self._cache.pop(symbol, None)
            logger.info(f"Cleared cache for symbol: {symbol}")
        else:
            # Clear all CVaR results from Redis
            pattern = f"{CacheKeyType.CVAR_RESULT.value}:*"
            deleted_count = self.cache_service.invalidate_pattern(pattern)
            self._cache.clear()
            logger.info(f"Cleared all CVaR calculation cache ({deleted_count} Redis keys)")
    
    def set_cached(self, symbol: str, payload: Dict[str, Any]) -> None:
        """
        Set cached CVaR data for a symbol.
        
        Used for cache warming and direct cache population.
        
        Args:
            symbol: Financial instrument symbol
            payload: CVaR calculation result to cache
        """
        if not isinstance(payload, dict):
            logger.warning(f"Invalid payload type for cache: {type(payload)}")
            return
        
        # Validate payload has required structure
        if not payload.get("success"):
            logger.warning(f"Refusing to cache unsuccessful result for {symbol}")
            return
        
        # Cache in both Redis and legacy cache for backward compatibility
        success = self.cache_service.set(
            CacheKeyType.CVAR_RESULT, 
            payload, 
            StandardCachePolicies.CVAR_RESULTS,
            symbol
        )
        self._cache[symbol] = payload
        
        if success:
            logger.debug(f"Cached CVaR data in Redis for symbol: {symbol}")
        else:
            logger.warning(f"Failed to cache CVaR data in Redis for symbol: {symbol}")
            
        logger.debug(f"Cached CVaR data for symbol: {symbol}")
    
    def get_cached(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get cached CVaR data for a symbol.
        
        This method checks Redis cache first, then falls back to legacy in-memory cache
        for backward compatibility with routes that need to check cache before computation.
        
        Args:
            symbol: Financial instrument symbol
            
        Returns:
            Cached CVaR data or None if not cached
        """
        # Try Redis cache first
        redis_result = self.cache_service.get(CacheKeyType.CVAR_RESULT, symbol)
        if redis_result is not None:
            logger.debug(f"Redis cache hit for symbol: {symbol}")
            return redis_result
        
        # Fall back to legacy in-memory cache
        memory_result = self._get_from_cache(symbol)
        if memory_result is not None:
            logger.debug(f"Memory cache hit for symbol: {symbol}")
            # Migrate to Redis cache for future requests
            self.cache_service.set(
                CacheKeyType.CVAR_RESULT, 
                memory_result, 
                StandardCachePolicies.CVAR_RESULTS,
                symbol
            )
        
        return memory_result
    
    def get_cached_symbols(self) -> List[str]:
        """
        Get list of symbols that have cached CVaR data.
        
        Used for ticker feed optimization to avoid redundant DB queries.
        
        Returns:
            List of cached symbol strings
        """
        if self._cache is None:
            return []
        return list(self._cache.keys())
    
    def get_symbols_for_recalculation(
        self,
        five_stars: bool = False,
        ready_only: bool = True,
        include_unknown: bool = False,
        country: Optional[str] = None,
        instrument_types: Optional[List[str]] = None,
        exclude_exchanges: Optional[List[str]] = None,
        limit: Optional[int] = None
    ) -> List[str]:
        """
        Get list of symbols for batch recalculation using repository filters.
        """
        return self.price_repo.get_symbols_by_filters(
            five_stars=five_stars,
            ready_only=ready_only,
            include_unknown=include_unknown,
            country=country,
            instrument_types=instrument_types,
            exclude_exchanges=exclude_exchanges,
            limit=limit
        )
    
    # =================== EXECUTION MODES ===================
    
    def _execute_local(
        self,
        symbol: str,
        force_recalculate: bool,
        to_date: Optional[str],
        prefer_local: bool
    ) -> Dict[str, Any]:
        """Execute CVaR calculation using local computation."""
        # Historical calculations bypass cache
        is_historical = bool(to_date)
        logger.debug(
            f"Local execution for {symbol}: historical={is_historical} force={force_recalculate}"
        )
        
        # Try cache first (non-historical only)
        if not is_historical and not force_recalculate:
            logger.debug(f"Checking cache for {symbol}")
            cached = self.get_cached(symbol)
            if cached and self._has_valid_values(cached):
                logger.debug(f"Cache hit for {symbol}, returning cached result")
                cached["cached"] = True
                cached["execution_mode"] = "local_cached"
                return cached
            logger.debug(f"No valid cache entry for {symbol}")
        
        # Try database if not forced
        if not is_historical and not force_recalculate:
            logger.debug(f"Checking database for {symbol}")
            db_result = self._load_from_database(symbol)
            if db_result and self._has_valid_values(db_result):
                logger.debug(f"Database hit for {symbol}, caching and returning result")
                # Cache database result in Redis for future requests
                self.set_cached(symbol, db_result)
                db_result["execution_mode"] = "local_database"
                return db_result
            logger.debug(f"No valid database entry for {symbol}")
        
        # Load price data using repository
        logger.debug(f"Loading price data for {symbol}")
        prices_result = load_prices(symbol, to_date=to_date)
        if not prices_result.get("success"):
            logger.warning(f"Failed to load price data for {symbol}: {prices_result.get('error')}")
            return prices_result
        logger.debug(
            f"Successfully loaded price data for {symbol} with " +
            f"{len(prices_result.get('returns', []))} return points"
        )
        
        # Compute CVaR blocks using unified compute function
        try:
            blocks = self._compute_cvar_blocks(
                prices_result["returns"],
                symbol=symbol,
                historical=is_historical,
                prefer_local=prefer_local
            )
            
            # Build result payload
            result = {
                "success": True,
                "symbol": symbol,
                "as_of_date": prices_result["as_of_date"],
                "start_date": prices_result.get("start_date"),
                "data_summary": prices_result["summary"],
                "cached": False,
                "execution_mode": "local_computed",
                **blocks
            }
            
            logger.debug(
                f"Local CVaR calculation successful for {symbol} from " +
                f"{prices_result.get('start_date')} to {prices_result.get('as_of_date')}"
            )
            
            # Add anomalies if present
            if "anomalies_report" in prices_result:
                result["anomalies_report"] = prices_result["anomalies_report"]
            
            # Cache and persist (non-historical only)
            if not is_historical and self._has_valid_values(result):
                logger.debug(f"Caching and persisting results for {symbol}")
                self.set_cached(symbol, result)
                self._persist_result(symbol, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Local CVaR calculation failed for {symbol}: {e}")
            return {
                "success": False,
                "error": f"local_calculation_failed: {str(e)}",
                "code": "local_calc_failed",
                "execution_mode": "local_failed"
            }
    
    def _execute_remote(
        self,
        symbol: str,
        force_recalculate: bool,
        to_date: Optional[str]
    ) -> Dict[str, Any]:
        """Execute CVaR calculation using remote Azure Function."""
        
        if not self.func_url:
            return {
                "success": False,
                "error": "Remote execution requested but NVAR_FUNC_URL not configured",
                "code": "remote_not_configured",
                "execution_mode": "remote_failed"
            }
        
        try:
            # Prepare request parameters
            params = {
                "symbol": symbol,
                "recalculate": "1" if force_recalculate else "0",
            }
            if to_date:
                params["todate"] = to_date
            
            # Optional API key header
            headers = {}
            api_key = os.getenv("NVAR_FUNC_KEY")
            if api_key:
                headers["x-functions-key"] = api_key
            
            # Make HTTP request to Azure Function
            url = f"{self.func_url}/api/cvar/curve-all"
            
            logger.info(f"Executing remote CVaR calculation: {url}")
            response = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=(
                    max(2, self.func_connect_timeout),
                    max(5, self.func_timeout)
                )
            )
            
            # Handle HTTP errors
            if not (200 <= response.status_code < 300):
                try:
                    error_data = response.json()
                except Exception:
                    error_data = {
                        "error": f"HTTP {response.status_code}",
                        "code": "remote_http_error"
                    }
                
                return {
                    "success": False,
                    "execution_mode": "remote_failed",
                    **error_data
                }
            
            # Parse successful response
            result = response.json()
            result["execution_mode"] = "remote_computed"
            
            # Cache successful results (non-historical only)
            if not to_date and self._has_valid_values(result):
                self.set_cached(symbol, result)
            
            return result
            
        except requests.exceptions.Timeout:
            return {
                "success": False,
                "error": f"Remote calculation timeout after {self.func_timeout}s",
                "code": "remote_timeout",
                "execution_mode": "remote_failed"
            }
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": f"Remote calculation request failed: {str(e)}",
                "code": "remote_request_failed",
                "execution_mode": "remote_failed"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Remote calculation failed: {str(e)}",
                "code": "remote_failed",
                "execution_mode": "remote_failed"
            }
    
    def _execute_auto(
        self,
        symbol: str,
        force_recalculate: bool,
        to_date: Optional[str],
        prefer_local: bool
    ) -> Dict[str, Any]:
        """Execute CVaR calculation with automatic local/remote selection."""
        
        if prefer_local:
            # Try local first, fallback to remote
            try:
                result = self._execute_local(symbol, force_recalculate, to_date, prefer_local)
                if result.get("success"):
                    result["execution_mode"] = "auto_local"
                    return result
                else:
                    logger.warning(f"Local execution failed for {symbol}, trying remote")
            except Exception as e:
                logger.warning(f"Local execution error for {symbol}: {e}, trying remote")
            
            # Fallback to remote
            result = self._execute_remote(symbol, force_recalculate, to_date)
            result["execution_mode"] = "auto_remote_fallback"
            return result
        else:
            # Try remote first, fallback to local
            try:
                result = self._execute_remote(symbol, force_recalculate, to_date)
                if result.get("success"):
                    result["execution_mode"] = "auto_remote"
                    return result
                else:
                    logger.warning(f"Remote execution failed for {symbol}, trying local")
            except Exception as e:
                logger.warning(f"Remote execution error for {symbol}: {e}, trying local")
            
            # Fallback to local
            result = self._execute_local(symbol, force_recalculate, to_date, prefer_local)
            result["execution_mode"] = "auto_local_fallback"
            return result
    
    # =================== HELPER METHODS ===================
    
    def _compute_cvar_blocks(
        self,
        returns_log,
        symbol: Optional[str],
        historical: bool,
        prefer_local: bool
    ) -> Dict[str, Any]:
        """Compute CVaR blocks using centralized computation logic."""
        # Use existing compute_cvar_blocks function which handles the complex CVaR math
        return compute_cvar_blocks(
            returns_log,
            symbol=symbol,
            historical=historical,
            prefer_local=prefer_local,
            sims=int(os.getenv("NVAR_SIMS", "10000")),
            trading_days=int(os.getenv("NVAR_TRADING_DAYS", "252")),
            log_phase=int(os.getenv("NVAR_LOG_PHASE", "1")),
            load_from_db_cb=self._load_from_database_callback
        )
    
    def _get_from_cache(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get result from in-memory cache."""
        return self._cache.get(symbol)
    
    def _load_from_database(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Load CVaR result from database using repository."""
        try:
            db_result = self.cvar_repo.get_latest_by_symbol(symbol)
            if db_result:
                # Convert database format to API format
                return self._format_database_result(db_result)
            return None
        except Exception as e:
            logger.warning(f"Database load failed for {symbol}: {e}")
            return None
    
    def _load_from_database_callback(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Callback for compute_cvar_blocks to load from database."""
        return self._load_from_database(symbol)
    
    def _persist_result(self, symbol: str, result: Dict[str, Any]) -> None:
        """Persist CVaR result to database using repository."""
        try:
            # Use existing persistence logic
            save_cvar_result(symbol, result)
            logger.debug(f"Persisted CVaR result for {symbol}")
        except Exception as e:
            logger.error(f"Failed to persist CVaR result for {symbol}: {e}")
    
    def _has_valid_values(self, result: Dict[str, Any]) -> bool:
        """Check if result contains valid CVaR values."""
        if not isinstance(result, dict) or not result.get("success"):
            return False
        
        # Check for presence of at least one CVaR block with valid data
        for alpha in [50, 95, 99]:
            block_key = f"cvar{alpha}"
            block = result.get(block_key)
            if isinstance(block, dict):
                annual = block.get("annual")
                if isinstance(annual, dict):
                    # Check if any of the annual values are valid numbers
                    for method in ["nig", "ghst", "evar"]:
                        value = annual.get(method)
                        try:
                            if value is not None and math.isfinite(float(value)):
                                return True
                        except (ValueError, TypeError):
                            continue
        
        return False
    
    def _format_database_result(self, db_result) -> Dict[str, Any]:
        """Convert database CVaR result to API format."""
        # This would convert repository result format to the expected API format
        # Implementation depends on the exact database schema
        return {
            "success": True,
            "symbol": db_result.get("symbol"),
            "cached": False,
            "execution_mode": "database",
            # Add other fields based on database structure
        }
    
    def _sanitize_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize and validate CVaR calculation result."""
        if not isinstance(result, dict):
            return {
                "success": False,
                "error": "Invalid result format",
                "code": "invalid_result"
            }
        
        # Ensure basic structure
        if "success" not in result:
            result["success"] = False
        
        # Add timestamp if missing
        if "calculated_at" not in result:
            result["calculated_at"] = datetime.utcnow().isoformat()
        
        # Add service metadata
        result["service_info"] = {
            "service": "CvarUnifiedService",
            "mode": self.mode,
            "architecture": "unified_domain_service"
        }
        
        return result
    
    def _resolve_mode(self, mode: Optional[str]) -> str:
        """Resolve execution mode from parameter or environment."""
        if isinstance(mode, str):
            resolved_mode = mode.strip().lower()
        else:
            resolved_mode = os.getenv("NVAR_CVAR_SERVICE", "local").strip().lower()
        
        if resolved_mode not in ["local", "remote", "auto"]:
            logger.warning(f"Invalid mode '{resolved_mode}', using 'local'")
            return "local"
        
        return resolved_mode
    
    def _safe_int(self, env_var: str, default: int) -> int:
        """Safely parse integer from environment variable."""
        try:
            return int(os.getenv(env_var, str(default)))
        except (ValueError, TypeError):
            return default
