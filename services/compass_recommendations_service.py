"""
This module contains the Compass Recommendations Service.

It implements the recommendation algorithm using CVaR data and proper risk
assessment to provide personalized financial product recommendations based on
user loss tolerance.
"""

import logging
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import and_, func  # type: ignore

from core.db import get_db_session
from core.models import CvarSnapshot, Symbols
from services.compass_secure import get_secure_compass

_LOG = logging.getLogger(__name__)

# Set up detailed logging
_LOG.setLevel(logging.DEBUG)
if not _LOG.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(
        logging.Formatter("%(asctime)s COMPASS_RECOMMEND %(levelname)s: %(message)s")
    )
    _LOG.addHandler(_h)


@dataclass
class SurvivorData:
    """Data for a product that passes the customer standard."""

    symbol: str
    mu: Optional[float]  # Expected return
    L: Optional[float]  # Risk measure (worst CVaR)
    name: Optional[str] = None
    instrument_type: Optional[str] = None
    exchange: Optional[str] = None


@dataclass
class RecommendationResult:
    """A product recommendation with computed Compass Score."""

    symbol: str
    name: Optional[str]
    type: Optional[str]
    country: Optional[str] = None
    currency: Optional[str] = None
    compass_score: int = 0
    nirvana_standard_pass: bool = False
    annualized_return: Optional[float] = None
    mu: Optional[float] = None
    L: Optional[float] = None
    R: Optional[float] = None
    S: Optional[float] = None


@dataclass(frozen=True)
class RecommendationConfig:
    """Non-proprietary recommendation configuration."""
    min_score_threshold: int = 3000
    max_results: int = 10
    # Standard statistical percentiles (non-proprietary)
    winsor_p_low: float = 0.01
    winsor_p_high: float = 0.99
    # Alpha confidence level for CVaR calculations (99% confidence)
    alpha_label: int = 99


class CompassRecommendationsService:
    """Service implementing secure recommendation algorithm."""

    def __init__(self):
        self.config = RecommendationConfig()
        self.compass = get_secure_compass()

    def get_recommendations(
        self,
        loss_tolerance_pct: float,
        country: str = "US",
        seed_symbol: Optional[str] = None,
    ) -> dict[str, Any]:
        """Get recommendations using Arman algorithm."""
        try:
            # Step 1: Get survivors (products passing customer standard)
            _LOG.info(
                "Starting recommendations: tolerance=%s%% country=%s seed_symbol=%s",
                loss_tolerance_pct,
                country,
                seed_symbol or "None"
            )

            # Convert percentage to decimal for internal calculations
            tolerance_decimal = loss_tolerance_pct / 100.0
            _LOG.debug(f"Converting tolerance {loss_tolerance_pct}%% to decimal: {tolerance_decimal}")
            survivors = self._get_survivors(tolerance_decimal, country)

            if not survivors:
                _LOG.warning(
                    f"No survivors found for tolerance={loss_tolerance_pct}% country={country}"
                )
                return {
                    "recommendations": [],
                    "metadata": {
                        "algorithm": "arman_correct",
                        "loss_tolerance": loss_tolerance_pct,
                        "country": country,
                        "survivors_count": 0,
                        "error": "No products pass the customer standard",
                    },
                }

            _LOG.info("Found %d survivors for %s", len(survivors), country)
            
            # Log first 5 survivors for debugging
            sample_survivors = survivors[:5]
            _LOG.debug(
                "Sample survivors: %s", 
                json.dumps([{"symbol": s.symbol, "mu": s.mu, "L": s.L} for s in sample_survivors])
            )

            # Steps 2-3: Get anchors and calculate scores
            _LOG.info("Getting anchor values for country %s", country)
            anchors = self._winsorize_and_recalibrate(survivors, country)
            _LOG.info(
                "Anchor values: mu_low=%.4f mu_high=%.4f median=%.4f", 
                anchors[0], anchors[1], anchors[2]
            )
            
            _LOG.info("Calculating Compass Scores for %d survivors", len(survivors))
            recommendations = self._calculate_scores(
                survivors, anchors, tolerance_decimal
            )

            # Step 4: Sort and limit results
            recommendations.sort(
                key=lambda x: (-x.compass_score, -(x.S or 0), -(x.R or 0))
            )

            # Apply result limit
            max_results = self.config.max_results
            recommendations = recommendations[:max_results]

            # Step 5: Build response
            response = {
                "recommendations": [self._to_dict(r) for r in recommendations],
                "metadata": {
                    "algorithm": "arman_correct",
                    "loss_tolerance": loss_tolerance_pct,
                    "country": country,
                    "survivors_count": len(survivors),
                    "returned_count": len(recommendations),
                    "anchors": {
                        "mu_low": f"{anchors[0]:.1%}",
                        "mu_high": f"{anchors[1]:.1%}",
                        "median_mu": f"{anchors[2]:.1%}",
                    },
                    "score_range": {
                        "min": min(r.compass_score for r in recommendations)
                        if recommendations
                        else None,
                        "max": max(r.compass_score for r in recommendations)
                        if recommendations
                        else None,
                    },
                    "timestamp": datetime.utcnow().isoformat(),
                },
            }

            _LOG.info(
                "Recommendations complete: returned %d results",
                len(recommendations),
            )
            return response

        except Exception as e:
            _LOG.exception("Error in get_recommendations")
            return {
                "recommendations": [],
                "metadata": {
                    "algorithm": "arman_correct",
                    "loss_tolerance": loss_tolerance_pct,
                    "country": country,
                    "survivors_count": 0,
                    "error": f"Algorithm error: {str(e)}",
                },
            }

        finally:
            # Clean up any remaining database resources
            try:
                pass
            except Exception:
                pass

    def _get_survivors(
        self, tolerance: float, country: str
    ) -> List[SurvivorData]:
        """Step 1: Apply customer standard - get surviving products."""
        _LOG.debug(f"Getting survivors with tolerance={tolerance} country={country}")
        sess = get_db_session()
        if sess is None:
            _LOG.error("Database session is None - cannot query survivors")
            return []

        try:
            # Get latest snapshots per symbol
            latest_per_symbol = (
                sess.query(
                    CvarSnapshot.symbol.label("symbol"),
                    func.max(CvarSnapshot.as_of_date).label("mx"),
                )
                .filter(CvarSnapshot.alpha_label == self.config.alpha_label)
                .group_by(CvarSnapshot.symbol)
                .subquery()
            )

            # Query with filters
            _LOG.debug(f"Building survivor query with alpha={self.config.alpha_label}")
            q = (
                sess.query(CvarSnapshot, Symbols)
                .join(
                    latest_per_symbol,
                    and_(
                        CvarSnapshot.symbol == latest_per_symbol.c.symbol,
                        CvarSnapshot.as_of_date == latest_per_symbol.c.mx,
                    ),
                )
                .outerjoin(
                    Symbols, Symbols.symbol == CvarSnapshot.symbol
                )
                .filter(CvarSnapshot.alpha_label == self.config.alpha_label)
                .filter(Symbols.country == country)
                .filter(Symbols.valid == 1)  # Only use valid products
            )
            
            _LOG.debug("Applied base filters: alpha=%s country=%s valid=1", 
                      self.config.alpha_label, country)

            # Count total valid products for country before filtering
            try:
                total_valid_count = sess.query(Symbols).filter(
                    Symbols.country == country,
                    Symbols.valid == 1
                ).count()
                _LOG.debug(f"Found {total_valid_count} total valid products for {country} before additional filters")
            except Exception as e:
                _LOG.warning(f"Error counting valid products: {e}")
                
            # REMOVED five_stars filter - using Harvard Release instead
            if country == "US":
                _LOG.debug("Using Harvard Release for US products (no five_stars filter)")
                # Не применяем фильтр five_stars == 1, используем только Harvard Release
                # который обеспечивается через valid == 1 и другие фильтры

            # Filter by instrument type based on validated universe
            # US: ETF, Mutual Fund, Common Stock (Non PINK Exchange)
            # UK: ETF, Common Stock
            # Canada: ETF
            type_lc = func.lower(Symbols.instrument_type)
            if country.upper() == "US":
                # US: ETF, Mutual Fund, Common Stock (excluding PINK exchange)
                allowed_types = [
                    "etf", "mutual fund", "mutual_fund", "fund", "common stock"
                ]
                
                # Count before instrument type filter
                try:
                    before_count = q.count()
                    _LOG.debug(f"Before US instrument type filter: {before_count} potential products")
                except Exception as e:
                    _LOG.warning(f"Error counting before instrument filter: {e}")
                
                q = q.filter(type_lc.in_(allowed_types))
                
                # Count after instrument type filter
                try:
                    after_count = q.count()
                    _LOG.debug(f"After instrument type filter: {after_count} products (removed {before_count - after_count})")
                except Exception as e:
                    _LOG.warning(f"Error counting after instrument filter: {e}")
                
                # Exclude PINK exchange
                before_count = after_count
                q = q.filter(func.upper(Symbols.exchange) != "PINK")
                
                # Count after PINK exchange filter
                try:
                    after_count = q.count()
                    _LOG.debug(f"After excluding PINK exchange: {after_count} products (removed {before_count - after_count})")
                except Exception as e:
                    _LOG.warning(f"Error counting after PINK filter: {e}")
                    
                _LOG.debug(f"Applied US-specific filters: type in {allowed_types}, exchange != PINK")
            elif country.upper() == "UK":
                # UK: ETF, Common Stock
                _LOG.debug("Applying UK-specific instrument type filters: ['etf', 'common stock']")
                q = q.filter(type_lc.in_(["etf", "common stock"]))
                
                # Special debug for UK products
                _LOG.debug("Checking UK products in database:")
                try:
                    uk_products = sess.query(Symbols).filter(Symbols.country == "UK").filter(Symbols.valid == 1).all()
                    _LOG.debug(f"Found {len(uk_products)} valid UK products in database")
                    if len(uk_products) > 0:
                        _LOG.debug(f"Sample UK products: {', '.join([p.symbol for p in uk_products[:5]])}")
                        
                        # Check CVaR values for UK products
                        sample_symbols = [p.symbol for p in uk_products[:5]]
                        cvar_values = sess.query(CvarSnapshot).filter(CvarSnapshot.symbol.in_(sample_symbols)).all()
                        _LOG.debug(f"Found {len(cvar_values)} CVaR records for sample UK products")
                except Exception as e:
                    _LOG.warning(f"Failed to check UK product availability: {e}")
            elif country.upper() in ["CA", "CANADA"]:
                # Canada: ETF only
                q = q.filter(type_lc.in_(["etf"]))
            else:
                # Default fallback for other countries
                allowed_types = [
                    "etf", "mutual fund", "mutual_fund", "fund", "common stock"
                ]
                q = q.filter(type_lc.in_(allowed_types))

            # Apply customer standard (loss tolerance filter)
            # Note: CVaR values in DB are stored as positive numbers representing losses
            # But tolerance is provided as negative (e.g., -40%), so we need to compare with abs(tolerance)
            worst_expr = func.greatest(
                CvarSnapshot.cvar_nig,
                CvarSnapshot.cvar_ghst,
                CvarSnapshot.cvar_evar,
            )
            # Convert negative tolerance to positive for comparison
            abs_tolerance = abs(tolerance)
            
            # Count before risk filter
            try:
                before_count = q.count()
                _LOG.debug(f"Before risk filter: {before_count} potential products")
            except Exception as e:
                _LOG.warning(f"Error counting before risk filter: {e}")
            
            q = q.filter(worst_expr <= abs_tolerance)
            
            # Log this filter application
            _LOG.debug(
                "Applied corrected risk filter: max(cvar_nig, cvar_ghst, cvar_evar) <= %f (absolute value of %f)",
                abs_tolerance, tolerance
            )
            
            # Count after risk filter
            try:
                after_count = q.count()
                _LOG.debug(f"After risk filter: {after_count} products (removed {before_count - after_count})")
                
                # If almost all products were filtered out by risk, get a sample of what was excluded
                if after_count < 10 and before_count > after_count + 10:
                    # Sample query without risk filter to see what's being excluded
                    sample_without_risk = q.filter(worst_expr > abs_tolerance).limit(5).all()
                    if sample_without_risk:
                        _LOG.debug("Sample products that failed the risk filter:")
                        for cs, ps in sample_without_risk:
                            worst_cvar = max(filter(
                                lambda x: x is not None,
                                [cs.cvar_nig, cs.cvar_ghst, cs.cvar_evar]
                            ))
                            _LOG.debug(f"  - {cs.symbol}: worst_cvar={worst_cvar:.4f}, tolerance={abs_tolerance:.4f}, return={cs.return_annual:.4f}")
            except Exception as e:
                _LOG.warning(f"Error in risk filter analysis: {e}")
            
            # Debug minimum CVaR values
            if country.upper() == "UK":
                try:
                    # Check minimum CVaR values in the database
                    min_query = sess.query(
                        func.min(CvarSnapshot.cvar_nig).label("min_nig"),
                        func.min(CvarSnapshot.cvar_ghst).label("min_ghst"), 
                        func.min(CvarSnapshot.cvar_evar).label("min_evar")
                    ).join(
                        Symbols, Symbols.symbol == CvarSnapshot.symbol
                    ).filter(
                        Symbols.country == "UK",
                        Symbols.valid == 1,
                        CvarSnapshot.alpha_label == self.config.alpha_label
                    )
                    min_values = min_query.first()
                    if min_values:
                        _LOG.debug(
                            f"UK minimum CVaR values - NIG: {min_values.min_nig}, " +
                            f"GHST: {min_values.min_ghst}, EVAR: {min_values.min_evar}"
                        )
                        _LOG.debug(
                            f"Current tolerance: {tolerance} (as absolute value: {abs(tolerance)}) - " +
                            f"Values must be <= {abs(tolerance)} to pass the filter"
                        )
                except Exception as e:
                    _LOG.warning(f"Failed to query minimum CVaR values: {e}")

            # Execute query
            _LOG.debug("Executing survivor query...")
            rows = q.all()
            _LOG.debug("Query returned %d potential survivors", len(rows))
            survivors = []

            for cvar_row, price_row in rows:
                try:
                    # Get the worst CVaR (highest loss)
                    worst_cvar = max(
                        filter(
                            lambda x: x is not None,
                            [
                                cvar_row.cvar_nig,
                                cvar_row.cvar_ghst,
                                cvar_row.cvar_evar,
                            ],
                        )
                    )

                    survivor = SurvivorData(
                        symbol=cvar_row.symbol,
                        mu=cvar_row.return_annual,
                        L=worst_cvar,
                        name=(
                            price_row.name if price_row else cvar_row.symbol
                        ),
                        instrument_type=(
                            price_row.instrument_type if price_row else None
                        ),
                        exchange=price_row.exchange if price_row else None,
                    )
                    survivors.append(survivor)

                except (TypeError, ValueError) as e:
                    _LOG.warning(
                        "Skipping invalid survivor %s: %s",
                        cvar_row.symbol,
                        str(e),
                    )
                    continue
            
            _LOG.debug("Final survivor count after validation: %d", len(survivors))

            return survivors

        finally:
            try:
                sess.close()
            except Exception:
                pass

    def _winsorize_and_recalibrate(
        self, survivors: List[SurvivorData], country: str
    ) -> Tuple[float, float, float, float, float]:
        """Steps 2-3: Use configured anchors or fallback to absolute."""
        _LOG.debug(f"Winsorizing and recalibrating for country={country}")
        if not survivors:
            _LOG.error("No survivors to winsorize")
            raise ValueError("No survivors to winsorize")

        # Try to get properly calibrated anchors from database
        anchors = None

        # Priority 1: Country-specific calibrated anchors (e.g., GLOBAL:US)
        try:
            country_category = f"GLOBAL:{country.upper()}"
            _LOG.debug(f"Looking up country-specific anchors: {country_category}")
            anchors = self._get_anchors_from_db(country_category)
            if anchors:
                msg = "Found country-specific anchors: %s"
                _LOG.info(msg, country_category)
                _LOG.debug("Anchor details: %s", json.dumps(anchors))
        except Exception as e:
            _LOG.warning(f"Error getting country-specific anchors: {e}")

        # Priority 2: Global calibrated anchors (GLOBAL:ALL)
        if not anchors:
            try:
                anchors = self._get_anchors_from_db("GLOBAL:ALL")
                if anchors:
                    _LOG.info("Using global calibrated anchors: GLOBAL:ALL")
            except Exception:
                pass

        # Priority 3: Manual/absolute anchors (legacy)
        if not anchors:
            try:
                anchors = self._get_anchors_from_db("COMPASS_ABSOLUTE")
                if anchors:
                    msg = "Falling back to manual anchors: COMPASS_ABSOLUTE"
                    _LOG.warning(msg)
            except Exception:
                pass

        if anchors:
            mu_low, mu_high = anchors['mu_low'], anchors['mu_high']
            _LOG.info(
                "Using DB anchors: mu_low=%.1f%% mu_high=%.1f%% (category=%s)",
                mu_low * 100, mu_high * 100, anchors.get('category', 'N/A')
            )
        else:
            # Fallback to environment/defaults only if no DB anchors exist
            # Safe fallback anchors (non-proprietary)
            mu_low = 0.02   # 2% - conservative estimate
            mu_high = 0.18  # 18% - growth estimate
            msg = (
                "No DB anchors found! Using ENV/default anchors: "
                "mu_low=%.1f%% mu_high=%.1f%% (source=fallback)"
            )
            _LOG.warning(msg, mu_low * 100, mu_high * 100)

        # Calculate survivor-based statistics for metadata
        mu_values = [s.mu for s in survivors if s.mu is not None]
        mu_values_sorted = sorted(mu_values)
        n = len(mu_values_sorted)
        default_median = 0.08  # 8% - reasonable market median
        median_mu = mu_values_sorted[n // 2] if n > 0 else default_median

        # Use configuration for percentiles and minimum sample size
        min_sample = 10  # Minimum data points for calibration
        p1 = (mu_values_sorted[int(self.config.winsor_p_low * n)]
              if n >= min_sample else mu_low)
        p99 = (mu_values_sorted[int(self.config.winsor_p_high * n)]
               if n >= min_sample else mu_high)

        return (float(mu_low), float(mu_high), float(median_mu),
                float(p1), float(p99))

    def _get_anchors_from_db(self, category: str) -> Optional[Dict[str, Any]]:
        """Get the latest anchors from database for given category."""
        _LOG.debug(f"Getting anchors from database for category: {category}")
        try:
            from core.models import CompassAnchor
            sess = get_db_session()
            if sess is None:
                _LOG.error("Database session is None - cannot query anchors")
                return None

            anchor = (
                sess.query(CompassAnchor)
                .filter(CompassAnchor.category == category)
                .order_by(CompassAnchor.created_at.desc())
                .first()
            )

            if anchor:
                _LOG.debug(
                    f"Found anchor: category={anchor.category} version={anchor.version} " +
                    f"mu_low={anchor.mu_low} mu_high={anchor.mu_high}"
                )
                return {
                    'category': anchor.category,
                    'version': anchor.version,
                    'mu_low': float(anchor.mu_low),
                    'mu_high': float(anchor.mu_high),
                    'median_mu': (float(anchor.median_mu)
                                  if anchor.median_mu else None),
                }
            _LOG.debug(f"No anchors found for category: {category}")
            return None
        except Exception as e:
            _LOG.warning("Failed to get anchors from DB: %s", e)
            return None

    def _calculate_scores(
        self,
        survivors: List[SurvivorData],
        anchors: Tuple[float, float, float, float, float],
        tolerance: float
    ) -> List[RecommendationResult]:
        """Step 4: Calculate Compass Scores for all survivors."""
        mu_low, mu_high, _median_mu, _p1, _p99 = anchors
        _LOG.debug(f"Calculating scores with anchors: mu_low={mu_low:.4f} mu_high={mu_high:.4f}")
        # Convert to absolute value for score calculation
        abs_tolerance = abs(tolerance)
        _LOG.debug(f"Using tolerance: {tolerance:.4f} (converted to absolute for score calculation: {abs_tolerance:.4f})")
        results = []

        for survivor in survivors:
            if survivor.mu is None or survivor.L is None:
                continue

            try:
                # Calculate proprietary score using secure interface
                _LOG.debug(
                    f"Computing score for {survivor.symbol}: mu={survivor.mu:.4f} L={survivor.L:.4f}"
                )
                compass_score = self.compass.compute_score(
                    survivor.mu,
                    survivor.L,
                    mu_low,
                    mu_high,
                    abs_tolerance,  # Use absolute value for tolerance
                )
                _LOG.debug(f"Computed score for {survivor.symbol}: {compass_score}")

                # Get normalized components from breakdown (for sorting only)
                breakdown = self.compass.compute_score_with_breakdown(
                    survivor.mu,
                    survivor.L,
                    mu_low,
                    mu_high,
                    abs_tolerance,  # Use absolute value for tolerance
                )
                breakdown_R = breakdown.get("R", 0.0)
                breakdown_S = breakdown.get("S", 0.0)
                R = (float(breakdown_R)
                     if isinstance(breakdown_R, (int, float)) else 0.0)
                S = (float(breakdown_S)
                     if isinstance(breakdown_S, (int, float)) else 0.0)

                # Only include if score meets minimum threshold
                if compass_score >= self.config.min_score_threshold:
                    # Get country and currency from database if available
                    country = None
                    currency = None
                    try:
                        sess = get_db_session()
                        if sess:
                            ps = sess.query(Symbols).filter(Symbols.symbol == survivor.symbol).first()
                            if ps:
                                country = ps.country
                                currency = ps.currency
                            sess.close()
                    except Exception as e:
                        _LOG.warning(f"Failed to get country/currency for {survivor.symbol}: {e}")
                                        
                    result = RecommendationResult(
                        symbol=survivor.symbol,
                        name=survivor.name,
                        type=survivor.instrument_type,
                        country=country,
                        currency=currency,
                        compass_score=int(compass_score),
                        nirvana_standard_pass=True,
                        annualized_return=survivor.mu,
                        mu=survivor.mu,
                        L=survivor.L,
                        R=R,
                        S=S,
                    )
                    results.append(result)
                    _LOG.debug(
                        f"Added {survivor.symbol} to results with score={compass_score} " +
                        f"R={R:.4f} S={S:.4f}"
                    )
                else:
                    _LOG.debug(
                        f"Skipping {survivor.symbol}: score {compass_score} below threshold " +
                        f"{self.config.min_score_threshold}"
                    )

            except Exception as e:
                _LOG.warning(
                    "Failed to calculate score for %s: %s",
                    survivor.symbol,
                    str(e)
                )
                continue

        return results

    @staticmethod
    def _format_date_for_display(date_val: Any) -> str:
        """Format various date types for API display."""
        if date_val is None:
            return "N/A"

        if isinstance(date_val, str):
            return date_val

        if hasattr(date_val, "isoformat"):
            return date_val.isoformat()

        # Fallback for other date types
        return str(date_val)

    @staticmethod
    def _get_latest_date(*dates: Any) -> str:
        """Get the latest date from a list of dates."""
        valid_dates = []
        for date_val in dates:
            if date_val is None:
                continue
            if isinstance(date_val, str):
                try:
                    from datetime import datetime
                    parsed = datetime.fromisoformat(
                        date_val.replace("Z", "+00:00")
                    )
                    valid_dates.append(parsed)
                except ValueError:
                    continue
            elif hasattr(date_val, "isoformat"):
                valid_dates.append(date_val)

        if not valid_dates:
            return "N/A"

        return max(valid_dates).isoformat()

    @staticmethod
    def _to_dict(result: RecommendationResult) -> dict[str, Any]:
        """Convert result to dictionary for API response."""
        return {
            "symbol": result.symbol,
            "name": result.name,
            "type": result.type,
            "country": result.country if hasattr(result, 'country') else None,
            "currency": result.currency if hasattr(result, 'currency') else None,
            "compass_score": result.compass_score,
            "nirvana_standard_pass": result.nirvana_standard_pass,
            "annualized_return": result.annualized_return,
        }
