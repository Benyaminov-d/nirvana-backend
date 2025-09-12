from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import floor
import logging
from typing import Any

import numpy as np

from core.db import get_db_session
from core.models import CompassAnchor, CvarSnapshot
from core.models import Symbols  # type: ignore
from core.compass_config import AnchorCalibConfig
# Lazy import to avoid circular dependency
import os


_LOG = logging.getLogger("nirvana.compass")


# AnchorCalibConfig moved to core.compass_config


def _type7_quantile(sorted_x: list[float], p: float) -> float:
    n = len(sorted_x)
    if n == 0:
        return float("nan")
    h = (n - 1) * p + 1
    k = floor(h)
    g = h - k
    k0 = max(1, min(k, n))
    k1 = max(1, min(k + 1, n))
    return (1 - g) * sorted_x[k0 - 1] + g * sorted_x[k1 - 1]


def _harrell_davis(sorted_x: list[float], p: float) -> float:
    n = len(sorted_x)
    if n == 0:
        return float("nan")
    try:
        from scipy.stats import beta as _beta  # type: ignore

        a = (n + 1) * p
        b = (n + 1) * (1 - p)
        w = []
        for i in range(1, n + 1):
            w_i = _beta.cdf(i / n, a, b) - _beta.cdf((i - 1) / n, a, b)
            w.append(w_i)
        return float(
            sum(w_i * sorted_x[i - 1] for i, w_i in enumerate(w, start=1))
        )
    except Exception:
        return _type7_quantile(sorted_x, p)


def calibrate_mu_anchors(
    mu_values: list[float],
    cfg: AnchorCalibConfig,
) -> tuple[float, float, float, float, float]:
    xs = [float(x) for x in mu_values if x is not None and np.isfinite(x)]
    if len(xs) < 10:
        raise ValueError("Insufficient data for calibration (need >= 10)")
    xs.sort()
    p1 = _type7_quantile(xs, cfg.p_winsor_low)
    p99 = _type7_quantile(xs, cfg.p_winsor_high)
    w = [min(max(x, p1), p99) for x in xs]
    w.sort()
    mu_low = _harrell_davis(w, cfg.p_hd_low)
    mu_high = _harrell_davis(w, cfg.p_hd_high)
    med = _harrell_davis(w, 0.5)
    spread = mu_high - mu_low
    if spread < cfg.min_spread:
        mu_low = min(mu_low, med - cfg.min_spread / 2)
        mu_high = max(mu_high, med + cfg.min_spread / 2)
    if spread > cfg.max_spread:
        mu_low = max(mu_low, med - cfg.max_spread / 2)
        mu_high = min(mu_high, med + cfg.max_spread / 2)
    return float(mu_low), float(mu_high), float(med), float(p1), float(p99)


def current_quarter_version(now: datetime | None = None) -> str:
    dt = now or datetime.utcnow()
    q = ((dt.month - 1) // 3) + 1
    return f"{dt.year}Q{q}"


def auto_calibrate_from_db(
    category: str,
    cfg: AnchorCalibConfig | None = None,
) -> bool:
    cfg = cfg or AnchorCalibConfig.from_environment()
    sess = get_db_session()
    if sess is None:
        return False
    try:
        # Already have anchors for current quarter?
        ver = current_quarter_version()
        existing = (
            sess.query(CompassAnchor)
            .filter(
                CompassAnchor.category == category,
                CompassAnchor.version == ver,
            )  # type: ignore
            .one_or_none()
        )
        if existing is not None:
            try:
                _LOG.info(
                    "anchors present: category=%s version=%s", category, ver
                )
            except Exception:
                pass
            return False

        # Build universe: latest return_annual per symbol (any alpha label),
        # restricted to instruments with sufficient history only
        from sqlalchemy import and_, func  # type: ignore
        latest = (
            sess.query(
                CvarSnapshot.symbol.label("symbol"),
                func.max(CvarSnapshot.as_of_date).label("mx"),
            )
            .group_by(CvarSnapshot.symbol)
            .subquery()
        )
        q = (
            sess.query(CvarSnapshot)
            .join(
                latest,
                and_(
                    CvarSnapshot.symbol == latest.c.symbol,
                    CvarSnapshot.as_of_date == latest.c.mx,
                ),
            )
            .outerjoin(Symbols, Symbols.symbol == CvarSnapshot.symbol)
            .filter(Symbols.insufficient_history == 0)  # type: ignore
        )
        rows = q.all()
        mu_vals: list[float] = []
        for r in rows:
            try:
                if r.return_annual is not None:
                    mu_vals.append(float(r.return_annual))
            except Exception:
                continue
        if len(mu_vals) < 10:
            try:
                _LOG.warning(
                    "auto-calibration skipped: insufficient mu values "
                    "n=%d",
                    len(mu_vals),
                )
            except Exception:
                pass
            return False

        mu_low, mu_high, med, p1, p99 = calibrate_mu_anchors(mu_vals, cfg)
        rec = CompassAnchor(
            category=category,
            version=ver,
            mu_low=mu_low,
            mu_high=mu_high,
            median_mu=med,
            p1=p1,
            p99=p99,
            metadata_json={
                "winsor": [cfg.p_winsor_low, cfg.p_winsor_high],
                "hd": [cfg.p_hd_low, cfg.p_hd_high],
                "min_spread": cfg.min_spread,
                "max_spread": cfg.max_spread,
                "n": len(mu_vals),
                "source": "auto_startup",
            },
        )
        sess.add(rec)
        sess.commit()
        try:
            _LOG.info(
                (
                    "auto-calibrated anchors category=%s version=%s "
                    "mu_low=%.6f mu_high=%.6f"
                ),
                category,
                ver,
                mu_low,
                mu_high,
            )
        except Exception:
            pass
        return True
    except Exception:
        try:
            sess.rollback()
        except Exception:
            pass
        return False
    finally:
        try:
            sess.close()
        except Exception:
            pass


# Global calibration: builds anchors over a unified universe, optionally
# filtered by countries and instrument types from env variables.
#
# ENV (optional):
# - COMPASS_ANCHOR_COUNTRIES: comma-separated (e.g., "US,CA")
# - COMPASS_ANCHOR_TYPES: comma-separated (e.g., "Common Stock,ETF,Fund")
def auto_calibrate_global_from_db(
    cfg: AnchorCalibConfig | None = None,
) -> bool:
    cfg = cfg or AnchorCalibConfig.from_environment()
    sess = get_db_session()
    if sess is None:
        return False
    try:
        ver = current_quarter_version()
        existing = (
            sess.query(CompassAnchor)
            .filter(
                CompassAnchor.category == "GLOBAL",
                CompassAnchor.version == ver,
            )  # type: ignore
            .one_or_none()
        )
        if existing is not None:
            try:
                _LOG.info("anchors present: category=GLOBAL version=%s", ver)
            except Exception:
                pass
            return False

        # Build latest-per-symbol snapshot, then join to Symbols
        from sqlalchemy import and_, func  # type: ignore

        latest = (
            sess.query(
                CvarSnapshot.symbol.label("symbol"),
                func.max(CvarSnapshot.as_of_date).label("mx"),
            )
            .group_by(CvarSnapshot.symbol)
            .subquery()
        )

        q = (
            sess.query(CvarSnapshot, Symbols)
            .join(
                latest,
                and_(
                    CvarSnapshot.symbol == latest.c.symbol,
                    CvarSnapshot.as_of_date == latest.c.mx,
                ),
            )
            .outerjoin(Symbols, Symbols.symbol == CvarSnapshot.symbol)
        )
        # Only include instruments with sufficient history and valid data
        q = q.filter(Symbols.insufficient_history == 0)  # type: ignore
        q = q.filter(Symbols.valid == 1)  # Only use valid products

        # Optional env filters
        def _parse_list(env_name: str) -> list[str]:
            try:
                raw = os.getenv(env_name, "") or ""
                vals = [v.strip() for v in raw.split(",") if v.strip()]
                return vals
            except Exception:
                return []

        countries = [v for v in _parse_list("COMPASS_ANCHOR_COUNTRIES")]
        types = [v for v in _parse_list("COMPASS_ANCHOR_TYPES")]
        if countries:
            q = q.filter(
                Symbols.country.in_(countries)
            )  # type: ignore
        if types:
            q = q.filter(
                Symbols.instrument_type.in_(types)
            )  # type: ignore

        rows = q.all()

        mu_vals: list[float] = []
        for r, _ps in rows:
            try:
                if r.return_annual is not None:
                    mu_vals.append(float(r.return_annual))
            except Exception:
                continue
        # Use global config for minimum sample size
        # Global config removed - using safe defaults only
        min_sample_size = 100  # Safe default instead of global_config
        
        if len(mu_vals) < min_sample_size:
            try:
                _LOG.warning(
                    "global calibration skipped: insufficient mu values n=%d",
                    len(mu_vals),
                )
            except Exception:
                pass
            return False

        mu_low, mu_high, med, p1, p99 = calibrate_mu_anchors(mu_vals, cfg)
        rec = CompassAnchor(
            category="GLOBAL",
            version=ver,
            mu_low=mu_low,
            mu_high=mu_high,
            median_mu=med,
            p1=p1,
            p99=p99,
            metadata_json={
                "winsor": [cfg.p_winsor_low, cfg.p_winsor_high],
                "hd": [cfg.p_hd_low, cfg.p_hd_high],
                "min_spread": cfg.min_spread,
                "max_spread": cfg.max_spread,
                "n": len(mu_vals),
                "source": "auto_startup_global",
                "countries": countries or None,
                "types": types or None,
            },
        )
        sess.add(rec)
        sess.commit()
        try:
            _LOG.info(
                (
                    "calibrated GLOBAL anchors version=%s "
                    "mu_low=%.6f mu_high=%.6f"
                ),
                ver,
                mu_low,
                mu_high,
            )
        except Exception:
            pass
        return True
    except Exception:
        try:
            sess.rollback()
        except Exception:
            pass
        return False
    finally:
        try:
            sess.close()
        except Exception:
            pass


def auto_calibrate_global_per_country_from_db(
    cfg: AnchorCalibConfig | None = None,
) -> dict:
    """Calibrate GLOBAL anchors separately for each country present in DB.

    Creates categories like GLOBAL:US, GLOBAL:Canada for the current quarter.
    Returns summary with counts per country.
    """
    cfg = cfg or AnchorCalibConfig()
    sess = get_db_session()
    summary: dict[str, int] = {"countries": 0, "calibrated": 0, "skipped": 0}
    if sess is None:
        return summary
    try:
        ver = current_quarter_version()
        # Distinct non-null countries from Symbols
        from sqlalchemy import func  # type: ignore

        countries: list[str] = []
        for (c,) in (
            sess.query(func.distinct(Symbols.country))
            .filter(Symbols.country.isnot(None))  # type: ignore
            .all()
        ):
            if c:
                countries.append(str(c))
        summary["countries"] = len(countries)

        if not countries:
            try:
                _LOG.info("global-per-country: no countries found; skipping")
            except Exception:
                pass
            return summary

        from sqlalchemy import and_  # type: ignore

        # Latest per symbol subquery reused across countries
        latest = (
            sess.query(
                CvarSnapshot.symbol.label("symbol"),
                func.max(CvarSnapshot.as_of_date).label("mx"),
            )
            .group_by(CvarSnapshot.symbol)
            .subquery()
        )

        for co in countries:
            cat = f"GLOBAL:{co}"
            existing = (
                sess.query(CompassAnchor)
                .filter(
                    CompassAnchor.category == cat,
                    CompassAnchor.version == ver,
                )  # type: ignore
                .one_or_none()
            )
            if existing is not None:
                summary["skipped"] += 1
                try:
                    _LOG.info(
                        "anchors present: category=%s version=%s",
                        cat,
                        ver,
                    )
                except Exception:
                    pass
                continue

            q = (
                sess.query(CvarSnapshot, Symbols)
                .join(
                    latest,
                    and_(
                        CvarSnapshot.symbol == latest.c.symbol,
                        CvarSnapshot.as_of_date == latest.c.mx,
                    ),
                )
                .outerjoin(
                    Symbols,
                    Symbols.symbol == CvarSnapshot.symbol,
                )
                .filter(Symbols.country == co)  # type: ignore
            )
            # Only include instruments with sufficient history
            q = q.filter(Symbols.insufficient_history == 0)  # type: ignore
            rows = q.all()

            mu_vals: list[float] = []
            for r, _ps in rows:
                try:
                    if r.return_annual is not None:
                        mu_vals.append(float(r.return_annual))
                except Exception:
                    continue
            if len(mu_vals) < 10:
                try:
                    _LOG.warning(
                        "global-per-country skipped: country=%s n=%d",
                        co,
                        len(mu_vals),
                    )
                except Exception:
                    pass
                continue

            mu_low, mu_high, med, p1, p99 = calibrate_mu_anchors(mu_vals, cfg)
            rec = CompassAnchor(
                category=cat,
                version=ver,
                mu_low=mu_low,
                mu_high=mu_high,
                median_mu=med,
                p1=p1,
                p99=p99,
                metadata_json={
                    "winsor": [cfg.p_winsor_low, cfg.p_winsor_high],
                    "hd": [cfg.p_hd_low, cfg.p_hd_high],
                    "min_spread": cfg.min_spread,
                    "max_spread": cfg.max_spread,
                    "n": len(mu_vals),
                    "source": "auto_startup_global_per_country",
                    "country": co,
                },
            )
            sess.add(rec)
            sess.commit()
            summary["calibrated"] += 1
            try:
                _LOG.info(
                    (
                        "calibrated anchors category=%s "
                        "mu_low=%.6f mu_high=%.6f"
                    ),
                    cat,
                    mu_low,
                    mu_high,
                )
            except Exception:
                pass

        return summary
    except Exception:
        try:
            sess.rollback()
        except Exception:
            pass
        return summary
    finally:
        try:
            sess.close()
        except Exception:
            pass


def auto_calibrate_by_type_country_from_db(
    cfg: AnchorCalibConfig | None = None,
) -> dict:
    """Calibrate anchors per (instrument_type, country) from DB.

    Stores categories as "<Type>:<Country>", e.g., "Fund:US".
    Returns summary.
    """
    cfg = cfg or AnchorCalibConfig()
    sess = get_db_session()
    summary: dict[str, int] = {
        "pairs": 0,
        "calibrated": 0,
        "skipped": 0,
    }
    if sess is None:
        return summary
    try:
        ver = current_quarter_version()
        # no special SQL functions needed here

        pairs: list[tuple[str, str]] = []
        for (t, c) in (
            sess.query(Symbols.instrument_type, Symbols.country)
            .filter(
                Symbols.country.isnot(None),  # type: ignore
                Symbols.instrument_type.isnot(None),  # type: ignore
            )
            .distinct()
            .all()
        ):
            if t and c:
                pairs.append((str(t), str(c)))
        summary["pairs"] = len(pairs)
        if not pairs:
            try:
                _LOG.info("by-type-country: no pairs found; skipping")
            except Exception:
                pass
            return summary

        from sqlalchemy import and_, func  # type: ignore

        latest = (
            sess.query(
                CvarSnapshot.symbol.label("symbol"),
                func.max(CvarSnapshot.as_of_date).label("mx"),
            )
            .group_by(CvarSnapshot.symbol)
            .subquery()
        )

        for typ, co in pairs:
            cat = f"{typ}:{co}"
            existing = (
                sess.query(CompassAnchor)
                .filter(
                    CompassAnchor.category == cat,
                    CompassAnchor.version == ver,
                )  # type: ignore
                .one_or_none()
            )
            if existing is not None:
                summary["skipped"] += 1
                try:
                    _LOG.info(
                        "anchors present: category=%s version=%s", cat, ver
                    )
                except Exception:
                    pass
                continue

            q = (
                sess.query(CvarSnapshot, Symbols)
                .join(
                    latest,
                    and_(
                        CvarSnapshot.symbol == latest.c.symbol,
                        CvarSnapshot.as_of_date == latest.c.mx,
                    ),
                )
                .outerjoin(
                    Symbols, Symbols.symbol == CvarSnapshot.symbol
                )
                .filter(
                    Symbols.country == co,  # type: ignore
                    Symbols.instrument_type == typ,  # type: ignore
                )
            )
            # Only include instruments with sufficient history
            q = q.filter(Symbols.insufficient_history == 0)  # type: ignore
            rows = q.all()

            mu_vals: list[float] = []
            for r, _ps in rows:
                try:
                    if r.return_annual is not None:
                        mu_vals.append(float(r.return_annual))
                except Exception:
                    continue
            if len(mu_vals) < 10:
                try:
                    _LOG.warning(
                        "by-type-country skipped: cat=%s n=%d",
                        cat,
                        len(mu_vals),
                    )
                except Exception:
                    pass
                continue

            mu_low, mu_high, med, p1, p99 = calibrate_mu_anchors(mu_vals, cfg)
            rec = CompassAnchor(
                category=cat,
                version=ver,
                mu_low=mu_low,
                mu_high=mu_high,
                median_mu=med,
                p1=p1,
                p99=p99,
                metadata_json={
                    "winsor": [cfg.p_winsor_low, cfg.p_winsor_high],
                    "hd": [cfg.p_hd_low, cfg.p_hd_high],
                    "min_spread": cfg.min_spread,
                    "max_spread": cfg.max_spread,
                    "n": len(mu_vals),
                    "source": "auto_startup_by_type_country",
                    "country": co,
                    "type": typ,
                },
            )
            sess.add(rec)
            sess.commit()
            summary["calibrated"] += 1
            try:
                _LOG.info(
                    "calibrated anchors category=%s mu_low=%.6f mu_high=%.6f",
                    cat,
                    mu_low,
                    mu_high,
                )
            except Exception:
                pass
        return summary
    except Exception:
        try:
            sess.rollback()
        except Exception:
            pass
        return summary
    finally:
        try:
            sess.close()
        except Exception:
            pass


def calibrate_validated_universe_anchors(
    cfg: AnchorCalibConfig | None = None,
) -> dict[str, Any]:
    """
    Calibrate anchors for the validated universe:
    - US: ETF, Mutual Fund, Common Stock (Non PINK Exchange)
    - UK: ETF, Common Stock
    - Canada: ETF
    
    Creates country-specific anchors: GLOBAL:US, GLOBAL:UK, GLOBAL:CA
    Returns summary of calibration results.
    """
    cfg = cfg or AnchorCalibConfig.from_environment()
    summary = {
        "US": {"status": "pending", "count": 0, "anchors": None},
        "UK": {"status": "pending", "count": 0, "anchors": None},
        "CA": {"status": "pending", "count": 0, "anchors": None},
    }
    
    # US calibration: ETF, Mutual Fund, Common Stock (Non PINK Exchange)
    us_result = _calibrate_country_specific(
        country="US", 
        instrument_types=["ETF", "Mutual Fund", "Common Stock"],
        exclude_exchanges=["PINK"],
        cfg=cfg
    )
    summary["US"] = us_result
    
    # UK calibration: ETF, Common Stock
    uk_result = _calibrate_country_specific(
        country="UK",
        instrument_types=["ETF", "Common Stock"],
        exclude_exchanges=None,
        cfg=cfg
    )
    summary["UK"] = uk_result
    
    # Canada calibration: ETF only
    ca_result = _calibrate_country_specific(
        country="CA",
        instrument_types=["ETF"],
        exclude_exchanges=None,
        cfg=cfg
    )
    summary["CA"] = ca_result
    
    return summary


def _calibrate_country_specific(
    country: str,
    instrument_types: list[str],
    exclude_exchanges: list[str] | None,
    cfg: AnchorCalibConfig,
) -> dict[str, Any]:
    """
    Calibrate anchors for a specific country with instrument type filters.
    """
    result = {
        "status": "failed",
        "count": 0,
        "anchors": None,
        "error": None
    }
    
    sess = get_db_session()
    if sess is None:
        result["error"] = "Database not available"
        return result
        
    try:
        from sqlalchemy import and_, func, or_  # type: ignore
        
        # Check if anchors already exist for current quarter
        ver = current_quarter_version()
        category = f"GLOBAL:{country.upper()}"
        existing = (
            sess.query(CompassAnchor)
            .filter(
                CompassAnchor.category == category,
                CompassAnchor.version == ver,
            )
            .one_or_none()
        )
        
        if existing is not None:
            result["status"] = "exists"
            result["anchors"] = {
                "mu_low": existing.mu_low,
                "mu_high": existing.mu_high,
                "median_mu": existing.median_mu
            }
            return result

        # Get latest CVaR snapshots per symbol
        latest = (
            sess.query(
                CvarSnapshot.symbol.label("symbol"),
                func.max(CvarSnapshot.as_of_date).label("mx"),
            )
            .group_by(CvarSnapshot.symbol)
            .subquery()
        )
        
        # Build query with filters - PRIORITY: use compass parameters data
        from core.models import CompassInputs
        
        # Try to use compass parameters data first (μ values with Level 1 winsorization)
        params_q = (
            sess.query(CompassInputs, Symbols)
            .join(Symbols, CompassInputs.instrument_id == Symbols.id)
            .filter(CompassInputs.category_id == country)  # Match category to country
            .filter(Symbols.country == country)
            .filter(Symbols.valid == 1)  # Only valid products
            .filter(Symbols.insufficient_history == 0)  # Sufficient history
        )
        
        # Fallback to CvarSnapshot if no clean data available
        q = (
            sess.query(CvarSnapshot, Symbols)
            .join(
                latest,
                and_(
                    CvarSnapshot.symbol == latest.c.symbol,
                    CvarSnapshot.as_of_date == latest.c.mx,
                ),
            )
            .outerjoin(Symbols, Symbols.symbol == CvarSnapshot.symbol)
            .filter(Symbols.country == country)
            .filter(Symbols.valid == 1)  # Only valid products
            .filter(Symbols.insufficient_history == 0)  # Sufficient history
        )
        
        # Apply filters to compass parameters query
        if instrument_types:
            type_filters = [
                func.lower(Symbols.instrument_type) == itype.lower() 
                for itype in instrument_types
            ]
            if len(type_filters) == 1:
                params_q = params_q.filter(type_filters[0])
            else:
                params_q = params_q.filter(or_(*type_filters))
        
        if exclude_exchanges:
            params_q = params_q.filter(
                ~func.upper(Symbols.exchange).in_(
                    [ex.upper() for ex in exclude_exchanges]
                )
            )
        
        # Try compass parameters data first
        params_rows = params_q.all()
        
        if params_rows:
            # SUCCESS: Use compass parameters data (μ values with Level 1 winsorization)
            mu_vals = []
            for params_row, price_row in params_rows:
                try:
                    if params_row.mu_i is not None:
                        mu_vals.append(float(params_row.mu_i))
                except (ValueError, TypeError):
                    continue
            
            result["data_source"] = "compass_inputs"  
            result["winsorization"] = "none at returns level (per specification)"
            _LOG.info(
                "Using compass parameters data for %s anchors: %d symbols",
                country, len(mu_vals)
            )
            
        else:
            # FALLBACK: Use CvarSnapshot data (no winsorization at returns level)
            # Apply same filters to fallback query
            if instrument_types:
                if len(type_filters) == 1:
                    q = q.filter(type_filters[0])
                else:
                    q = q.filter(or_(*type_filters))
            
            if exclude_exchanges:
                q = q.filter(
                    ~func.upper(Symbols.exchange).in_(
                        [ex.upper() for ex in exclude_exchanges]
                    )
                )
            
            # Execute fallback query
            rows = q.all()
            
            # Extract mu values from CvarSnapshot
            mu_vals = []
            for cvar_row, price_row in rows:
                try:
                    if cvar_row.return_annual is not None:
                        mu_vals.append(float(cvar_row.return_annual))
                except (ValueError, TypeError):
                    continue
            
            result["data_source"] = "cvar_snapshot"
            result["winsorization"] = "none at returns level"
            _LOG.warning(
                "FALLBACK to CvarSnapshot for %s anchors: %d symbols "
                "(clean timeseries not available)",
                country, len(mu_vals)
            )
        
        result["count"] = len(mu_vals)
        
        # Check minimum sample size
        # Global config removed - using safe defaults only
        min_sample_size = 100  # Safe default instead of global_config
        absolute_min_size = 5  # Absolute minimum to create any anchor
        
        if len(mu_vals) < min_sample_size:
            _LOG.warning(
                "Low sample size for %s anchors: %d < %d (recommended)",
                country, len(mu_vals), min_sample_size
            )
            
            # Still proceed if we have at least the absolute minimum
            if len(mu_vals) < absolute_min_size:
                result["error"] = f"Insufficient data: {len(mu_vals)} < {absolute_min_size} (absolute minimum)"
                return result
            
            result["warning"] = f"Low sample size: {len(mu_vals)} < {min_sample_size} (recommended)"
        
        # Calibrate anchors
        mu_low, mu_high, med, p1, p99 = calibrate_mu_anchors(mu_vals, cfg)
        
        # Save to database
        anchor_record = CompassAnchor(
            category=category,
            version=ver,
            mu_low=mu_low,
            mu_high=mu_high,
            median_mu=med,
            p1=p1,
            p99=p99,
            metadata_json={
                "instrument_types": instrument_types,
                "exclude_exchanges": exclude_exchanges or [],
                "sample_size": len(mu_vals),
                "calibration_date": datetime.utcnow().isoformat(),
                "config": {
                    "p_winsor_low": cfg.p_winsor_low,
                    "p_winsor_high": cfg.p_winsor_high,
                    "p_hd_low": cfg.p_hd_low,
                    "p_hd_high": cfg.p_hd_high,
                    "min_spread": cfg.min_spread,
                    "max_spread": cfg.max_spread,
                }
            }
        )
        
        sess.add(anchor_record)
        sess.commit()
        
        result["status"] = "calibrated"
        result["anchors"] = {
            "mu_low": mu_low,
            "mu_high": mu_high,
            "median_mu": med
        }
        
        try:
            _LOG.info(
                "Calibrated %s anchors: mu_low=%.1f%% mu_high=%.1f%% (n=%d)",
                category, mu_low * 100, mu_high * 100, len(mu_vals)
            )
        except Exception:
            pass
            
        return result
        
    except Exception as e:
        result["error"] = str(e)
        try:
            _LOG.error("Failed to calibrate %s anchors: %s", country, e)
        except Exception:
            pass
        return result
        
    finally:
        try:
            sess.close()
        except Exception:
            pass


def _calibrate_with_filters(
    *,
    category: str,
    countries: list[str] | None,
    types: list[str] | None,
    cfg: AnchorCalibConfig | None = None,
) -> bool:
    """Internal helper to calibrate anchors for an arbitrary filtered universe.

    Stores result under the provided category name (e.g., GLOBAL:ALL).
    """
    cfg = cfg or AnchorCalibConfig.from_environment()
    sess = get_db_session()
    if sess is None:
        return False
    try:
        ver = current_quarter_version()
        existing = (
            sess.query(CompassAnchor)
            .filter(
                CompassAnchor.category == category,
                CompassAnchor.version == ver,
            )  # type: ignore
            .one_or_none()
        )
        if existing is not None:
            return False

        from sqlalchemy import and_, func  # type: ignore
        latest = (
            sess.query(
                CvarSnapshot.symbol.label("symbol"),
                func.max(CvarSnapshot.as_of_date).label("mx"),
            )
            .group_by(CvarSnapshot.symbol)
            .subquery()
        )
        q = (
            sess.query(CvarSnapshot, Symbols)
            .join(
                latest,
                and_(
                    CvarSnapshot.symbol == latest.c.symbol,
                    CvarSnapshot.as_of_date == latest.c.mx,
                ),
            )
            .outerjoin(Symbols, Symbols.symbol == CvarSnapshot.symbol)
        )
        if countries:
            q = q.filter(
                Symbols.country.in_(countries)
            )  # type: ignore
        if types:
            q = q.filter(
                Symbols.instrument_type.in_(types)
            )  # type: ignore
        # Only include instruments with sufficient history and valid data
        q = q.filter(Symbols.insufficient_history == 0)  # type: ignore
        q = q.filter(Symbols.valid == 1)  # Only use valid products
        rows = q.all()

        mu_vals: list[float] = []
        for r, _ps in rows:
            try:
                if r.return_annual is not None:
                    mu_vals.append(float(r.return_annual))
            except Exception:
                continue
        # Use global config for minimum sample size
        # Global config removed - using safe defaults only
        min_sample_size = 100  # Safe default instead of global_config
        
        if len(mu_vals) < min_sample_size:
            return False

        mu_low, mu_high, med, p1, p99 = calibrate_mu_anchors(mu_vals, cfg)
        rec = CompassAnchor(
            category=category,
            version=ver,
            mu_low=mu_low,
            mu_high=mu_high,
            median_mu=med,
            p1=p1,
            p99=p99,
            metadata_json={
                "winsor": [cfg.p_winsor_low, cfg.p_winsor_high],
                "hd": [cfg.p_hd_low, cfg.p_hd_high],
                "min_spread": cfg.min_spread,
                "max_spread": cfg.max_spread,
                "n": len(mu_vals),
                "source": "calibrate_special_sets",
                "countries": countries or None,
                "types": types or None,
            },
        )
        sess.add(rec)
        sess.commit()
        return True
    except Exception:
        try:
            sess.rollback()
        except Exception:
            pass
        return False
    finally:
        try:
            sess.close()
        except Exception:
            pass


def calibrate_special_sets(cfg: AnchorCalibConfig | None = None) -> dict:
    """Create additional anchors per request:
    1) GLOBAL:ALL
    2) GLOBAL:ALL-MUTUAL-FUND-ETF (types Mutual Fund, ETF)
    3) GLOBAL:US-MUTUAL-FUND-ETF (US, types Mutual Fund, ETF)
    4) GLOBAL:Canada-MUTUAL-FUND-ETF (Canada, types Mutual Fund, ETF)
    """
    cfg = cfg or AnchorCalibConfig()
    summary = {"created": 0, "skipped": 0}

    def _run(
        category: str,
        countries: list[str] | None,
        types: list[str] | None,
    ) -> None:
        ok = _calibrate_with_filters(
            category=category,
            countries=countries,
            types=types,
            cfg=cfg,
        )
        if ok:
            summary["created"] += 1
        else:
            summary["skipped"] += 1

    _run("GLOBAL:ALL", None, None)
    _run("GLOBAL:ALL-MUTUAL-FUND-ETF", None, ["Mutual Fund", "ETF"])
    _run("GLOBAL:US-MUTUAL-FUND-ETF", ["US"], ["Mutual Fund", "ETF"])
    _run("GLOBAL:Canada-MUTUAL-FUND-ETF", ["Canada"], ["Mutual Fund", "ETF"])
    return summary


def calibrate_harvard_universe_anchors(cfg: AnchorCalibConfig | None = None) -> dict[str, Any]:
    """
    Calibrate anchors specifically for Harvard Universe products.
    
    Creates anchors:
    - HARVARD-US: US products from Harvard Universe
    - HARVARD-UK: UK products from Harvard Universe  
    - HARVARD-CA: CA products from Harvard Universe
    - GLOBAL-HARVARD: All Harvard Universe products
    
    Returns:
        Summary of calibration results
    """
    cfg = cfg or AnchorCalibConfig.from_environment()
    
    try:
        # Lazy import to avoid circular dependency
        from services.universe_manager import get_harvard_universe_manager
        manager = get_harvard_universe_manager()
        
        # Get all Harvard Universe products
        all_products = manager.get_universe_products()
        _LOG.info("Calibrating Harvard anchors for %d products", len(all_products))
        
        results = {
            "HARVARD-US": {"status": "pending", "count": 0},
            "HARVARD-UK": {"status": "pending", "count": 0}, 
            "HARVARD-CA": {"status": "pending", "count": 0},
            "GLOBAL-HARVARD": {"status": "pending", "count": 0},
        }
        
        # Group products by country
        products_by_country = {}
        all_symbols = []
        
        for product in all_products:
            if product.has_cvar:  # Only products with CVaR data can be used for anchors
                products_by_country.setdefault(product.country, []).append(product)
                all_symbols.append(product.symbol)
        
        # Create country-specific anchors
        for country, products in products_by_country.items():
            category = f"HARVARD-{country}"
            symbols = [p.symbol for p in products]
            
            success = _calibrate_harvard_anchor(category, symbols, cfg)
            results[category] = {
                "status": "success" if success else "failed",
                "count": len(symbols),
            }
            _LOG.info("Harvard anchor %s: %d products, success=%s", 
                     category, len(symbols), success)
        
        # Create global Harvard anchor
        if all_symbols:
            success = _calibrate_harvard_anchor("GLOBAL-HARVARD", all_symbols, cfg)
            results["GLOBAL-HARVARD"] = {
                "status": "success" if success else "failed", 
                "count": len(all_symbols),
            }
            _LOG.info("Global Harvard anchor: %d products, success=%s", 
                     len(all_symbols), success)
        
        return results
        
    except Exception as e:
        _LOG.error("Harvard anchor calibration failed: %s", str(e))
        return {"error": str(e)}


def _calibrate_harvard_anchor(category: str, symbols: list[str], cfg: AnchorCalibConfig) -> bool:
    """
    Create anchor for specific Harvard Universe symbols.
    
    Args:
        category: Anchor category name (e.g., "HARVARD-US")
        symbols: List of symbols to include
        cfg: Calibration configuration
        
    Returns:
        True if successful, False otherwise
    """
    sess = get_db_session()
    if sess is None:
        return False
        
    try:
        # Check if anchor already exists
        ver = current_quarter_version()
        existing = sess.query(CompassAnchor).filter(
            CompassAnchor.category == category,
            CompassAnchor.version == ver,
        ).one_or_none()
        
        if existing:
            _LOG.info("Harvard anchor already exists: category=%s version=%s", category, ver)
            return True  # Возвращаем True, так как якорь уже существует
        
        # Try to get data from CompassInputs first (preferred source)
        from sqlalchemy import and_, func
        from core.models import CompassInputs, Symbols
        
        # Get latest version of CompassInputs
        latest_version = sess.query(func.max(CompassInputs.version_id)).scalar()
        
        if latest_version:
            # Get mu_i values from CompassInputs
            q = (
                sess.query(CompassInputs.mu_i)
                .join(Symbols, Symbols.id == CompassInputs.instrument_id)
                .filter(
                    Symbols.symbol.in_(symbols),
                    CompassInputs.version_id == latest_version,
                    CompassInputs.mu_i.isnot(None)
                )
            )
            
            # Extract μ values
            mu_vals = []
            for r in q.all():
                try:
                    if r.mu_i is not None:
                        mu_vals.append(float(r.mu_i))
                except Exception:
                    continue
                    
            _LOG.info("Using CompassInputs for %s anchor calibration: found %d values", 
                     category, len(mu_vals))
        else:
            # Fallback to CvarSnapshot if no CompassInputs are available
            _LOG.warning("No CompassInputs found, falling back to CvarSnapshot for %s anchor", category)
            
            latest = (
                sess.query(
                    CvarSnapshot.symbol.label("symbol"),
                    func.max(CvarSnapshot.as_of_date).label("mx"),
                )
                .filter(CvarSnapshot.symbol.in_(symbols))
                .group_by(CvarSnapshot.symbol)
                .subquery()
            )
            
            q = (
                sess.query(CvarSnapshot)
                .join(
                    latest,
                    and_(
                        CvarSnapshot.symbol == latest.c.symbol,
                        CvarSnapshot.as_of_date == latest.c.mx,
                    ),
                )
                .filter(CvarSnapshot.return_annual.isnot(None))
            )
            
            # Extract μ values
            mu_vals = []
            for r in q.all():
                try:
                    if r.return_annual is not None:
                        mu_vals.append(float(r.return_annual))
                except Exception:
                    continue
                    
            _LOG.warning("Using CvarSnapshot fallback for %s anchor: found %d values", 
                        category, len(mu_vals))
        
        # Если нет данных или их слишком мало, создаем якорь с дефолтными значениями
        min_sample_size = 5  # Уменьшаем минимальное количество для Канады
        if len(mu_vals) < min_sample_size:
            _LOG.warning("Harvard anchor %s: insufficient data %d < %d, creating default anchor", 
                        category, len(mu_vals), min_sample_size)
            
            # Создаем якорь с дефолтными значениями
            anchor = CompassAnchor(
                category=category,
                version=ver,
                mu_low=0.01,     # 1% минимальная годовая доходность
                mu_high=0.15,    # 15% максимальная годовая доходность
                median_mu=0.06,  # 6% медианная годовая доходность
                p1=0.01,         # 1% процентиль
                p99=0.15,        # 99% процентиль
                metadata_json=f'{{"n": 0, "source": "harvard_universe_default", "symbols": {len(symbols)}}}',
            )
            
            sess.add(anchor)
            sess.commit()
            
            _LOG.info("Created default Harvard anchor: category=%s (insufficient data)", category)
            return True
        
        # Calibrate anchors
        mu_low, mu_high, med, p1, p99 = calibrate_mu_anchors(mu_vals, cfg)
        
        # Save to database
        anchor = CompassAnchor(
            category=category,
            version=ver,
            mu_low=mu_low,
            mu_high=mu_high,
            median_mu=med,
            p1=p1,
            p99=p99,
            metadata_json=f'{{"n": {len(mu_vals)}, "source": "harvard_universe", "symbols": {len(symbols)}}}',
        )
        
        sess.add(anchor)
        sess.commit()
        
        _LOG.info("Created Harvard anchor: category=%s n=%d symbols=%d", 
                 category, len(mu_vals), len(symbols))
        return True
        
    except Exception as e:
        _LOG.error("Failed to create Harvard anchor %s: %s", category, str(e))
        return False
    finally:
        try:
            sess.close()
        except Exception:
            pass
