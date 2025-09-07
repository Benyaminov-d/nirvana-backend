"""Business logic bootstrap - CVaR, caching, Compass, reconciliation."""

from __future__ import annotations

import logging
import os
from typing import Dict, Any, Union

from core.db import get_db_session
from core.models import CvarSnapshot

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Ensure detailed bootstrap logs are visible


def warm_cache_from_db(db_ready: bool) -> None:
    """Warm cache from existing CVaR snapshots."""
    if not db_ready:
        return
    
    from services.domain.cvar_unified_service import CvarUnifiedService
    
    try:
        sess = get_db_session()
        if sess is None:
            return
        
        try:
            rows = (
                sess.query(CvarSnapshot)
                .order_by(
                    CvarSnapshot.symbol.asc(),
                    CvarSnapshot.as_of_date.desc(),
                )
                .all()
            )
            by_symbol: dict[str, dict[int, CvarSnapshot]] = {}
            for r in rows:
                slot = by_symbol.setdefault(r.symbol, {})
                if r.alpha_label not in slot:
                    slot[r.alpha_label] = r
            
            calc = CvarUnifiedService()
            for sym, parts in by_symbol.items():
                any_block = parts.get(95) or parts.get(99) or parts.get(50)
                if any_block is None:
                    continue
                as_of = any_block.as_of_date.isoformat()

                def _triple(opt: CvarSnapshot | None) -> dict:
                    if opt is None:
                        return {"nig": None, "ghst": None, "evar": None}
                    return {"nig": opt.cvar_nig, "ghst": opt.cvar_ghst, "evar": opt.cvar_evar}

                payload = {
                    "success": True,
                    "as_of_date": as_of,
                    "start_date": None,
                    "data_summary": {},
                    "cached": True,
                    "cvar50": {
                    "annual": _triple(parts.get(50)),
                    "snapshot": _triple(parts.get(50)),
                    "alpha": 0.5,
                },
                    "cvar95": {
                    "annual": _triple(parts.get(95)),
                    "snapshot": _triple(parts.get(95)),
                    "alpha": 0.05,
                },
                    "cvar99": {
                    "annual": _triple(parts.get(99)),
                    "snapshot": _triple(parts.get(99)),
                    "alpha": 0.01,
                },
                }
                try:
                    calc.set_cached(sym, payload)
                except Exception:
                    continue
        finally:
            try:
                sess.close()
            except Exception:
                pass
    except Exception:
        logger.exception("Cache warming failed")


def enqueue_all_if_snapshots_empty(db_ready: bool) -> Union[dict, None]:
    """If snapshots are empty, enqueue all symbols for CVaR calculation."""
    if not db_ready:
        return None
    
    try:
        from startup.cvar_bootstrap import enqueue_all_if_snapshots_empty as _enqueue_all
        return _enqueue_all(db_ready)
    except Exception as e:
        logger.error("CVaR bootstrap failed: %s", str(e))
        return {"error": str(e)}


def reconcile_cvar_snapshots() -> Dict[str, Union[int, str]]:
    """Reconcile CVaR snapshots to bind them to instruments."""
    try:
        from startup.reconcile_snapshots import reconcile_cvar_snapshots as _reconcile
        return _reconcile()
    except Exception as e:
        logger.error("Snapshot reconciliation failed: %s", str(e))
        return {"error": str(e)}


def setup_compass_parameters() -> None:
    """Setup Compass parameters for validated universe."""
    try:
        from services.compass_parameters_service import (
            create_compass_parameters_for_validated_universe,
            setup_reference_data,
        )
        logger.info("Creating compass parameters for validated universe")
        # Setup minimal reference data first
        setup_reference_data()
        
        # Process parameters (μ from EODHD, L from CVaR snapshots)
        create_compass_parameters_for_validated_universe()
        logger.info("Compass parameters processing completed")
    except Exception as e:
        logger.error("Compass parameters creation failed: %s", str(e))


def calibrate_compass_anchors(scope: str = "default") -> Dict[str, Any]:
    """Auto-calibrate Compass anchors based on scope."""
    try:
        from services.compass_anchors import (
            auto_calibrate_from_db,
            auto_calibrate_global_per_country_from_db,
            auto_calibrate_by_type_country_from_db,
            calibrate_validated_universe_anchors,
        )
        
        results = {}
        
        if scope == "validated":
            # Validated universe anchors (filters: US/UK/CA)
            logger.info("Calibrating validated universe anchors")
            res_validated = calibrate_validated_universe_anchors()
            results['validated'] = res_validated
            logger.info("Validated anchors summary: %s", res_validated)
            
        elif scope == "global":
            # Per-country GLOBAL (discover from DB)
            res1 = auto_calibrate_global_per_country_from_db()
            results['per_country'] = res1
            logger.info("Anchors per-country summary: %s", res1)
            
            # Per (type,country)
            res2 = auto_calibrate_by_type_country_from_db()
            results['by_type_country'] = res2
            logger.info("Anchors by-type-country summary: %s", res2)
            
        else:
            # Default US-Equity-Returns calibration
            if auto_calibrate_from_db("US-Equity-Returns"):
                results['default'] = "calibrated"
                logger.info("Compass anchors calibrated (US-Equity-Returns)")
            else:
                results['default'] = "present_or_skipped"
                logger.info("Compass anchors present or skipped (US-Equity-Returns)")
        
        return results
    except Exception as e:
        logger.error("Compass anchors calibration failed: %s", str(e))
        return {"error": str(e)}


def calibrate_experiment_anchors() -> Dict[str, Any]:
    """Calibrate experimental anchors for /score-experiment page."""
    try:
        from services.compass_anchors import calibrate_special_sets
        
        res_exp = calibrate_special_sets()
        logger.info("Experiment anchors summary: %s", res_exp)
        return res_exp
    except Exception as e:
        logger.error("Experiment anchors calibration failed: %s", str(e))
        return {"error": str(e)}


def run_business_bootstrap(db_ready: bool) -> Dict[str, Union[bool, str, Dict[str, Any]]]:
    """Run all business logic bootstrap tasks with environment controls."""
    if not db_ready:
        return {"error": "Database not ready"}
    
    results = {}
    
    # Compass parameters - controlled via STARTUP_COMPASS_PARAMETERS
    try:
        compass_params_flag = os.getenv("STARTUP_COMPASS_PARAMETERS", "0").lower()
        if compass_params_flag in ("1", "true", "yes"):
            setup_compass_parameters()
            results['compass_parameters'] = True
        else:
            logger.info(
                "Compass parameters creation skipped (STARTUP_COMPASS_PARAMETERS=%s)",
                compass_params_flag,
            )
            results['compass_parameters'] = 'skipped'
    except Exception:
        logger.exception("Compass parameters creation failed")
        results['compass_parameters'] = 'error'
    
    # Compass anchors - controlled via STARTUP_COMPASS_ANCHORS
    try:
        compass_anchors_flag = os.getenv("STARTUP_COMPASS_ANCHORS", "1").lower()
        if compass_anchors_flag in ("1", "true", "yes"):
            print("  - Calibrating Compass anchors...")  # Force visibility
            scope = os.getenv("COMPASS_ANCHOR_SCOPE", "").lower()
            anchor_results = calibrate_compass_anchors(scope)
            print(f"  - Compass anchors calibrated: {anchor_results}")  # Force visibility
            results['compass_anchors'] = anchor_results
            
            # Optional experiment anchors - controlled via COMPASS_EXPERIMENT_ANCHORS
            exp_flag = os.getenv("COMPASS_EXPERIMENT_ANCHORS", "0").lower()
            if exp_flag in ("1", "true", "yes"):
                exp_results = calibrate_experiment_anchors()
                results['experiment_anchors'] = exp_results
            else:
                logger.info("Experiment anchors skipped (COMPASS_EXPERIMENT_ANCHORS=%s)", exp_flag)
                results['experiment_anchors'] = 'skipped'
        else:
            logger.info("Compass anchors skipped (STARTUP_COMPASS_ANCHORS=0)")
            results['compass_anchors'] = 'skipped'
            results['experiment_anchors'] = 'skipped'
    except Exception:
        logger.exception("Compass anchors auto-calibration failed")
        results['compass_anchors'] = 'error'
        results['experiment_anchors'] = 'error'
    
    # CVaR bootstrap - controlled via STARTUP_CVAR_BOOTSTRAP
    try:
        cvar_bootstrap_flag = os.getenv("STARTUP_CVAR_BOOTSTRAP", "1").lower()
        if cvar_bootstrap_flag in ("1", "true", "yes"):
            print("  - Running CVaR bootstrap...")  # Force visibility
            res_boot = enqueue_all_if_snapshots_empty(db_ready)
            print(f"  - CVaR bootstrap completed: {res_boot}")  # Force visibility
            results['cvar_bootstrap'] = res_boot
            if res_boot is not None:
                logger.info("CVaR bootstrap summary: %s", res_boot)
        else:
            print(f"  - CVaR bootstrap skipped (STARTUP_CVAR_BOOTSTRAP={cvar_bootstrap_flag})")  # Force visibility
            logger.info(
                "CVaR bootstrap skipped (STARTUP_CVAR_BOOTSTRAP=%s)",
                cvar_bootstrap_flag,
            )
            results['cvar_bootstrap'] = 'skipped'
    except Exception:
        logger.exception("CVaR bootstrap enqueue failed")
        results['cvar_bootstrap'] = 'error'
    
    # Reconcile CVaR snapshots - controlled via NIR_RECONCILE_SNAPSHOTS
    try:
        reconcile_flag = os.getenv("NIR_RECONCILE_SNAPSHOTS", "0").lower()
        if reconcile_flag in ("1", "true", "yes"):
            res_rec = reconcile_cvar_snapshots()
            results['reconcile_snapshots'] = res_rec
            logger.info("Reconcile snapshots summary: %s", res_rec)
        else:
            logger.info("Reconcile snapshots skipped (NIR_RECONCILE_SNAPSHOTS=0)")
            results['reconcile_snapshots'] = 'skipped'
    except Exception:
        logger.exception("Reconcile snapshots failed")
        results['reconcile_snapshots'] = 'error'
    
    # Cache warming - controlled via STARTUP_CACHE_WARMING
    try:
        cache_warming_flag = os.getenv("STARTUP_CACHE_WARMING", "1").lower()
        if cache_warming_flag in ("1", "true", "yes"):
            print("  - Warming CVaR cache from database...")  # Force visibility
            warm_cache_from_db(db_ready)
            print("  - Cache warming completed")  # Force visibility
            results['cache_warming'] = True
        else:
            print(f"  - Cache warming skipped (STARTUP_CACHE_WARMING={cache_warming_flag})")  # Force visibility
            logger.info(
                "Cache warming skipped (STARTUP_CACHE_WARMING=%s)",
                cache_warming_flag,
            )
            results['cache_warming'] = 'skipped'
    except Exception:
        logger.exception("Cache warming failed")
        results['cache_warming'] = 'error'
    
    return results
