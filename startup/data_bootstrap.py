"""Data bootstrap - exchanges, symbols, market data."""

from __future__ import annotations

import logging
import os
from typing import Dict, Any
from datetime import datetime

from core.db import get_db_session
from core.models import Symbols, Symbols
from core.persistence import bootstrap_annual_violations_from_csv
from utils.common import (
    canonical_instrument_type as _canon_type,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Ensure detailed bootstrap logs are visible


def bootstrap_annual_if_any(db_ready: bool) -> None:
    """Bootstrap annual violations from CSV if available."""
    if not db_ready:
        return
    try:
        bootstrap_annual_violations_from_csv("data/annual_cvar_violations.csv")
    except Exception:
        logger.debug("No annual violations CSV found or failed to load")


def normalize_instrument_types() -> int:
    """Normalize instrument types to canonical dictionary."""
    sess = get_db_session()
    if sess is None:
        return 0
    try:
        updated = 0
        rows = (
            sess.query(Symbols)
            .filter(Symbols.instrument_type.isnot(None))
            .all()
        )
        for row in rows:
            if row.instrument_type:
                canon = _canon_type(row.instrument_type)
                if canon != row.instrument_type:
                    row.instrument_type = canon
                    updated += 1
        if updated > 0:
            sess.commit()
        return updated
    except Exception:
        try:
            sess.rollback()
        except Exception:
            pass
        return 0
    finally:
        try:
            sess.close()
        except Exception:
            pass


def normalize_countries() -> int:
    """Normalize countries to canonical labels."""
    sess = get_db_session()
    if sess is None:
        return 0
    try:
        # Simple country normalization - can be expanded
        country_map = {
            'US': 'USA',
            'UK': 'United Kingdom',
            'CA': 'Canada',
        }
        updated = 0
        for old_country, new_country in country_map.items():
            result = (
                sess.query(Symbols)
                .filter(Symbols.country == old_country)
                .update(
                    {'country': new_country}, synchronize_session=False
                )
            )
            updated += result
        if updated > 0:
            sess.commit()
        return updated
    except Exception:
        try:
            sess.rollback()
        except Exception:
            pass
        return 0
    finally:
        try:
            sess.close()
        except Exception:
            pass


def bootstrap_exchanges_if_needed(db_ready: bool) -> int:
    """Bootstrap exchanges from EODHD API if needed."""
    if not db_ready:
        return 0
    
    try:
        from services.exchanges_sync import (
            sync_exchanges_once,
            should_sync_exchanges,
        )
        
        if should_sync_exchanges():
            logger.info("Starting exchanges bootstrap from EODHD")
            count = sync_exchanges_once(force=False)
            logger.info(
                "Exchanges bootstrap completed, synced %d exchanges", count
            )
            return count
        else:
            logger.info(
                "Exchanges bootstrap skipped (table not empty and env flag not set)"
            )
            return 0
            
    except Exception as e:
        logger.error("Exchanges bootstrap failed: %s", str(e))
        return 0


def bootstrap_symbols_if_empty(db_ready: bool) -> None:
    """Bootstrap symbols from EODHD if table is empty."""
    if not db_ready:
        return
    sess = get_db_session()
    if sess is None:
        return
    try:
        has_any = sess.query(Symbols.id).limit(1).all()
        if not has_any:
            from app import _sync_symbols_once
            _sync_symbols_once(force=True)
    except Exception as e:
        logger.error("Failed to bootstrap symbols: %s", str(e))
    finally:
        try:
            sess.close()
        except Exception:
            pass


def bootstrap_eodhd_symbols_by_exchanges(db_ready: bool) -> int:
    """Bootstrap symbols from EODHD for configured exchanges."""
    if not db_ready:
        return 0
    
    try:
        from app import _sync_symbols_once
        count = _sync_symbols_once(force=True)
        logger.info(
            "EODHD symbols bootstrap completed, processed %d symbols", count
        )
        return count
    except Exception as e:
        logger.error("EODHD symbols bootstrap failed: %s", str(e))
        return 0


def import_local_symbol_catalogs(db_ready: bool) -> int:
    """Import symbols from local CSV catalogs."""
    if not db_ready:
        return 0
    
    try:
        from startup.symbols_from_files import (
            import_local_symbol_catalogs as _import_local,
        )
        return _import_local(db_ready)
    except Exception as e:
        logger.error("Local symbols import failed: %s", str(e))
        return 0


def load_core_symbols(db_ready: bool) -> Dict[str, Any]:
    """Load symbols from core structure with flags."""
    if not db_ready:
        return {"error": "Database not ready"}
    
    try:
        from startup.symbols_core_loader import (
            load_core_symbols as _load_core,
        )
        return _load_core(db_ready=db_ready)
    except Exception as e:
        logger.error("Core symbols import failed: %s", str(e))
        return {"error": str(e)}


def mark_five_stars_from_files(db_ready: bool) -> None:
    """Mark five star symbols from files."""
    if not db_ready:
        return
    
    try:
        from startup.symbols_from_files import (
            mark_five_stars as _mark_five_stars,
        )
        _mark_five_stars(db_ready)
    except Exception as e:
        logger.error("Five stars processing failed: %s", str(e))


def reconcile_five_stars_usa(db_ready: bool) -> Dict[str, Any]:
    """Reconcile five stars USA data."""
    if not db_ready:
        return {"error": "Database not ready"}
    
    try:
        from startup.symbols_from_files import (
            reconcile_five_stars_usa as _reconcile_usa,
        )
        return _reconcile_usa(db_ready)
    except Exception as e:
        logger.error("Five stars USA reconciliation failed: %s", str(e))
        return {"error": str(e)}


def import_five_stars_canada(db_ready: bool) -> None:
    """Import Canada five-star OCR PDFs."""
    if not db_ready:
        return
    
    try:
        from startup.symbols_from_files import (
            import_five_stars_canada as _import_canada,
        )
        _import_canada(db_ready)
    except Exception as e:
        logger.error("Five stars Canada import failed: %s", str(e))


def validate_data_integrity(db_ready: bool) -> Dict[str, Any]:
    """Validate data integrity on startup."""
    if not db_ready:
        return {"error": "Database not ready"}
    
    try:
        from startup.data_validation import validate_all_data
        
        # Run validation
        validation_flag = os.getenv("STARTUP_VALIDATE_DATA", "1").lower()
        if validation_flag in ("1", "true", "yes"):
            logger.info("Validating data integrity...")
            validation_result = validate_all_data()
            
            # Log any critical issues
            if not validation_result.get("success", False):
                critical_issues = validation_result.get("critical_issues", [])
                for issue in critical_issues:
                    logger.error("Critical data issue: %s", issue)
                    
            return validation_result
        else:
            logger.info("Data validation skipped (STARTUP_VALIDATE_DATA=%s)", validation_flag)
            return {"skipped": True}
            
    except Exception as e:
        logger.exception("Data validation failed")
        return {"error": str(e)}


def run_data_bootstrap(db_ready: bool) -> Dict[str, Any]:
    """Run all data bootstrap tasks with environment variable controls."""
    if not db_ready:
        return {"error": "Database not ready"}
    
    results = {}
    
    # Validate data integrity
    try:
        print("  - Validating data integrity...")  # Force visibility
        validation_result = validate_data_integrity(db_ready)
        if validation_result.get("skipped", False):
            print("  - Data validation skipped")  # Force visibility
            results['data_validation'] = 'skipped'
        elif validation_result.get("error"):
            print(f"  - Data validation error: {validation_result['error']}")  # Force visibility
            results['data_validation'] = 'error'
        elif validation_result.get("critical_issues"):
            print(f"  - Data validation found {len(validation_result.get('critical_issues', []))} critical issues")  # Force visibility
            results['data_validation'] = {
                'success': False,
                'critical_issues': len(validation_result.get('critical_issues', [])),
            }
        else:
            print("  - Data validation passed")  # Force visibility
            results['data_validation'] = {
                'success': True,
                'timestamp': datetime.utcnow().isoformat(),
            }
    except Exception:
        print("  - Data validation failed")  # Force visibility
        logger.exception("Data validation failed")
        results['data_validation'] = 'error'
    
    # Annual violations bootstrap
    try:
        print("  - Bootstrapping annual violations...")  # Force visibility
        bootstrap_annual_if_any(db_ready)
        print("  - Annual violations bootstrap completed")  # Force visibility
        results['annual'] = True
    except Exception:
        print("  - Annual bootstrap failed")  # Force visibility
        logger.exception("Annual bootstrap failed")
        results['annual'] = False
    
    # Exchanges bootstrap - controlled via STARTUP_EXCHANGES_BOOTSTRAP
    try:
        exchanges_flag = os.getenv("STARTUP_EXCHANGES_BOOTSTRAP", "1").lower()
        if exchanges_flag in ("1", "true", "yes"):
            print("  - Bootstrapping exchanges...")  # Force visibility
            count = bootstrap_exchanges_if_needed(db_ready)
            print(f"  - Exchanges bootstrap completed: {count} exchanges processed")  # Force visibility
            results['exchanges'] = count
        else:
            print(f"  - Exchanges bootstrap skipped (STARTUP_EXCHANGES_BOOTSTRAP={exchanges_flag})")  # Force visibility
            logger.info("Exchanges bootstrap skipped (STARTUP_EXCHANGES_BOOTSTRAP=%s)", exchanges_flag)
            results['exchanges'] = 'skipped'
    except Exception:
        print("  - Exchanges bootstrap failed")  # Force visibility
        logger.exception("Exchanges bootstrap failed")
        results['exchanges'] = 'error'
    
    # Symbols bootstrap - controlled via STARTUP_SYMBOLS_BOOTSTRAP
    try:
        symbols_flag = os.getenv("STARTUP_SYMBOLS_BOOTSTRAP", "1").lower()
        if symbols_flag in ("1", "true", "yes"):
            print("  - Bootstrapping symbols...")  # Force visibility
            bootstrap_symbols_if_empty(db_ready)
            print("  - Symbols bootstrap completed")  # Force visibility
            results['symbols'] = True
        else:
            print(f"  - Symbols bootstrap skipped (STARTUP_SYMBOLS_BOOTSTRAP={symbols_flag})")  # Force visibility
            logger.info("Symbols bootstrap skipped (STARTUP_SYMBOLS_BOOTSTRAP=%s)", symbols_flag)
            results['symbols'] = 'skipped'
    except Exception:
        print("  - Symbols bootstrap failed")  # Force visibility
        logger.exception("Symbols bootstrap failed")
        results['symbols'] = 'error'
    
    # Local symbols import - controlled via STARTUP_LOCAL_SYMBOLS_IMPORT
    try:
        local_import_flag = os.getenv("STARTUP_LOCAL_SYMBOLS_IMPORT", "1").lower()
        if local_import_flag in ("1", "true", "yes"):
            print("  - Importing local symbol catalogs...")  # Force visibility
            count = import_local_symbol_catalogs(db_ready)
            print(f"  - Local symbols import completed: {count} catalogs processed")  # Force visibility
            results['local_import'] = count
        else:
            print(f"  - Local symbols import skipped (STARTUP_LOCAL_SYMBOLS_IMPORT={local_import_flag})")  # Force visibility
            logger.info("Local symbols import skipped (STARTUP_LOCAL_SYMBOLS_IMPORT=%s)", local_import_flag)
            results['local_import'] = 'skipped'
    except Exception:
        print("  - Local symbols import failed")  # Force visibility
        logger.exception("Local symbols import failed")
        results['local_import'] = 'error'
    
    # Core symbols - controlled via STARTUP_CORE_SYMBOLS
    try:
        core_flag = os.getenv("STARTUP_CORE_SYMBOLS", "1").lower()
        if core_flag in ("1", "true", "yes"):
            print("  - Loading core symbols...")  # Force visibility
            core_result = load_core_symbols(db_ready)
            if "error" in core_result:
                print(f"  - Core symbols loading error: {core_result.get('error')}")  # Force visibility
                results['core_symbols'] = 'error'
            else:
                print(f"  - Core symbols loaded: {core_result.get('count', 0)} processed")  # Force visibility
                results['core_symbols'] = core_result.get('count', 0)
        else:
            print(f"  - Core symbols skipped (STARTUP_CORE_SYMBOLS={core_flag})")  # Force visibility
            logger.info("Core symbols skipped (STARTUP_CORE_SYMBOLS=%s)", core_flag)
            results['core_symbols'] = 'skipped'
    except Exception:
        print("  - Core symbols loading failed")  # Force visibility
        logger.exception("Core symbols loading failed")
        results['core_symbols'] = 'error'
    
    # Five-star symbols - controlled via STARTUP_FIVE_STARS
    try:
        five_stars_flag = os.getenv("STARTUP_FIVE_STARS", "1").lower()
        if five_stars_flag in ("1", "true", "yes"):
            print("  - Processing five-star symbols...")  # Force visibility
            mark_five_stars_from_files(db_ready)
            usa_result = reconcile_five_stars_usa(db_ready)
            import_five_stars_canada(db_ready)
            print("  - Five-star symbols processed")  # Force visibility
            results['five_stars'] = usa_result
        else:
            print(f"  - Five-star symbols skipped (STARTUP_FIVE_STARS={five_stars_flag})")  # Force visibility
            logger.info("Five-star symbols skipped (STARTUP_FIVE_STARS=%s)", five_stars_flag)
            results['five_stars'] = 'skipped'
    except Exception:
        print("  - Five-star symbols processing failed")  # Force visibility
        logger.exception("Five-star symbols processing failed")
        results['five_stars'] = 'error'
    
    # Normalize instrument types and countries - controlled via STARTUP_NORMALIZE
    try:
        normalize_flag = os.getenv("STARTUP_NORMALIZE", "1").lower()
        if normalize_flag in ("1", "true", "yes"):
            print("  - Normalizing instrument types and countries...")  # Force visibility
            updated_types = normalize_instrument_types()
            updated_countries = normalize_countries()
            print(f"  - Normalized {updated_types} instrument types and {updated_countries} countries")  # Force visibility
            results['normalize'] = {
                'instrument_types': updated_types,
                'countries': updated_countries
            }
        else:
            print(f"  - Normalization skipped (STARTUP_NORMALIZE={normalize_flag})")  # Force visibility
            logger.info("Normalization skipped (STARTUP_NORMALIZE=%s)", normalize_flag)
            results['normalize'] = 'skipped'
    except Exception:
        print("  - Normalization failed")  # Force visibility
        logger.exception("Normalization failed")
        results['normalize'] = 'error'
    
    return results