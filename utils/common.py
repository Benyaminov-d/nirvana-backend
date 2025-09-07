from __future__ import annotations

from datetime import datetime, timedelta
from typing import List


def parse_csv_list(val: str) -> List[str]:
    items: List[str] = []
    for tok in (val or "").split(","):
        s = tok.strip().upper()
        if s:
            items.append(s)
    return items


def seconds_until(hour: int, minute: int) -> int:
    now = datetime.now()
    run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if run <= now:
        run = run + timedelta(days=1)
    return int((run - now).total_seconds())


def parse_exchanges_env() -> list[str]:
    import os

    # Now supports countries and exchange codes
    # Default includes US, CA (Canada), and LSE (London Stock Exchange for UK)
    raw = os.getenv("EODHD_EXCHANGES", "US,CA,LSE")
    out: list[str] = []
    for tok in (raw or "").split(","):
        t = tok.strip().upper()
        if t:
            out.append(t)
    seen: set[str] = set()
    uniq: list[str] = []
    for e in out:
        if e not in seen:
            uniq.append(e)
            seen.add(e)
    return uniq or ["US"]


def parse_instrument_types_env() -> list[str]:
    """Parse allowed instrument types from environment.

    Defaults to Mutual Fund and ETF types only.
    Can be overridden with EODHD_INSTRUMENT_TYPES env var.
    """
    import os

    raw = os.getenv("EODHD_INSTRUMENT_TYPES", "FUND,ETF,MUTUAL FUND")
    out: list[str] = []
    for tok in (raw or "").split(","):
        t = tok.strip().upper()
        if t:
            out.append(t)
    return out or ["FUND", "ETF", "MUTUAL FUND"]


def parse_excluded_exchanges_env() -> list[str]:
    """Parse excluded exchanges from environment.

    Defaults to PINK exchange.
    Can be overridden with EODHD_EXCLUDED_EXCHANGES env var.
    """
    import os

    raw = os.getenv("EODHD_EXCLUDED_EXCHANGES", "PINK")
    out: list[str] = []
    for tok in (raw or "").split(","):
        exchange = tok.strip().upper()
        if exchange:
            out.append(exchange)
    return out or ["PINK"]


def should_exclude_exchange(exchange: str | None) -> bool:
    """Check if exchange should be excluded based on env filter.

    Args:
        exchange: The exchange to check

    Returns:
        True if exchange should be excluded, False otherwise
    """
    if not exchange:
        return False

    excluded_exchanges = parse_excluded_exchanges_env()
    if not excluded_exchanges:
        return False  # No exclusions means include all

    # Normalize for comparison
    normalized_exchange = str(exchange).strip().upper()

    # Check if exchange is in excluded list
    return normalized_exchange in excluded_exchanges


def should_include_instrument_type(instrument_type: str | None) -> bool:
    """Check if instrument type should be included based on env filter.

    Args:
        instrument_type: The instrument type to check

    Returns:
        True if instrument should be included, False otherwise
    """
    if not instrument_type:
        return False

    allowed_types = parse_instrument_types_env()
    if not allowed_types:
        return True  # No filter means include all

    # Normalize for comparison
    normalized_type = str(instrument_type).strip().upper()

    # Check direct matches and common variations
    for allowed in allowed_types:
        if normalized_type == allowed:
            return True
        # Handle common variations
        if allowed == "FUND" and "FUND" in normalized_type:
            return True
        if allowed == "ETF" and normalized_type == "ETF":
            return True
        if (allowed == "MUTUAL FUND" and "MUTUAL" in normalized_type
                and "FUND" in normalized_type):
            return True

    return False


# Canonical instrument type mapping.
_INSTRUMENT_TYPE_MAP: dict[str, str] = {
    # funds
    "fund": "Fund",
    "mutual fund": "Mutual Fund",
    # equities
    "common stock": "Common Stock",
    "preferred stock": "Preferred Stock",
    # etfs and others
    "etf": "ETF",
    "unit": "Unit",
    "notes": "Notes",
    "etc": "ETC",
    "bond": "Bond",
}


def canonical_instrument_type(value: object | None) -> str | None:
    """Normalize instrument type/category to a canonical label.

    Trims whitespace, case-insensitive mapping via _INSTRUMENT_TYPE_MAP.
    Returns None when empty or invalid.
    """
    if value is None:
        return None
    try:
        s = str(value).strip()
    except Exception:
        return None
    if not s:
        return None
    key = s.lower()
    return _INSTRUMENT_TYPE_MAP.get(key, s)


def _eodhd_suffix_for(exchange: str | None, country: str | None) -> str:
    """Map DB exchange/country to EODHD endpoint suffix.
    
    Priority:
    1. Use exchange directly if it's a known EODHD exchange code
    2. Map common exchange names to EODHD codes
    3. Fall back to country-based defaults
    """
    try:
        ex = (exchange or "").strip().upper()
        co = (country or "").strip().upper()
        
        # Direct EODHD exchange codes (use as-is)
        direct_exchanges = {
            "LSE", "TO", "V", "CN", "NE", "IL", "US", "PA", "SW", 
            "XETRA", "F", "MC", "AS", "BR", "VI", "LU", "HE", 
            "OL", "CO", "ST", "JSE", "AU", "SN", "BA", "SA", "MX"
        }
        if ex in direct_exchanges:
            return f".{ex}"
        
        # Common exchange name mappings to EODHD codes
        exchange_mappings = {
            "TSX": ".TO",
            "TSXV": ".V", 
            "TORONTO": ".TO",
            "NASDAQ": ".US",
            "NYSE": ".US",
            "LONDON": ".LSE",
            "PARIS": ".PA",
            "FRANKFURT": ".F",
            "XFRA": ".F"
        }
        if ex in exchange_mappings:
            return exchange_mappings[ex]
        
        # Country-based fallbacks when exchange is unknown
        if co in ("CA", "CANADA"):
            return ".TO"  # Default to Toronto Stock Exchange
        if co in ("UK", "UNITED KINGDOM", "GB", "GBR"):
            return ".LSE"  # Default to London Stock Exchange
        if co in ("FR", "FRANCE"):
            return ".PA"  # Paris
        if co in ("DE", "GERMANY"):
            return ".XETRA"  # Germany
        
        # Default US
        return ".US"
    except Exception:
        return ".US"


def resolve_eodhd_endpoint_symbol(symbol: str) -> str:
    """Resolve a ticker into an EODHD endpoint code, using DB metadata when available.

    Rules:
    - Support SYMBOL:SUFFIX syntax (e.g., "WFSPX:US")
    - If already qualified (contains a dot), return as-is
    - Special symbols (BTC, ETH, SP500TR) map to provider-specific codes
    - Otherwise, try DB PriceSeries.exchange/country to choose suffix
    - Fallback to env EODHD_DEFAULT_SUFFIX or .US
    """
    try:
        # Prefer library parser for SYMBOL:SUFFIX semantics when available
        try:
            from nirvana_risk.pipeline.symbols import (
                parse_symbol_suffix,
            )  # type: ignore
        except Exception:
            parse_symbol_suffix = None  # type: ignore
        raw = (symbol or "").strip().upper()
        if not raw:
            return raw
        # Handle SYMBOL:SUFFIX via lib (or fallback local logic)
        if ":" in raw and "." not in raw:
            try:
                if callable(parse_symbol_suffix):
                    base, suf = parse_symbol_suffix(raw)  # type: ignore
                    if base and suf:
                        return f"{base}{suf}"
                else:
                    base, suf = raw.split(":", 1)
                    suf = (suf or "").strip().upper()
                    if not suf.startswith('.'):
                        suf = "." + suf
                    return f"{base}{suf}"
            except Exception:
                pass
        sym = raw
        if not sym:
            return sym
        # Already qualified
        if "." in sym:
            return sym
        # Special mappings
        special = {
            "BTC": "BTC-USD.CC",
            "ETH": "ETH-USD.CC",
            "SP500TR": "SP500TR.INDX",
        }
        if sym in special:
            return special[sym]

        # Try DB lookup for exchange/country
        # WARNING: If multiple countries have same symbol, we need context to choose the right one!
        # For now, we'll prefer US, then Canada, then others as fallback for ambiguous symbols
        from core.db import get_db_session
        from core.models import PriceSeries  # type: ignore
        sess = get_db_session()
        exchange = None
        country = None
        if sess is not None:
            try:
                # Check if there are multiple records for this symbol
                all_rows = (
                    sess.query(
                        PriceSeries.exchange, PriceSeries.country
                    )  # type: ignore
                    .filter(PriceSeries.symbol == sym)  # type: ignore
                    .all()
                )
                
                if len(all_rows) == 1:
                    # Unique symbol - use it
                    exchange, country = all_rows[0]
                elif len(all_rows) > 1:
                    # Multiple symbols - prefer US, then Canada, then alphabetical
                    country_priority = {'US': 1, 'Canada': 2}
                    sorted_rows = sorted(all_rows, key=lambda x: (
                        country_priority.get(x[1], 99),  # Country priority
                        x[1] or 'ZZZ'  # Alphabetical fallback
                    ))
                    exchange, country = sorted_rows[0]
                    
                    # Log the ambiguity for debugging
                    import logging
                    _logger = logging.getLogger(__name__)
                    countries = [row[1] for row in all_rows]
                    _logger.warning(f"⚠️  Ambiguous symbol '{sym}' found in countries {countries}, using {country}")
                    
            except Exception:
                pass
            finally:
                try:
                    sess.close()
                except Exception:
                    pass

        if exchange or country:
            return f"{sym}{_eodhd_suffix_for(exchange, country)}"

        # Env fallback
        import os
        suf = os.getenv("EODHD_DEFAULT_SUFFIX", ".US") or ".US"
        if not suf.startswith("."):
            suf = "." + suf
        return f"{sym}{suf}"
    except Exception:
        return f"{(symbol or '').upper()}.US"


def db_base_symbol(symbol: str) -> str:
    """Return canonical DB symbol without country/exchange suffix.

    Strips common suffix formats:
    - SYMBOL:SUFFIX (e.g., AAPL:US)
    - SYMBOL.SUFFIX (e.g., AAPL.US, BCE.TO)

    Uses library parser when available; otherwise falls back to simple split.
    """
    try:
        raw = (symbol or "").strip().upper()
        if not raw:
            return raw
        # Prefer library parser if available
        try:
            from nirvana_risk.pipeline.symbols import parse_symbol_suffix  # type: ignore
        except Exception:
            parse_symbol_suffix = None  # type: ignore
        if callable(parse_symbol_suffix):
            try:
                base, _suf = parse_symbol_suffix(raw)
                if base:
                    return base
            except Exception:
                pass
        # Fallback: strip trailing .XX or :XX tokens (2-3 letters)
        if ":" in raw:
            base = raw.split(":", 1)[0].strip()
            if base:
                return base
        if "." in raw:
            # Keep left side only when right side looks like a suffix
            left, right = raw.rsplit(".", 1)
            rs = right.strip()
            if rs.isalpha() and 1 < len(rs) <= 3:
                return left.strip()
        return raw
    except Exception:
        return (symbol or "").strip().upper()
