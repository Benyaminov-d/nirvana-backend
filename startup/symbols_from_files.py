from __future__ import annotations
from pathlib import Path as _P
from typing import Dict, List
import csv
from core.db import get_db_session
from core.persistence import upsert_price_series_bulk
from core.models import Symbols, Symbols  # Symbols is backward compatibility alias
from datetime import datetime as _dt
from pathlib import Path as _Path
import re
import logging


_logger = logging.getLogger("symbols_import")


def _row_to_item(row: Dict[str, str]) -> Dict[str, str]:
    """Normalize a CSV row to the expected item keys for upsert.

    Expected output keys: Code, Name, Country, Exchange, Currency, Type, Isin
    """
    def _g(*names: str) -> str | None:
        for n in names:
            if n in row and row.get(n) not in (None, ""):
                return str(row.get(n))
        # try case-insensitive fallback
        lower_map = {str(k).lower(): k for k in row.keys()}
        for n in names:
            k = lower_map.get(str(n).lower())
            if k is not None and row.get(k) not in (None, ""):
                return str(row.get(k))
        return None

    return {
        "Code": (_g("Code", "code", "Ticker") or "").strip(),
        "Name": _g("Name", "name") or "",
        "Country": _g("Country", "country") or "",
        "Exchange": _g("Exchange", "exchange") or "",
        "Currency": _g("Currency", "currency") or "",
        "Type": _g("Type", "type", "TypeCode") or "",
        "Isin": _g("Isin", "ISIN", "isin") or "",
    }


def import_local_symbol_catalogs(db_ready: bool) -> int:
    """Scan local catalog CSVs and upsert symbols into DB.

    - Looks under backend/symbols/by_country/*.csv
    - Safe to run on every startup; performs idempotent upserts
    - Returns total number of rows processed (sum over files)
    """
    if not db_ready:
        return 0
    # Quick DB availability check
    if get_db_session() is None:
        return 0

    base = _P(__file__).parents[1] / "symbols" / "by_country"
    if not base.exists() or not base.is_dir():
        return 0

    try:
        _logger.info(
            "import_local_symbol_catalogs: scanning %s", str(base)
        )
    except Exception:
        pass
    total_processed = 0
    # Chunk to keep memory and transaction sizes reasonable
    CHUNK = 2000
    for csv_path in sorted(base.glob("*.csv")):
        try:
            try:
                _logger.info("reading CSV: %s", csv_path.name)
            except Exception:
                pass
            # Infer country from filename when possible (e.g., *_CANADA.csv, *_US.csv)
            file_country: str | None = None
            try:
                name_low = csv_path.name.lower()
                if "canada" in name_low or name_low.endswith("_ca.csv"):
                    file_country = "Canada"
                elif any(tok in name_low for tok in ("_us.csv", "usa", "united_states")):
                    file_country = "US"
            except Exception:
                file_country = None
            items: List[Dict[str, str]] = []
            with csv_path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        item = _row_to_item(row)
                        # If CSV has no explicit country, use the one inferred from filename
                        if (not item.get("Country")) and file_country:
                            item["Country"] = file_country
                        if not item.get("Code"):
                            continue
                        items.append(item)
                        if len(items) >= CHUNK:
                            n = upsert_price_series_bulk(items)
                            total_processed += n
                            try:
                                _logger.info(
                                    "upserted chunk: %d from %s",
                                    int(n), csv_path.name,
                                )
                            except Exception:
                                pass
                            items.clear()
                    except Exception:
                        continue
            if items:
                n = upsert_price_series_bulk(items)
                total_processed += n
                try:
                    _logger.info(
                        "upserted tail: %d from %s", int(n), csv_path.name
                    )
                except Exception:
                    pass
        except Exception:
            # Continue with the next file
            continue

    try:
        _logger.info(
            "import_local_symbol_catalogs: total upserted=%d",
            int(total_processed),
        )
    except Exception:
        pass
    return int(total_processed)


def import_five_stars_canada(db_ready: bool) -> int:
    """Import Canada five-star lists from CSV files and upsert into DB.

    - Looks under backend/symbols/five_stars_canada/*.csv
    - Extracts symbol, name, currency; defaults country=Canada
    - Marks five_stars=1 and sets country=Canada
    """
    if not db_ready:
        return 0
    if get_db_session() is None:
        return 0
    folder = _Path(__file__).parents[1] / "symbols" / "five_stars_canada"
    if not folder.exists() or not folder.is_dir():
        return 0

    # Optional whitelist of valid CA symbols from EODHD CSV to remove noise
    allowed_codes: set[str] | None = None
    allowed_bases: set[str] | None = None
    try:
        import os as _os
        use_whitelist = (
            _os.getenv("CA_FIVESTARS_FILTER_TO_CSV", "1").lower()
            in ("1", "true", "yes")
        )
    except Exception:
        use_whitelist = True
    if use_whitelist:
        try:
            ca_csv = _Path(__file__).parents[1] / "symbols" / "by_country" / "EODHD_SYMBOLS_CANADA.csv"
            if ca_csv.exists():
                allowed_codes = set()
                with ca_csv.open("r", encoding="utf-8", newline="") as _f:
                    _r = csv.DictReader(_f)
                    for _row in _r:
                        c = str((_row.get("Code") or "")).strip().upper()
                        if c:
                            allowed_codes.add(c)
                try:
                    _logger.info(
                        "import_five_stars_canada: whitelist loaded=%d",
                        int(len(allowed_codes)),
                    )
                except Exception:
                    pass
                # Build whitelist of base tickers without exchange suffix
                try:
                    allowed_bases = set()
                    for code in allowed_codes:
                        base = code
                        if "." in base:
                            base = base.split(".")[0]
                        if ":" in base:
                            base = base.split(":")[0]
                        allowed_bases.add(base)
                except Exception:
                    allowed_bases = None
        except Exception:
            allowed_codes = None
            allowed_bases = None

    # PDF OCR path removed per request; rely on DOCX tables only
    items: list[dict] = []
    parsed_symbols: set[str] = set()
    first_seen: dict[str, dict] = {}

    def _flush() -> int:
        nonlocal items
        if not items:
            return 0
        n = upsert_price_series_bulk(items)
        items = []
        return int(n)

    total = 0
    # Heuristics: lines often contain "SYMBOL  Name ..... Currency  Category"
    # Accept base symbols 2..16 chars, alnum and optional single dash; must start with letter
    sym_re = re.compile(r"^[A-Z][A-Z0-9\-]{1,15}$")
    stop_words = {"ENDOBJ", "STREAM", "ENDSTREAM", "OBJ", "NULL", "TRUE", "FALSE"}
    curr_set = {"CAD", "USD", "CAD$", "US$"}

    try:
        _logger.info(
            "import_five_stars_canada: scanning %s", str(folder)
        )
    except Exception:
        pass
    # Parse CSV files only
    csv_files = sorted(folder.glob("*.csv"))
    if csv_files:
        for csv_path in csv_files:
            try:
                _logger.info("parsing CSV: %s", csv_path.name)
            except Exception:
                pass
            try:
                with csv_path.open("r", encoding="utf-8", newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if not row:
                            continue
                        lower_map = {str(k).lower(): k for k in row.keys()}
                        def _g(*names: str) -> str:
                            for n in names:
                                k = lower_map.get(str(n).lower())
                                if k is not None and row.get(k) not in (None, ""):
                                    return str(row.get(k))
                            return ""
                        raw_sym = _g("symbol", "ticker", "code")
                        raw_name = _g("name", "fund", "security", "security name", "product", "title")
                        raw_curr = _g("currency", "curr")
                        # Fallback to positional columns for these cleaned CSVs
                        if not raw_name:
                            raw_name = row.get("Unnamed: 1", "")
                        if not raw_sym:
                            raw_sym = row.get("Unnamed: 2", "")
                        if not raw_curr:
                            raw_curr = row.get("Unnamed: 3", "")
                        s = (raw_sym or "").upper().strip()
                        if not s:
                            continue
                        try:
                            s = re.sub(r"[^A-Z0-9\-\.:]", "", s)
                        except Exception:
                            s = s
                        base = s
                        if "." in base:
                            base = base.split(".")[0]
                        if ":" in base:
                            base = base.split(":")[0]
                        sym = base
                        if not sym or not sym_re.match(sym):
                            continue
                        if sym not in first_seen:
                            first_seen[sym] = {
                                "Name": (raw_name or sym).strip(),
                                "Currency": (raw_curr or "CAD").strip() or "CAD",
                            }
                        parsed_symbols.add(sym)
            except Exception:
                continue
        try:
            _logger.info(
                "import_five_stars_canada: parsed from CSV=%d", int(len(first_seen))
            )
        except Exception:
            pass

    # PDF parsing removed
    # Rebuild unique items from first_seen
    if first_seen:
        items = [
            {
                "Code": sym,
                "Name": meta.get("Name") or sym,
                "Country": "Canada",
                "Exchange": "",
                "Currency": meta.get("Currency") or "CAD",
                "Type": "Fund",
                "Isin": "",
                "five_stars": True,
            }
            for sym, meta in first_seen.items()
        ]
        try:
            _logger.info(
                "import_five_stars_canada: unique symbols=%d", int(len(items))
            )
        except Exception:
            pass
    # Enrich via EODHD search before upsert
    try:
        import os
        import time as _time
        import requests as _req  # type: ignore
        api_key = (
            os.getenv("EODHD_API_TOKEN", "")
            or os.getenv("EODHD_API_KEY", "")
        )
        if not api_key:
            _logger.warning("EODHD token missing; using parsed fields only")
        else:
            enriched: list[dict] = []
            total_items = len(items)
            try:
                verbose = (
                    os.getenv("EODHD_ENRICH_VERBOSE", "1").lower()
                    in ("1", "true", "yes")
                )
            except Exception:
                verbose = True
            for idx, it in enumerate(items):
                sym = it.get("Code") or ""
                if not sym:
                    continue
                try:
                    url = (
                        f"https://eodhistoricaldata.com/api/search/{sym}"
                    )
                    if verbose:
                        try:
                            _logger.info(
                                "enrich request %d/%d sym=%s",
                                int(idx + 1), int(total_items), sym,
                            )
                        except Exception:
                            pass
                    resp = _req.get(
                        url,
                        params={"api_token": api_key, "limit": 20},
                        timeout=20,
                    )
                    if resp.ok:
                        data = resp.json() if resp.content else []
                    else:
                        if verbose:
                            try:
                                _logger.info(
                                    "enrich resp sym=%s http=%s",
                                    sym, str(resp.status_code),
                                )
                            except Exception:
                                pass
                        data = []
                except Exception as _err:
                    if verbose:
                        try:
                            _logger.info(
                                "enrich resp sym=%s error=%s",
                                sym, str(_err),
                            )
                        except Exception:
                            pass
                    data = []
                # pick best Canada match
                best = None
                try:
                    for row in data or []:
                        co = str(row.get("Country") or "").lower()
                        code = str(row.get("Code") or "").upper()
                        if co in ("ca", "canada") and (
                            code.startswith(sym + ".")
                            or code.startswith(sym + ":")
                            or code == sym
                        ):
                            best = row
                            break
                    if best is None:
                        # try suffix attempts
                        for suf in (".TO", ".V", ".CN", ".NE"):
                            for row in data or []:
                                code = str(row.get("Code") or "").upper()
                                if code == f"{sym}{suf}":
                                    best = row
                                    break
                            if best is not None:
                                break
                except Exception:
                    best = None
                if isinstance(best, dict):
                    if verbose:
                        try:
                            _logger.info(
                                "enrich pick sym=%s code=%s exch=%s",
                                sym,
                                str(best.get("Code") or ""),
                                str(best.get("Exchange") or ""),
                            )
                        except Exception:
                            pass
                    try:
                        it["Name"] = (
                            str(best.get("Name"))
                            if best.get("Name") is not None
                            else it.get("Name")
                        )
                        it["Exchange"] = (
                            str(best.get("Exchange"))
                            if best.get("Exchange") is not None
                            else it.get("Exchange")
                        )
                        it["Country"] = (
                            str(best.get("Country"))
                            if best.get("Country") is not None
                            else it.get("Country")
                        )
                        cur = best.get("Currency")
                        if cur:
                            it["Currency"] = str(cur)
                        typ = best.get("Type")
                        if typ:
                            it["Type"] = str(typ)
                    except Exception:
                        pass
                else:
                    if verbose:
                        try:
                            _logger.info(
                                "enrich pick sym=%s none",
                                sym,
                            )
                        except Exception:
                            pass
                enriched.append(it)
                # tiny sleep to be gentle with provider
                _time.sleep(0.03)
            items = enriched
            _logger.info(
                "import_five_stars_canada: enriched via EODHD=%d",
                int(len(items)),
            )
    except Exception:
        _logger.warning("EODHD enrichment failed; using parsed items")

    flushed = upsert_price_series_bulk(items)
    total += flushed
    # Force-set five_stars=1 and country=Canada for parsed symbols
    try:
        sess2 = get_db_session()
        if sess2 is not None and parsed_symbols:
            from core.models import Symbols as _PS
            # chunk symbols to avoid parameter limits
            syms = sorted(parsed_symbols)
            CH = 500
            updated = 0
            updated_plain = 0
            updated_suffix = 0
            suffixes = [".TO", ".V", ".CN", ".NE", ":TO", ":V", ":CN", ":NE"]
            for i in range(0, len(syms), CH):
                chunk = syms[i : i + CH]
                # 1) plain symbols
                rows_plain = sess2.query(_PS).filter(_PS.symbol.in_(chunk)).all()  # type: ignore[arg-type]
                for r in rows_plain:
                    try:
                        r.five_stars = 1
                        r.country = r.country or "Canada"
                        updated += 1
                        updated_plain += 1
                    except Exception:
                        continue
                # 2) symbols with common CA suffixes
                exp: list[str] = []
                for s in chunk:
                    for suf in suffixes:
                        exp.append(f"{s}{suf}")
                if exp:
                    rows_suf = (
                        sess2.query(_PS)
                        .filter(_PS.symbol.in_(exp))  # type: ignore[arg-type]
                        .all()
                    )
                    for r in rows_suf:
                        try:
                            r.five_stars = 1
                            r.country = r.country or "Canada"
                            updated += 1
                            updated_suffix += 1
                        except Exception:
                            continue
                sess2.commit()
            _logger.info(
                (
                    "import_five_stars_canada: enforced five_stars total=%d "
                    "(plain=%d, suffix=%d)"
                ),
                int(updated),
                int(updated_plain),
                int(updated_suffix),
            )
            try:
                sess2.close()
            except Exception:
                pass
    except Exception:
        pass
    try:
        _logger.info(
            "import_five_stars_canada: total upserted=%d", int(total)
        )
    except Exception:
        pass
    # Post-verify in DB: how many CA five_stars are present overall
    try:
        sess = get_db_session()
        if sess is not None:
            from core.models import Symbols as _PS
            cnt = (
                sess.query(_PS)
                .filter((_PS.country == "Canada") & (_PS.five_stars == 1))  # type: ignore
                .count()
            )
            _logger.info(
                "import_five_stars_canada: DB count five_stars(CA)=%d",
                int(cnt),
            )
            # Sample a few symbols for visibility
            rows = (
                sess.query(_PS.symbol)
                .filter((_PS.country == "Canada") & (_PS.five_stars == 1))  # type: ignore
                .order_by(_PS.symbol.asc())
                .limit(10)
                .all()
            )
            _logger.info(
                "import_five_stars_canada: sample=%s",
                ",".join([r[0] for r in rows]) if rows else "",
            )
            try:
                sess.close()
            except Exception:
                pass
    except Exception:
        pass
    return int(total)


def mark_five_stars(db_ready: bool) -> None:
    """Set `five_stars=1` for symbols listed in `symbols/five_stars.txt`.

    - Creates missing `price_series` rows if a symbol is not present
    - Safe to run on every startup
    """
    if not db_ready:
        return
    session = get_db_session()
    if session is None:
        return
    try:
        # Preferred: backend/symbols/five_stars.txt (project-local)
        stars_path = _P(__file__).parents[1] / "symbols" / "five_stars.txt"
        # Fallbacks: repo-root symbols/ and legacy backend/data paths
        if not stars_path.exists():
            stars_path = _P(__file__).parents[2] / "symbols" / "five_stars.txt"
        if not stars_path.exists():
            stars_path = (
                _P(__file__).parents[1] / "backend" / "data" / "five_stars.txt"
            )
        if not stars_path.exists():
            stars_path = (
                _P(__file__).parents[1] / "backend" / "data" / "5stars.txt"
            )
        if not stars_path.exists():
            stars_path = _P(__file__).parents[0] / "data" / "5stars.txt"
        if not stars_path.exists():
            return
        raw = stars_path.read_text(encoding="utf-8")
        symbols = [s.strip().upper() for s in raw.splitlines() if s.strip()]
        if not symbols:
            return
        now = _dt.utcnow()
        existing = {
            r.symbol: r
            for r in session.query(Symbols)
            .filter(Symbols.symbol.in_(symbols))  # type: ignore[arg-type]
            .all()
        }
        to_add = []
        for sym in symbols:
            rec = existing.get(sym)
            if rec is None:
                rec = Symbols(
                    symbol=sym,
                    name=sym,
                    created_at=now,
                    updated_at=now,
                    five_stars=1,
                )
                to_add.append(rec)
            else:
                try:
                    rec.five_stars = 1
                    rec.updated_at = now
                except Exception:
                    pass
        if to_add:
            session.add_all(to_add)
        session.commit()
    finally:
        try:
            session.close()
        except Exception:
            pass


def reconcile_five_stars_usa(db_ready: bool) -> dict:
    """Reconcile USA five-star flags in DB against symbols/five_stars.txt.

    - Reads `backend/symbols/five_stars.txt` as the source of truth (US list)
    - For DB rows with country in {US, USA, United States}:
      - Sets five_stars=0 if symbol not in the list
      - Sets five_stars=1 if symbol is in the list
    - Creates missing rows for symbols from the file (US) when absent

    Returns a summary dict with counts.
    """
    summary = {
        "total_file": 0,
        "db_us_rows": 0,
        "unset_count": 0,
        "set_count": 0,
        "created_count": 0,
        "unchanged_count": 0,
    }
    if not db_ready:
        return summary
    session = get_db_session()
    if session is None:
        return summary
    try:
        try:
            _logger.info("reconcile_five_stars_usa: start")
        except Exception:
            pass
        # Load stars file
        stars_path = _P(__file__).parents[1] / "symbols" / "five_stars.txt"
        if not stars_path.exists():
            try:
                _logger.info(
                    "reconcile_five_stars_usa: file not found"
                )
            except Exception:
                pass
            return summary
        raw = stars_path.read_text(encoding="utf-8")
        symbols = [s.strip().upper() for s in raw.splitlines() if s.strip()]
        # Normalize file symbols to base tickers (strip .US / :US etc.)
        def _to_base(sym: str) -> str:
            try:
                s = str(sym).upper().strip()
                if "." in s:
                    s = s.split(".")[0]
                if ":" in s:
                    s = s.split(":")[0]
                return s
            except Exception:
                return str(sym).upper().strip()
        stars_set = set(_to_base(s) for s in symbols)
        summary["total_file"] = len(stars_set)
        try:
            _logger.info(
                "reconcile_five_stars_usa: loaded file=%s symbols=%d",
                stars_path.name,
                int(len(stars_set)),
            )
        except Exception:
            pass

        # Fetch US rows plus candidates with NULL/empty country
        US_COUNTRIES = ["US", "USA", "United States"]
        try:
            from sqlalchemy import or_ as _or  # type: ignore
        except Exception:
            _or = None  # type: ignore
        if _or is not None:
            rows = (
                session.query(Symbols)
                .filter(
                    _or(
                        Symbols.country.in_(US_COUNTRIES),  # type: ignore
                        Symbols.country == None,  # noqa: E711
                        Symbols.country == "",
                    )
                )
                .all()
            )
        else:
            rows = (
                session.query(Symbols)
                .filter(Symbols.country.in_(US_COUNTRIES))  # type: ignore
                .all()
            )
        summary["db_us_rows"] = len(rows)
        try:
            _logger.info(
                "reconcile_five_stars_usa: db_us_rows=%d",
                int(summary["db_us_rows"]),
            )
        except Exception:
            pass

        # Build indices by exact symbol and by base symbol
        by_sym = {
            r.symbol.upper(): r
            for r in rows
            if isinstance(r.symbol, str)
        }
        by_base: dict[str, list[Symbols]] = {}
        for r in rows:
            try:
                k = _to_base(r.symbol)
                by_base.setdefault(k, []).append(r)
            except Exception:
                continue

        unset_syms: list[str] = []
        set_syms: list[str] = []
        norm_syms: list[str] = []

        # Unset flags for US rows not present in file
        for sym, rec in by_sym.items():
            try:
                base = _to_base(sym)
                if rec.five_stars and base not in stars_set:
                    rec.five_stars = 0
                    summary["unset_count"] += 1
                    unset_syms.append(sym)
                elif (not rec.five_stars) and base in stars_set:
                    rec.five_stars = 1
                    summary["set_count"] += 1
                    set_syms.append(sym)
                    # Normalize country if missing/empty
                    try:
                        if (
                            rec.country is None
                            or str(rec.country).strip() == ""
                        ):
                            rec.country = "US"
                            norm_syms.append(sym)
                    except Exception:
                        pass
                else:
                    summary["unchanged_count"] += 1
            except Exception:
                continue

        # Create missing rows for any symbols from file not in DB US set
        now = _dt.utcnow()
        to_create: list[Symbols] = []  # type: ignore[name-defined]
        created_syms: list[str] = []
        for sym in stars_set:
            # Create only if we have no row with this base at all
            if sym not in by_base:
                try:
                    r = Symbols(
                        symbol=sym,
                        name=sym,
                        country="US",
                        exchange=None,
                        currency=None,
                        instrument_type=None,
                        isin=None,
                        five_stars=1,
                        created_at=now,
                        updated_at=now,
                    )
                    to_create.append(r)
                    created_syms.append(sym)
                except Exception:
                    continue

        if to_create:
            try:
                session.add_all(to_create)
            except Exception:
                pass
        try:
            session.commit()
        except Exception:
            try:
                session.rollback()
            except Exception:
                pass
        summary["created_count"] = len(to_create)
        # Log summary and a short sample of changes
        try:
            _logger.info(
                (
                    "reconcile_five_stars_usa: done file=%d db_us=%d "
                    "unset=%d set=%d created=%d unchanged=%d"
                ),
                int(summary["total_file"]),
                int(summary["db_us_rows"]),
                int(summary["unset_count"]),
                int(summary["set_count"]),
                int(summary["created_count"]),
                int(summary["unchanged_count"]),
            )

            def _sample(lst: list[str]) -> str:
                try:
                    if not lst:
                        return ""
                    lim = 20
                    if len(lst) <= lim:
                        return ",".join(sorted(lst))
                    head = ",".join(sorted(lst)[:lim])
                    return f"{head} (+{len(lst)-lim} more)"
                except Exception:
                    return ""
            if unset_syms:
                _logger.info(
                    "reconcile_five_stars_usa: unset -> %s",
                    _sample(unset_syms),
                )
            if set_syms:
                _logger.info(
                    "reconcile_five_stars_usa: set -> %s",
                    _sample(set_syms),
                )
            if norm_syms:
                _logger.info(
                    "reconcile_five_stars_usa: normalized country -> %s",
                    _sample(norm_syms),
                )
            if created_syms:
                _logger.info(
                    "reconcile_five_stars_usa: created -> %s",
                    _sample(created_syms),
                )
        except Exception:
            pass
        return summary
    finally:
        try:
            session.close()
        except Exception:
            pass
