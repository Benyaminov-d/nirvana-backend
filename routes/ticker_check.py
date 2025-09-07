import io
import csv
import logging
from typing import List, Dict, Any, Optional

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from sqlalchemy import func

from core.db import get_db_session
from core.models import PriceSeries
from core.persistence import upsert_price_series_bulk
from utils.auth import require_pub_or_basic as _require_pub_or_basic

router = APIRouter()
_logger = logging.getLogger("ticker_check")


@router.get("/ticker/check", response_class=HTMLResponse)
def ticker_check_page(
    _auth: None = Depends(_require_pub_or_basic)
) -> HTMLResponse:
    """Ticker Check page - compare CSV tickers with database"""

    # Simple HTML page that loads React component
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Ticker Check - Nirvana App</title>
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont,
                'Segoe UI', system-ui, sans-serif;
                margin: 0; padding: 0;
            }
            .container {
                padding: 20px; max-width: 1400px; margin: 0 auto;
            }
            .loading {
                text-align: center; padding: 40px; color: #666;
            }
        </style>
    </head>
    <body>
        <div id="ticker-check-root">
            <div class="container">
                <div class="loading">Loading Ticker Check...</div>
            </div>
        </div>
        <script type="module">
            // Will be replaced with actual React component
            console.log('Ticker Check page loaded');
        </script>
    </body>
    </html>
    """

    return HTMLResponse(content=html_content)


@router.get("/api/ticker/check/countries")
def get_countries(
    _auth: None = Depends(_require_pub_or_basic)
) -> Dict[str, Any]:
    """Get list of available countries from database"""

    session = get_db_session()
    if session is None:
        raise HTTPException(500, "Database not available")

    try:
        # Get distinct countries with counts
        countries = (
            session.query(
                PriceSeries.country,
                func.count(PriceSeries.id).label('count')
            )
            .filter(PriceSeries.country.isnot(None))
            .group_by(PriceSeries.country)
            .order_by(func.count(PriceSeries.id).desc())
            .all()
        )

        result = [
            {"code": country, "name": country, "count": count}
            for country, count in countries
        ]

        return {"countries": result}

    except Exception as e:
        raise HTTPException(500, f"Failed to get countries: {str(e)}")
    finally:
        try:
            session.close()
        except Exception:
            pass


@router.get("/api/ticker/check/instrument-types")
def get_instrument_types(
    _auth: None = Depends(_require_pub_or_basic)
) -> Dict[str, Any]:
    """Get list of available instrument types from database"""

    session = get_db_session()
    if session is None:
        raise HTTPException(500, "Database not available")

    try:
        # Get distinct instrument types with counts
        types = (
            session.query(
                PriceSeries.instrument_type,
                func.count(PriceSeries.id).label('count')
            )
            .filter(PriceSeries.instrument_type.isnot(None))
            .group_by(PriceSeries.instrument_type)
            .order_by(func.count(PriceSeries.id).desc())
            .all()
        )

        result = [
            {"type": inst_type, "name": inst_type, "count": count}
            for inst_type, count in types
        ]

        return {"instrument_types": result}

    except Exception as e:
        raise HTTPException(500, f"Failed to get instrument types: {str(e)}")
    finally:
        try:
            session.close()
        except Exception:
            pass


@router.get("/api/ticker/check/symbols")
def get_symbols(
    country: Optional[str] = None,
    instrument_types: Optional[str] = None,
    _auth: None = Depends(_require_pub_or_basic)
) -> Dict[str, Any]:
    """Get symbols from database by country and instrument types"""

    session = get_db_session()
    if session is None:
        raise HTTPException(500, "Database not available")

    try:
        query = session.query(PriceSeries)

        # Apply country filter
        if country:
            query = query.filter(PriceSeries.country == country)

        # Apply instrument types filter
        if instrument_types:
            types_list = [
                t.strip() for t in instrument_types.split(',') if t.strip()
            ]
            if types_list:
                query = query.filter(
                    PriceSeries.instrument_type.in_(types_list)
                )

        # Get symbols with relevant info
        symbols = query.order_by(PriceSeries.symbol).all()

        result = []
        for symbol in symbols:
            result.append({
                "symbol": symbol.symbol,
                "name": symbol.name,
                "country": symbol.country,
                "exchange": symbol.exchange,
                "currency": symbol.currency,
                "instrument_type": symbol.instrument_type,
                "isin": symbol.isin,
                "insufficient_history": symbol.insufficient_history,
                "five_stars": symbol.five_stars
            })

        return {"symbols": result, "count": len(result)}

    except Exception as e:
        raise HTTPException(500, f"Failed to get symbols: {str(e)}")
    finally:
        try:
            session.close()
        except Exception:
            pass


@router.post("/api/ticker/check/parse-csv")
def parse_csv(
    file: UploadFile = File(...),
    country: Optional[str] = Form(None),
    instrument_types: Optional[str] = Form(None),
    _auth: None = Depends(_require_pub_or_basic)
) -> Dict[str, Any]:
    """Parse CSV file and compare with database symbols"""

    if not file.filename or not file.filename.endswith('.csv'):
        raise HTTPException(400, "Please upload a CSV file")

    try:
        # Read CSV file
        contents = file.file.read().decode('utf-8')
        csv_reader = csv.DictReader(io.StringIO(contents))

        # Parse CSV rows
        csv_symbols = []
        for row in csv_reader:
            # Try common column names for symbol
            symbol = None
            symbol_col_names = [
                'symbol', 'Symbol', 'SYMBOL', 'code', 'Code', 'CODE',
                'ticker', 'Ticker', 'Ticker/Share class', 'ticker/share class',
                'share class', 'Share class', 'Share Class'
            ]
            for col_name in symbol_col_names:
                if col_name in row and row[col_name]:
                    symbol = row[col_name].strip().upper()
                    break

            if symbol:
                csv_record = {
                    "symbol": symbol,
                    "name": (
                        row.get('Fund name') or row.get('fund name') or
                        row.get('Fund Name') or row.get('FUND NAME') or
                        row.get('name') or row.get('Name') or
                        row.get('NAME') or ''
                    ),
                    "country": (
                        row.get('Domicile') or row.get('domicile') or
                        row.get('DOMICILE') or
                        row.get('country') or row.get('Country') or
                        row.get('COUNTRY') or country or ''
                    ),
                    "exchange": (
                        row.get('exchange') or row.get('Exchange') or
                        row.get('EXCHANGE') or ''
                    ),
                    "currency": (
                        row.get('currency') or row.get('Currency') or
                        row.get('CURRENCY') or ''
                    ),
                    "instrument_type": (
                        row.get('Type') or row.get('type') or
                        row.get('TYPE') or row.get('instrument_type') or ''
                    ),
                    "isin": (
                        row.get('ISIN') or row.get('isin') or
                        row.get('Isin') or ''
                    ),
                }
                csv_symbols.append(csv_record)

        if not csv_symbols:
            raise HTTPException(400, "No valid symbols found in CSV file")

        # Get database symbols for comparison
        session = get_db_session()
        if session is None:
            raise HTTPException(500, "Database not available")

        try:
            # Build query for database symbols
            query = session.query(PriceSeries)

            if country:
                query = query.filter(PriceSeries.country == country)

            if instrument_types:
                types_list = [
                    t.strip() for t in instrument_types.split(',') if t.strip()
                ]
                if types_list:
                    query = query.filter(
                        PriceSeries.instrument_type.in_(types_list)
                    )

            db_symbols = query.all()

            # Convert to dict for fast lookup
            db_symbols_dict = {symbol.symbol: symbol for symbol in db_symbols}

            # Compare CSV symbols with database
            csv_symbols_set = {record['symbol'] for record in csv_symbols}
            db_symbols_set = set(db_symbols_dict.keys())

            # Find matches and differences
            matches = csv_symbols_set & db_symbols_set
            csv_only = csv_symbols_set - db_symbols_set  # CSV but not DB
            db_only = db_symbols_set - csv_symbols_set   # DB but not CSV

            # Prepare response data
            csv_data = []
            for record in csv_symbols:
                status = (
                    "match" if record['symbol'] in matches
                    else "missing_in_db"
                )
                csv_data.append({**record, "status": status})

            db_data = []
            for symbol in db_symbols:
                status = (
                    "match" if symbol.symbol in matches
                    else "missing_in_csv"
                )
                db_data.append({
                    "symbol": symbol.symbol,
                    "name": symbol.name,
                    "country": symbol.country,
                    "exchange": symbol.exchange,
                    "currency": symbol.currency,
                    "instrument_type": symbol.instrument_type,
                    "isin": symbol.isin,
                    "insufficient_history": symbol.insufficient_history,
                    "five_stars": symbol.five_stars,
                    "status": status
                })

            summary = {
                "total_csv": len(csv_symbols),
                "total_db": len(db_symbols),
                "matches": len(matches),
                "csv_only": len(csv_only),
                "db_only": len(db_only),
                "match_percentage": (
                    round(len(matches) / len(csv_symbols) * 100, 1)
                    if csv_symbols else 0
                )
            }

            return {
                "summary": summary,
                "csv_data": csv_data,
                "db_data": db_data,
                "missing_symbols": [
                    record for record in csv_data
                    if record['status'] == 'missing_in_db'
                ]
            }

        finally:
            try:
                session.close()
            except Exception:
                pass

    except Exception as e:
        _logger.exception("Failed to parse CSV")
        raise HTTPException(500, f"Failed to parse CSV: {str(e)}")


@router.post("/api/ticker/check/sync")
def sync_missing_symbols(
    symbols: List[Dict[str, Any]],
    _auth: None = Depends(_require_pub_or_basic)
) -> Dict[str, Any]:
    """Sync missing symbols from CSV to database"""

    if not symbols:
        raise HTTPException(400, "No symbols provided for sync")

    try:
        # Validate and prepare symbols for bulk upsert
        items = []
        for symbol_data in symbols:
            if not symbol_data.get('symbol'):
                continue

            item = {
                "Code": symbol_data.get('symbol', '').strip().upper(),
                "Name": symbol_data.get('name', '').strip(),
                "Country": symbol_data.get('country', '').strip(),
                "Exchange": symbol_data.get('exchange', '').strip(),
                "Currency": symbol_data.get('currency', '').strip(),
                "Type": symbol_data.get('instrument_type', '').strip(),
                "Isin": symbol_data.get('isin', '').strip(),
            }
            items.append(item)

        if not items:
            raise HTTPException(400, "No valid symbols to sync")

        # Use existing bulk upsert function
        count = upsert_price_series_bulk(items)

        _logger.info("Synced %d symbols from CSV to database", count)

        return {
            "success": True,
            "synced_count": count,
            "message": f"Successfully synced {count} symbols to database"
        }

    except Exception as e:
        _logger.exception("Failed to sync symbols")
        raise HTTPException(500, f"Failed to sync symbols: {str(e)}")