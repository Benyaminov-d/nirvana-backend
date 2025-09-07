from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import Response
from utils.auth import require_pub_or_basic as _require_pub_or_basic
import io
import os


router = APIRouter()


@router.get("/get_time_series")
def get_time_series(
    ticker: str,
    _auth: None = Depends(_require_pub_or_basic),
) -> Response:
    import requests  # type: ignore
    api_key = os.getenv("EODHD_API_KEY")
    if not api_key:
        raise HTTPException(501, "EODHD_API_KEY not set")
    m = {"BTC": "BTC-USD.CC", "ETH": "ETH-USD.CC", "SP500TR": "SP500TR.INDX"}
    from utils.common import resolve_eodhd_endpoint_symbol
    endpoint_symbol = m.get(ticker, resolve_eodhd_endpoint_symbol(ticker))
    url = (
        f"https://eodhistoricaldata.com/api/eod/{endpoint_symbol}"
    )
    try:
        resp = requests.get(
            url,
            params={"api_token": api_key, "fmt": "json"},
            timeout=60,
        )
        resp.raise_for_status()
        raw = resp.json()
    except Exception as exc:
        raise HTTPException(502, f"API error: {exc}")
    buf = io.StringIO()
    import csv as _csv
    w = _csv.writer(buf)
    # price picks adjusted_close when available/positive, else close
    w.writerow(["date", "price", "close", "adjusted_close", "volume"])
    for row in raw:
        try:
            d = str(row.get("date"))
            close_v = row.get("close")
            adj_v = row.get("adjusted_close")
            vol = row.get("volume")
            price_v = (
                adj_v if (adj_v is not None and float(adj_v) > 0) else close_v
            )
            w.writerow([d, price_v, close_v, adj_v, vol])
        except Exception:
            continue
    data = buf.getvalue().encode("utf-8")
    headers = {
        "Content-Disposition": (
            f"attachment; filename={ticker}_timeseries.csv"
        ),
    }
    return Response(
        content=data,
        media_type="text/csv",
        headers=headers,
    )


