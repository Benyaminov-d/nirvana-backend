from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse

from utils.auth import (
    basic_auth_if_configured as _basic_auth_if_configured,
    require_pub_or_basic as _require_pub_or_basic,
)
from services.compass_recommendations_service import (
    CompassRecommendationsService,
)


router = APIRouter()


@router.get("/products", response_class=HTMLResponse)
def products_page(
    _auth: None = Depends(_basic_auth_if_configured),
) -> HTMLResponse:
    from pathlib import Path as _Path

    page = (
        _Path(__file__).parents[2]
        / "frontend"
        / "templates"
        / "products.html"
    )
    if page.exists():
        return HTMLResponse(page.read_text(), 200)
    raise HTTPException(404, "products.html not found")


@router.get("/products/search")
def products_search(
    tolerance: float = Query(
        ...,
        description=(
            "Loss tolerance as fraction (e.g., 0.2) or percent (e.g., 20)"
        ),
    ),
    alpha: int = Query(99, description="Alpha label: 95 or 99 (default 99)"),
    five_stars: bool = Query(
        False, description="Restrict to five-star list"
    ),
    country: str | None = Query(
        None,
        description=(
            "Restrict to instruments where PriceSeries.country matches"
        ),
    ),
    limit: int = Query(
        20,
        description="Top N results sorted by Compass Score",
    ),
    _auth: None = Depends(_require_pub_or_basic),
) -> JSONResponse:
    """Search products using Arman's algorithm."""
    # Normalize tolerance: allow percent input like 20 meaning 0.20
    try:
        tol = float(tolerance)
        if tol > 1.0:
            tol = tol / 100.0
        if tol < 0:
            tol = 0.0
    except Exception:
        raise HTTPException(400, "invalid tolerance")

    if alpha not in (95, 99):
        raise HTTPException(400, "alpha must be one of 95, 99")

    # Use Arman's algorithm via service with environment-aware configuration
    # Note: alpha_label and max_results can be overridden via environment variables
    # RecommendationConfig moved to compass_recommendations_service
    
    # Create service with default configuration
    service = CompassRecommendationsService()

    try:
        # Convert tolerance back to percentage for service
        loss_tolerance_pct = -100.0 * tol

        result = service.get_recommendations(
            loss_tolerance_pct=loss_tolerance_pct,
            country=country or "US",
            seed_symbol=None,
        )

        # Convert service response to products API format
        items = []
        for r in result.get("results", []):
            # Add extra fields expected by products API
            item = dict(r)
            item["cvar_worst"] = None  # Not available in service response
            item["return_annual"] = (
                r["annualized_return"]["value_pct"] / 100.0
                if r["annualized_return"]["value_pct"] is not None
                else None
            )
            items.append(item)

        return JSONResponse(
            {
                "items": items,
                "count": len(items),
                "alpha": alpha,
                "tolerance": tol,
                "country": country,
                "algorithm": "arman_winsorized",
                "metadata": result.get("metadata", {}),
            }
        )

    except Exception as e:
        raise HTTPException(
            500, f"Products search service error: {str(e)}"
        )


@router.get("/products/countries")
def products_countries(
    _auth: None = Depends(_require_pub_or_basic),
) -> JSONResponse:
    """Return distinct non-null countries from PriceSeries (sorted)."""
    from core.db import get_db_session
    from core.models import PriceSeries
    from sqlalchemy import func  # type: ignore

    sess = get_db_session()
    if sess is None:
        raise HTTPException(501, "Database not configured")
    try:
        rows = (
            sess.query(func.distinct(PriceSeries.country))
            .filter(PriceSeries.country.isnot(None))  # type: ignore
            .all()
        )
        vals = sorted([c for (c,) in rows if c])
        return JSONResponse({"countries": vals})
    finally:
        try:
            sess.close()
        except Exception:
            pass


@router.get("/products/instrument_types")
def products_instrument_types(
    _auth: None = Depends(_require_pub_or_basic),
) -> JSONResponse:
    """Return distinct non-null instrument types per country (sorted)."""
    from core.db import get_db_session
    from core.models import PriceSeries

    sess = get_db_session()
    if sess is None:
        raise HTTPException(501, "Database not configured")
    try:
        rows = (
            sess.query(
                PriceSeries.country, PriceSeries.instrument_type
            )
            .filter(
                PriceSeries.country.isnot(None),  # type: ignore
                PriceSeries.instrument_type.isnot(None),  # type: ignore
            )
            .distinct()
            .all()
        )
        by_country: dict[str, set[str]] = {}
        for c, t in rows:
            if not c or not t:
                continue
            by_country.setdefault(str(c), set()).add(str(t))
        result = {k: sorted(list(v)) for k, v in sorted(by_country.items())}
        return JSONResponse(result)
    finally:
        try:
            sess.close()
        except Exception:
            pass
