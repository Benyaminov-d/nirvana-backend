"""
Country-specific ticker feed endpoint.

Implements the business logic for country-specific ticker data:
- US: Show five_stars=1 products only
- Other countries: Show country-specific products, fallback to US five_stars
  if insufficient
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Dict, Any
import logging
import random

from repositories.cvar_repository import CvarRepository
from repositories.query_builders import CvarQueryBuilder
from utils.auth import require_pub_or_basic as _require_auth

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/ticker/feed")
def ticker_feed_country_specific(
    country: str = Query(
        "US", description="Country code (US, UK, CA, etc.)"
    ),
    mode: str = Query(
        "five_stars", description="Mode: five_stars, cvar95, cvar99"
    ),
    limit: int = Query(
        20, description="Maximum number of items to return"
    ),
    _auth: None = Depends(_require_auth),
) -> Dict[str, Any]:
    """
    Get ticker feed data with country-specific logic.

    Business Rules:
    - US: Show only five_stars=1 products
    - Other countries: Show country-specific products, fallback to US
      five_stars if insufficient
    - Minimum 10 items required, maximum 20
    """

    try:
        cvar_repo = CvarRepository()
        query_builder = CvarQueryBuilder()

        # Get country-specific products first
        country_products = []
        if country.upper() != "US":
            country_products = query_builder.get_symbols_with_filters(
                country=country,
                five_stars=False,  # Don't require five_stars for non-US
                limit=limit
            )

        # Get US five_stars products
        us_five_stars = query_builder.get_symbols_with_filters(
            country="US",
            five_stars=True,
            limit=limit
        )

        # If no data found, return empty result with helpful message
        if not country_products and not us_five_stars:
            return {
                "items": [],
                "title_suffix": (
                    "No data available - please load symbols first"
                ),
                "country": country,
                "mode": mode,
                "total_items": 0,
                "country_products_count": 0,
                "us_five_stars_count": 0,
                "success": True,
                "message": (
                    "No symbols found in database. "
                    "Please load symbols from CSV files first."
                )
            }

        # Determine which products to use based on business rules
        final_products = []
        title_suffix = ""

        if country.upper() == "US":
            # US: Only five_stars products
            final_products = us_five_stars[:limit]
            title_suffix = (
                "MORNINGSTAR GOLD MEDALIST + 5-STAR RATED US MUTUAL FUNDS "
                "(99-CVAR, ANNUALISED)"
            )
        else:
            # Other countries: Country-specific first, then US five_stars if
            # needed
            if len(country_products) >= 10:
                # Sufficient country-specific products
                final_products = country_products[:limit]
                title_suffix = (
                    f"{country} ETF and MUTUAL FUNDS (99-CVAR, ANNUALISED) "
                    "Aug 22, 2025"
                )
            else:
                # Insufficient country products, mix with US five_stars
                final_products = country_products + us_five_stars
                final_products = final_products[:limit]
                title_suffix = (
                    f"MORNINGSTAR GOLD MEDALIST + 5-STAR RATED {country} "
                    "and US "
                    "ETF and MUTUAL FUNDS (99-CVAR, ANNUALISED) Aug 22, 2025"
                )
        # Get CVaR data for the selected products
        items = []
        for symbol in final_products:
            try:
                # Get latest CVaR data for this symbol
                cvar_snapshots = cvar_repo.get_latest_by_symbol(symbol)
                if cvar_snapshots:
                    # Find snapshots for different alpha levels
                    cvar95_snapshot = None
                    cvar99_snapshot = None
                    cvar50_snapshot = None

                    for snapshot in cvar_snapshots:
                        if snapshot.alpha_label == 95:
                            cvar95_snapshot = snapshot
                        elif snapshot.alpha_label == 99:
                            cvar99_snapshot = snapshot
                        elif snapshot.alpha_label == 50:
                            cvar50_snapshot = snapshot

                    # Use the most recent date from any snapshot
                    as_of_date = '2025-09-06'
                    if cvar_snapshots:
                        as_of_date = (
                            cvar_snapshots[0].as_of_date.strftime('%Y-%m-%d')
                            if cvar_snapshots[0].as_of_date
                            else '2025-09-06'
                        )

                    # Get CVaR values based on mode
                    cvar95_value = (
                        cvar95_snapshot.cvar_ghst
                        if cvar95_snapshot else None
                    )
                    cvar99_value = (
                        cvar99_snapshot.cvar_nig
                        if cvar99_snapshot else None
                    )
                    cvar50_value = (
                        cvar50_snapshot.cvar_evar
                        if cvar50_snapshot else None
                    )

                    # Determine which CVaR value to use for filtering based on mode
                    primary_cvar_value = None
                    if mode == 'cvar99' and cvar99_value is not None:
                        primary_cvar_value = cvar99_value
                    elif mode == 'cvar95' and cvar95_value is not None:
                        primary_cvar_value = cvar95_value
                    elif mode == 'five_stars' and cvar99_value is not None:
                        primary_cvar_value = cvar99_value

                    # Only include symbols with valid CVaR data and above -95%
                    if (primary_cvar_value is not None and
                            primary_cvar_value >= -0.95):  # -95% threshold
                        items.append({
                            "symbol": symbol,
                            "as_of": as_of_date,
                            "cvar95": cvar95_value,
                            "cvar99": cvar99_value,
                            "cvar50": cvar50_value,
                        })
                    else:
                        logger.debug(
                            f"Excluding {symbol}: CVaR value "
                            f"{primary_cvar_value} is null or below -95%"
                        )
                else:
                    logger.debug(f"Excluding {symbol}: No CVaR snapshots found")
            except Exception as e:
                logger.warning(
                    f"Failed to get CVaR data for {symbol}: {e}"
                )
                # Don't add symbols without data

        # If we have too few items after filtering, try to get more symbols
        if len(items) < 10 and len(final_products) < 50:
            logger.info(
                f"Only {len(items)} valid items found, trying to get more"
            )
            # Try to get more symbols with a higher limit
            additional_limit = min(50, limit * 3)

            if country.upper() == "US":
                additional_products = query_builder.get_symbols_with_filters(
                    country="US",
                    five_stars=True,
                    limit=additional_limit
                )
            else:
                additional_country_products = (
                    query_builder.get_symbols_with_filters(
                        country=country,
                        five_stars=False,
                        limit=additional_limit
                    )
                )
                additional_us_products = query_builder.get_symbols_with_filters(
                    country="US",
                    five_stars=True,
                    limit=additional_limit
                )
                additional_products = (
                    additional_country_products + additional_us_products
                )

            # Process additional symbols that weren't already processed
            processed_symbols = {item["symbol"] for item in items}
            for symbol in additional_products:
                if symbol in processed_symbols or len(items) >= limit:
                    continue

                try:
                    cvar_snapshots = cvar_repo.get_latest_by_symbol(symbol)
                    if cvar_snapshots:
                        # Find snapshots for different alpha levels
                        cvar95_snapshot = None
                        cvar99_snapshot = None
                        cvar50_snapshot = None

                        for snapshot in cvar_snapshots:
                            if snapshot.alpha_label == 95:
                                cvar95_snapshot = snapshot
                            elif snapshot.alpha_label == 99:
                                cvar99_snapshot = snapshot
                            elif snapshot.alpha_label == 50:
                                cvar50_snapshot = snapshot

                        # Get CVaR values
                        cvar95_value = (
                            cvar95_snapshot.cvar_ghst
                            if cvar95_snapshot else None
                        )
                        cvar99_value = (
                            cvar99_snapshot.cvar_nig
                            if cvar99_snapshot else None
                        )
                        cvar50_value = (
                            cvar50_snapshot.cvar_evar
                            if cvar50_snapshot else None
                        )

                        # Determine primary CVaR value for filtering
                        primary_cvar_value = None
                        if mode == 'cvar99' and cvar99_value is not None:
                            primary_cvar_value = cvar99_value
                        elif mode == 'cvar95' and cvar95_value is not None:
                            primary_cvar_value = cvar95_value
                        elif mode == 'five_stars' and cvar99_value is not None:
                            primary_cvar_value = cvar99_value

                        # Apply the same filtering criteria
                        if (primary_cvar_value is not None and
                                primary_cvar_value >= -0.95):

                            as_of_date = (
                                cvar_snapshots[0].as_of_date.strftime(
                                    '%Y-%m-%d'
                                )
                                if cvar_snapshots[0].as_of_date
                                else '2025-09-06'
                            )

                            items.append({
                                "symbol": symbol,
                                "as_of": as_of_date,
                                "cvar95": cvar95_value,
                                "cvar99": cvar99_value,
                                "cvar50": cvar50_value,
                            })
                            processed_symbols.add(symbol)
                except Exception as e:
                    logger.warning(
                        f"Failed to process additional symbol {symbol}: {e}"
                    )

        # Randomly shuffle items before returning
        random.shuffle(items)

        return {
            "items": items,
            "title_suffix": title_suffix,
            "country": country,
            "mode": mode,
            "total_items": len(items),
            "country_products_count": len(country_products),
            "us_five_stars_count": len(us_five_stars),
            "success": True
        }

    except Exception as e:
        logger.error(f"Failed to get ticker feed for {country}: {e}")
        raise HTTPException(500, f"Failed to get ticker data: {str(e)}")
