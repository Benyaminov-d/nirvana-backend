"""
Time Series Repository - bulk upserts and queries for price_time_series.
"""

from __future__ import annotations

from typing import Iterable, List, Dict, Optional
from datetime import date

from sqlalchemy.orm import Session  # type: ignore
from sqlalchemy.dialects.postgresql import insert as pg_insert  # type: ignore

from repositories.base_repository import BaseRepository
from core.models import PriceTimeSeries


class TimeSeriesRepository(BaseRepository[PriceTimeSeries]):
    def __init__(self) -> None:
        super().__init__(PriceTimeSeries)

    def bulk_upsert(
        self,
        session: Session,
        *,
        symbol_id: int,
        version_id: str,
        source_type: str,
        rows: Iterable[Dict[str, object]],
        chunk_size: int = 5000,
    ) -> int:
        """
        Idempotent bulk upsert of daily rows for a symbol/version.

        rows items: {"date": date|str, "price": float|Decimal, "volume": int|None}
        Returns number of rows written/updated.
        """
        total = 0
        buf: List[Dict[str, object]] = []
        for r in rows:
            d = r.get("date")
            p = r.get("price")
            if not d or p is None:
                continue
            buf.append({
                "symbol_id": symbol_id,
                "date": d,
                "price": p,
                "volume": r.get("volume"),
                "source_type": source_type,
                "version_id": version_id,
            })
            if len(buf) >= chunk_size:
                total += self._upsert_chunk(session, buf)
                buf.clear()
        if buf:
            total += self._upsert_chunk(session, buf)
        return total

    def _upsert_chunk(self, session: Session, buf: List[Dict[str, object]]) -> int:
        stmt = pg_insert(PriceTimeSeries.__table__).values(buf)
        stmt = stmt.on_conflict_do_update(
            index_elements=[PriceTimeSeries.symbol_id, PriceTimeSeries.date, PriceTimeSeries.version_id],
            set_={
                "price": stmt.excluded.price,  # type: ignore[attr-defined]
                "volume": stmt.excluded.volume,  # type: ignore[attr-defined]
            },
        )
        res = session.execute(stmt)
        return int(res.rowcount or 0)

    def query_series(
        self,
        session: Session,
        *,
        symbol_id: int,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
        limit: Optional[int] = None,
    ) -> List[PriceTimeSeries]:
        q = session.query(PriceTimeSeries).filter(PriceTimeSeries.symbol_id == symbol_id)
        if from_date is not None:
            q = q.filter(PriceTimeSeries.date >= from_date)
        if to_date is not None:
            q = q.filter(PriceTimeSeries.date <= to_date)
        q = q.order_by(PriceTimeSeries.date.asc())
        if limit and limit > 0:
            q = q.limit(limit)
        return list(q.all())


