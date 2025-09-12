from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from services.infrastructure.azure_service_bus_client import AzureServiceBusClient, QueueMessage, MessagePriority
from utils.service_bus import sb_connection_string, sb_symbols_queue
from repositories.price_series_repository import PriceSeriesRepository


@dataclass
class UniverseFilter:
    five_stars: bool = False
    country: Optional[str] = None
    instrument_types: Optional[List[str]] = None
    exclude_exchanges: Optional[List[str]] = None
    limit: Optional[int] = None


class QueueOrchestrationService:
    def __init__(self) -> None:
        conn = sb_connection_string()
        if not conn:
            raise RuntimeError("Service Bus connection string not configured")
        self.client = AzureServiceBusClient(connection_string=conn)
        if not self.client.connect():
            raise RuntimeError("Failed to connect to Service Bus")

    def enqueue_symbol_batch(self, symbols: List[str], *, source: str = "eodhd", as_of: Optional[str] = None) -> Dict[str, Any]:
        if not symbols:
            return {"success": True, "batches": 0, "messages": 0}
        q = sb_symbols_queue()
        if not q:
            raise RuntimeError("Symbols queue not configured")

        # Qualify symbols with market suffix when possible (to improve EODHD resolution)
        def _qualify_symbols(sym_list: List[str]) -> List[str]:
            # Skip already-qualified entries like "SYM:.TO"
            pending = [s for s in sym_list if s and ":" not in s]
            if not pending:
                return sym_list
            repo = PriceSeriesRepository()

            # For precision, look up exchange and country per symbol (small N in TEST mode)
            exch_country: dict[str, tuple[str, str]] = {}
            for s in pending:
                try:
                    ex, co = repo.get_symbol_exchange_country(s)
                    exch_country[s.upper()] = (
                        (ex or "").strip().upper(),
                        (co or "").strip().upper(),
                    )
                except Exception:
                    exch_country[s.upper()] = ("", "")

            qualified: List[str] = []
            for s in sym_list:
                if not s or ":" in s:
                    qualified.append(s)
                    continue
                ex, co = exch_country.get(s.upper(), ("", ""))
                suf = None
                if co in ("CANADA", "CA"):
                    suf = ".TO"
                elif ex in ("LSE",):
                    suf = ".LSE"
                elif co in ("UNITED KINGDOM", "UK", "GB", "GREAT BRITAIN"):
                    suf = ".L"
                # Add more mappings if needed (e.g., AU -> .AU, EU venues)
                if suf:
                    qualified.append(f"{s}:{suf}")
                else:
                    qualified.append(s)
            return qualified

        symbols = _qualify_symbols(symbols)
        batch_id = str(uuid.uuid4())
        corr = str(uuid.uuid4())
        messages: List[QueueMessage] = []
        max_per_message = int(os.getenv("SYMBOLS_PER_MESSAGE", "200"))
        for i in range(0, len(symbols), max_per_message):
            chunk = symbols[i : i + max_per_message]
            body = {
                "batch_id": batch_id,
                "batch_index": i // max_per_message,
                "batch_size": len(symbols),
                "symbols": chunk,
                "source": source,
                "as_of": as_of,
            }
            messages.append(
                QueueMessage(
                    id=None,
                    body=body,
                    priority=MessagePriority.NORMAL,
                    correlation_id=corr,
                    metadata={
                        "message_id": f"batch-{batch_id}-{i // max_per_message}",
                        "type": "symbols_batch",
                    },
                )
            )
        res = self.client.send_batch_messages(messages, queue_name=q, batch_size=int(os.getenv("SB_SEND_BATCH", "100")))
        return res | {"batch_id": batch_id, "correlation_id": corr}

    def start_universe_processing(self, filters: UniverseFilter) -> Dict[str, Any]:
        # Defer to repository to fetch symbol list
        from repositories.price_series_repository import PriceSeriesRepository

        repo = PriceSeriesRepository()
        symbols = repo.get_symbols_by_filters(
            five_stars=filters.five_stars,
            ready_only=True,
            include_unknown=False,
            country=filters.country,
            instrument_types=filters.instrument_types,
            exclude_exchanges=filters.exclude_exchanges,
            limit=filters.limit,
        )
        return self.enqueue_symbol_batch(symbols)


