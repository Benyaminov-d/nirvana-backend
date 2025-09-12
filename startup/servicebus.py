from __future__ import annotations

import logging
import threading

from services.sb_consumer import start_consumer_loop as _sb_start_loop
from services.sb_series_writer import start_series_writer_consumer as _sb_series_writer
from services.sb_validation_consumer import start_validation_consumer as _sb_validation
from services.sb_compass_results_consumer import start_compass_results_consumer as _sb_compass


def start_servicebus_consumer() -> None:
    """Start all Service Bus consumers in separate daemon threads.

    Important: individual consumer loops are blocking; they MUST run in their own threads.
    """
    _log = logging.getLogger("startup.servicebus")
    consumers = [
        (_sb_start_loop, "CvarResultsConsumer"),
        (_sb_series_writer, "SeriesWriterConsumer"),
        (_sb_validation, "ValidationResultsConsumer"),
        (_sb_compass, "CompassResultsConsumer"),
    ]
    for func, name in consumers:
        try:
            t = threading.Thread(target=func, name=name, daemon=True)
            t.start()
            _log.info("Service Bus consumer thread started: %s", name)
        except Exception:
            _log.exception("Failed to start Service Bus consumer: %s", name)
