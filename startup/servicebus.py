from __future__ import annotations

from services.sb_consumer import start_consumer_loop as _sb_start_loop


def start_servicebus_consumer() -> None:
    try:
        _sb_start_loop()
    except Exception:
        pass
