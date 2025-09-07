from typing import Optional
import threading


nightly_last_run: Optional[str] = None

# Shared ticker state
submitted_symbols: set[str] = set()
ticker_lock = threading.Lock()
