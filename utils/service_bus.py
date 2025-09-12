import os
import logging

logger = logging.getLogger(__name__)

def sb_connection_string() -> str | None:
    conn_str = os.getenv("SB_CONNECTION") or os.getenv("SERVICEBUS_CONNECTION")
    if not conn_str:
        logger.warning("Service Bus connection string not found in environment variables")
    return conn_str


def sb_queue_name() -> str | None:
    queue_name = os.getenv("SB_QUEUE") or os.getenv("SERVICEBUS_QUEUE")
    if not queue_name:
        logger.warning("Service Bus queue name not found in environment variables")
    return queue_name


def _sb_results_queue() -> str | None:
    """Get the name of the results queue."""
    return os.getenv("SB_RESULTS_QUEUE") or os.getenv("SERVICEBUS_RESULTS_QUEUE")


def sb_symbols_queue() -> str | None:
    return os.getenv("SYMBOLS_QUEUE") or os.getenv("SB_QUEUE") or os.getenv("SERVICEBUS_QUEUE")


def sb_validation_results_queue() -> str | None:
    return os.getenv("VALIDATION_RESULTS_QUEUE")


def sb_cvar_calculations_queue() -> str | None:
    return os.getenv("CVAR_CALCULATIONS_QUEUE")


def sb_cvar_results_queue() -> str | None:
    return os.getenv("CVAR_RESULTS_QUEUE") or _sb_results_queue()


def sb_compass_params_queue() -> str | None:
    return os.getenv("COMPASS_PARAMS_QUEUE")


def sb_compass_results_queue() -> str | None:
    return os.getenv("COMPASS_RESULTS_QUEUE")


def sb_series_write_queue() -> str | None:
    return os.getenv("SERIES_WRITE_QUEUE")


