import os


def sb_connection_string() -> str | None:
    return os.getenv("SB_CONNECTION") or os.getenv("SERVICEBUS_CONNECTION")


def sb_queue_name() -> str | None:
    return os.getenv("SB_QUEUE") or os.getenv("SERVICEBUS_QUEUE")


