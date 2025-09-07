from __future__ import annotations

import os
from sqlalchemy import create_engine  # type: ignore
from sqlalchemy.orm import sessionmaker, declarative_base  # type: ignore


# DATABASE_URL example: postgresql+psycopg2://user:password@db:5432/nirvana
DATABASE_URL = os.getenv("DATABASE_URL", "")

Base = declarative_base()


def _make_engine():
    url = DATABASE_URL
    if not url:
        return None
    try:
        # Tune pool to avoid connection starvation and long waits
        # Values are conservative; can be adjusted via env if needed later
        engine = create_engine(
            url,
            pool_pre_ping=True,
            echo=False,
            future=True,
            pool_size=int(os.getenv("DB_POOL_SIZE", "20")),
            max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "40")),
            pool_timeout=int(os.getenv("DB_POOL_TIMEOUT", "5")),
            pool_recycle=int(os.getenv("DB_POOL_RECYCLE", "1800")),
        )
        return engine
    except Exception:
        return None


engine = _make_engine()
SessionLocal = sessionmaker(bind=engine) if engine is not None else None


def get_db_session():
    if SessionLocal is None:
        return None
    return SessionLocal()
