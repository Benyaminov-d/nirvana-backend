from __future__ import annotations
# flake8: noqa

from typing import Any, Dict, Optional
import logging as _log
import json
import math
from datetime import datetime, date

from sqlalchemy.exc import SQLAlchemyError  # type: ignore
from sqlalchemy import text, JSON, bindparam  # type: ignore

from core.db import get_db_session, Base, engine
from core.models import (
    CvarSnapshot,
    AnnualCvarViolation,
    Symbols,
    InsufficientDataEvent,
    PriceLast,
    AnomalyReport,
    User,
    AuthAttempt,
    Exchange,
)
from utils.common import canonical_instrument_type as _canon_type  # type: ignore
from utils.common import db_base_symbol as _db_base_symbol  # type: ignore


def init_db_if_configured() -> bool:
    """Create tables if `DATABASE_URL` configured; return True if ready."""
    if engine is None:
        return False
    try:
        Base.metadata.create_all(bind=engine)
        # best-effort, idempotent column widening for Postgres
        with engine.begin() as conn:
            try:
                ddl_lines = [
                    "DO $$",
                    "BEGIN",
                    "    IF EXISTS (SELECT 1 FROM information_schema.columns",
                    "                WHERE table_name='symbols'",
                    "                  AND column_name='symbol'",
                    "                  AND character_maximum_length < 128)",
                    "                  THEN",
                    "        ALTER TABLE symbols ALTER COLUMN",
                    "        symbol TYPE VARCHAR(128);",
                    "    END IF;",
                    "    IF EXISTS (SELECT 1 FROM information_schema.columns",
                    "                WHERE table_name='symbols'",
                    "                  AND column_name='name'",
                    "                  AND data_type <> 'text') THEN",
                    "        ALTER TABLE symbols ALTER COLUMN",
                    "        name TYPE TEXT;",
                    "    END IF;",
                    "    -- add five_stars flag to symbols if missing",
                    "    IF NOT EXISTS (SELECT 1 FROM information_schema.columns",
                    "        WHERE table_name='symbols' AND column_name='five_stars') THEN",
                    "        ALTER TABLE symbols ADD COLUMN five_stars INTEGER NOT NULL DEFAULT 0;",
                    "    END IF;",
                    "    -- add new EODHD metadata columns to symbols if missing",
                    "    IF NOT EXISTS (SELECT 1 FROM information_schema.columns",
                    "        WHERE table_name='symbols' AND column_name='country') THEN",
                    "        ALTER TABLE symbols ADD COLUMN country VARCHAR(64) NULL;",
                    "    END IF;",
                    "    IF NOT EXISTS (SELECT 1 FROM information_schema.columns",
                    "        WHERE table_name='symbols' AND column_name='exchange') THEN",
                    "        ALTER TABLE symbols ADD COLUMN exchange VARCHAR(64) NULL;",
                    "    END IF;",
                    "    IF NOT EXISTS (SELECT 1 FROM information_schema.columns",
                    "        WHERE table_name='symbols' AND column_name='currency') THEN",
                    "        ALTER TABLE symbols ADD COLUMN currency VARCHAR(32) NULL;",
                    "    END IF;",
                    "    IF NOT EXISTS (SELECT 1 FROM information_schema.columns",
                    "        WHERE table_name='symbols' AND column_name='type') THEN",
                    "        ALTER TABLE symbols ADD COLUMN type VARCHAR(64) NULL;",
                    "    END IF;",
                    "    IF NOT EXISTS (SELECT 1 FROM information_schema.columns",
                    "        WHERE table_name='symbols' AND column_name='isin') THEN",
                    "        ALTER TABLE symbols ADD COLUMN isin VARCHAR(64) NULL;",
                    "    END IF;",
                    "    -- add alternative_names JSONB to symbols if missing",
                    "    IF NOT EXISTS (SELECT 1 FROM information_schema.columns",
                    "        WHERE table_name='symbols' AND column_name='alternative_names') THEN",
                    "        ALTER TABLE symbols ADD COLUMN alternative_names JSONB NULL;",
                    "    END IF;",
                    "    -- add dropped_points diagnostics columns to symbols if missing",
                    "    IF NOT EXISTS (SELECT 1 FROM information_schema.columns",
                    "        WHERE table_name='symbols' AND column_name='dropped_points_recent') THEN",
                    "        ALTER TABLE symbols ADD COLUMN dropped_points_recent INTEGER NULL;",
                    "    END IF;",
                    "    IF NOT EXISTS (SELECT 1 FROM information_schema.columns",
                    "        WHERE table_name='symbols' AND column_name='has_dropped_points_recent') THEN",
                    "        ALTER TABLE symbols ADD COLUMN has_dropped_points_recent INTEGER NOT NULL DEFAULT 0;",
                    "    END IF;",
                    "    -- ensure insufficient_history exists and is nullable (tri-state: NULL|0|1)",
                    "    IF NOT EXISTS (SELECT 1 FROM information_schema.columns",
                    "        WHERE table_name='symbols' AND column_name='insufficient_history') THEN",
                    "        ALTER TABLE symbols ADD COLUMN insufficient_history INTEGER NULL DEFAULT NULL;",
                    "    END IF;",
                    "    -- relax NOT NULL / DEFAULT if previously created as NOT NULL DEFAULT 1",
                    "    IF EXISTS (SELECT 1 FROM information_schema.columns",
                    "        WHERE table_name='symbols' AND column_name='insufficient_history' AND is_nullable='NO') THEN",
                    "        ALTER TABLE symbols ALTER COLUMN insufficient_history DROP NOT NULL;",
                    "    END IF;",
                    "    -- drop default if any to let NULL be the default",
                    "    BEGIN",
                    "        ALTER TABLE symbols ALTER COLUMN insufficient_history DROP DEFAULT;",
                    "    EXCEPTION WHEN OTHERS THEN",
                    "        -- ignore if no default",
                    "        NULL;",
                    "    END;",
                    "    -- migrate unique constraint from (symbol) to (symbol,country)",
                    "    -- drop old unique if present",
                    "    IF EXISTS (SELECT 1 FROM information_schema.table_constraints",
                    "              WHERE table_name='symbols' AND constraint_name='uq_symbols_symbol') THEN",
                    "        BEGIN",
                    "            ALTER TABLE symbols DROP CONSTRAINT uq_symbols_symbol;",
                    "        EXCEPTION WHEN OTHERS THEN",
                    "            NULL;",
                    "        END;",
                    "    END IF;",
                    "    -- add new unique if missing",
                    "    IF NOT EXISTS (SELECT 1 FROM information_schema.table_constraints",
                    "        WHERE table_name='symbols' AND constraint_name='uq_symbols_symbol_country') THEN",
                    "        BEGIN",
                    "            ALTER TABLE symbols ADD CONSTRAINT uq_symbols_symbol_country UNIQUE(symbol, country);",
                    "        EXCEPTION WHEN OTHERS THEN",
                    "            NULL;",
                    "        END;",
                    "    END IF;",
                    "    IF NOT EXISTS (SELECT 1 FROM",
                    "    information_schema.columns",
                    "                      WHERE table_name='cvar_snapshot'",
                    "                        AND column_name=",
                    "                        'updated_at')",
                    "                        THEN",
                    "        ALTER TABLE cvar_snapshot ADD COLUMN",
                    "        updated_at TIMESTAMP NOT NULL",
                    "        DEFAULT now();",
                    "    END IF;",
                    "    -- add alpha (confidence) column if missing",
                    "    IF NOT EXISTS (SELECT 1 FROM",
                    "        information_schema.columns",
                    "        WHERE table_name='cvar_snapshot'",
                    "          AND column_name='alpha') THEN",
                    "        ALTER TABLE cvar_snapshot ADD COLUMN",
                    "        alpha DOUBLE PRECISION NULL;",
                    "    END IF;",
                    "    -- add return fields to cvar_snapshot if missing",
                    "    IF NOT EXISTS (SELECT 1 FROM information_schema.columns",
                    "        WHERE table_name='cvar_snapshot' AND column_name='return_as_of') THEN",
                    "        ALTER TABLE cvar_snapshot ADD COLUMN return_as_of DOUBLE PRECISION NULL;",
                    "    END IF;",
                    "    IF NOT EXISTS (SELECT 1 FROM information_schema.columns",
                    "        WHERE table_name='cvar_snapshot' AND column_name='return_annual') THEN",
                    "        ALTER TABLE cvar_snapshot ADD COLUMN return_annual DOUBLE PRECISION NULL;",
                    "    END IF;",
                    "    -- add instrument_id to cvar_snapshot if missing and migrate unique constraint",
                    "    IF NOT EXISTS (SELECT 1 FROM information_schema.columns",
                    "        WHERE table_name='cvar_snapshot' AND column_name='instrument_id') THEN",
                    "        ALTER TABLE cvar_snapshot ADD COLUMN instrument_id INTEGER NULL;",
                    "        BEGIN",
                    "            ALTER TABLE cvar_snapshot ADD CONSTRAINT fk_cvar_snapshot_instrument",
                    "                FOREIGN KEY (instrument_id) REFERENCES symbols(id);",
                    "        EXCEPTION WHEN OTHERS THEN",
                    "            NULL;",
                    "        END;",
                    "        BEGIN",
                    "            CREATE INDEX IF NOT EXISTS ix_cvar_snapshot_instrument_id ON cvar_snapshot(instrument_id);",
                    "        EXCEPTION WHEN OTHERS THEN",
                    "            NULL;",
                    "        END;",
                    "        -- backfill instrument_id by symbol match, preferring US when available",
                    "        BEGIN",
                    "            UPDATE cvar_snapshot cs SET instrument_id = (",
                    "                SELECT ps.id FROM symbols ps ",
                    "                WHERE ps.symbol = cs.symbol ",
                    "                ORDER BY CASE WHEN upper(COALESCE(ps.country,'')) IN ('US','USA','UNITED STATES') THEN 0 ELSE 1 END, ps.id ",
                    "                LIMIT 1",
                    "            ) WHERE cs.instrument_id IS NULL;",
                    "        EXCEPTION WHEN OTHERS THEN",
                    "            NULL;",
                    "        END;",
                    "    END IF;",
                    "    -- drop old unique constraint if present",
                    "    IF EXISTS (SELECT 1 FROM information_schema.table_constraints",
                    "              WHERE table_name='cvar_snapshot' AND constraint_name='uq_cvar_snapshot_key') THEN",
                    "        BEGIN",
                    "            ALTER TABLE cvar_snapshot DROP CONSTRAINT uq_cvar_snapshot_key;",
                    "        EXCEPTION WHEN OTHERS THEN",
                    "            NULL;",
                    "        END;",
                    "    END IF;",
                    "    -- create new unique constraint including instrument_id",
                    "    IF NOT EXISTS (SELECT 1 FROM information_schema.table_constraints",
                    "        WHERE table_name='cvar_snapshot' AND constraint_name='uq_cvar_snapshot_by_instrument') THEN",
                    "        BEGIN",
                    "            ALTER TABLE cvar_snapshot ADD CONSTRAINT uq_cvar_snapshot_by_instrument",
                    "                UNIQUE(symbol, as_of_date, alpha_label, instrument_id);",
                    "        EXCEPTION WHEN OTHERS THEN",
                    "            NULL;",
                    "        END;",
                    "    END IF;",
                    "    -- ensure exchange table exists",
                    "    CREATE TABLE IF NOT EXISTS exchange (",
                    "        id SERIAL PRIMARY KEY,",
                    "        code VARCHAR(32) NOT NULL UNIQUE,",
                    "        name VARCHAR(256) NULL,",
                    "        operating_mic VARCHAR(128) NULL,",
                    "        country VARCHAR(128) NULL,",
                    "        currency VARCHAR(16) NULL,",
                    "        country_iso2 VARCHAR(8) NULL,",
                    "        country_iso3 VARCHAR(8) NULL,",
                    "        created_at TIMESTAMP NOT NULL DEFAULT now(),",
                    "        updated_at TIMESTAMP NOT NULL DEFAULT now()",
                    "    );",
                    "    CREATE UNIQUE INDEX IF NOT EXISTS uq_exchange_code ON exchange(code);",
                    "    CREATE INDEX IF NOT EXISTS ix_exchange_country ON exchange(country);",
                    "    -- placeholder: insufficient_data_event is created via ORM",
                    "    -- keep block for future DDL adjustments if needed",
                    "END$$;",
                ]
                ddl = "\n".join(ddl_lines)
                conn.execute(text(ddl))
            except Exception:
                pass
            # ---------- Hot-path indexes (Postgres, idempotent) ----------
            try:
                if engine.dialect.name in ("postgresql", "postgres"):
                    idx_sql = [
                        "CREATE INDEX IF NOT EXISTS ix_cvar_snapshot_symbol_asof ON cvar_snapshot(symbol, as_of_date);",
                        "CREATE INDEX IF NOT EXISTS ix_cvar_snapshot_alpha ON cvar_snapshot(alpha_label);",
                        "CREATE INDEX IF NOT EXISTS ix_catalogue_snapshot_asof ON catalogue_snapshot(as_of_utc);",
                        "CREATE INDEX IF NOT EXISTS ix_catalogue_entry_snapshot ON catalogue_snapshot_entry(snapshot_id);",
                        "CREATE INDEX IF NOT EXISTS ix_sur_model_snapshot ON sur_model(snapshot_id);",
                        "CREATE INDEX IF NOT EXISTS ix_portfolio_metrics_portfolio ON portfolio_metrics(portfolio_id);",
                        "CREATE INDEX IF NOT EXISTS ix_ticker_lookup_ts ON ticker_lookup_log(timestamp_utc);",
                        "CREATE INDEX IF NOT EXISTS ix_proximity_search_ts ON proximity_search_log(timestamp_utc);",
                        "CREATE INDEX IF NOT EXISTS ix_digest_delivery_user_status ON digest_delivery(user_id, status);",
                        "CREATE INDEX IF NOT EXISTS ix_digest_delivery_unsub ON digest_delivery(unsubscribe_token);",
                        "CREATE INDEX IF NOT EXISTS ix_auth_attempt_email_time ON auth_attempt(email, timestamp_utc);",
                    ]
                    for sql in idx_sql:
                        try:
                            conn.execute(text(sql))
                        except Exception:
                            pass
                    # Ensure price_time_series exists with required indexes/constraints (idempotent)
                    try:
                        conn.execute(text(
                            """
                            DO $$
                            BEGIN
                                IF NOT EXISTS (
                                    SELECT 1 FROM information_schema.tables
                                    WHERE table_name='price_time_series'
                                ) THEN
                                    CREATE TABLE price_time_series (
                                        id SERIAL PRIMARY KEY,
                                        symbol_id INTEGER NOT NULL REFERENCES symbols (id) ON DELETE CASCADE,
                                        date DATE NOT NULL,
                                        price NUMERIC(20, 6) NOT NULL,
                                        volume BIGINT NULL,
                                        source_type VARCHAR(20) NOT NULL,
                                        version_id VARCHAR(128) NOT NULL,
                                        created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT NOW()
                                    );
                                END IF;

                                -- Indexes
                                BEGIN
                                    CREATE INDEX IF NOT EXISTS ix_pts_symbol_id ON price_time_series(symbol_id);
                                EXCEPTION WHEN OTHERS THEN NULL; END;
                                BEGIN
                                    CREATE INDEX IF NOT EXISTS ix_pts_date ON price_time_series(date);
                                EXCEPTION WHEN OTHERS THEN NULL; END;
                                BEGIN
                                    CREATE INDEX IF NOT EXISTS ix_pts_symbol_date ON price_time_series(symbol_id, date);
                                EXCEPTION WHEN OTHERS THEN NULL; END;

                                -- Unique index for idempotent writes
                                BEGIN
                                    CREATE UNIQUE INDEX IF NOT EXISTS uq_pts_symbol_date_version ON price_time_series(symbol_id, date, version_id);
                                EXCEPTION WHEN OTHERS THEN NULL; END;
                            END$$;
                            """
                        ))
                    except Exception:
                        pass
            except Exception:
                pass
            # Ensure new auth-related columns exist (Postgres path)
            try:
                if engine.dialect.name in ("postgresql", "postgres"):
                    conn.execute(text(
                        """
                        DO $$
                        BEGIN
                            IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                                           WHERE table_name='user' AND column_name='password_hash') THEN
                                ALTER TABLE "user" ADD COLUMN password_hash VARCHAR(128) NULL;
                            END IF;
                            IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                                           WHERE table_name='user' AND column_name='email_verified') THEN
                                ALTER TABLE "user" ADD COLUMN email_verified INTEGER NOT NULL DEFAULT 0;
                            END IF;
                            IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                                           WHERE table_name='user' AND column_name='verification_token') THEN
                                ALTER TABLE "user" ADD COLUMN verification_token VARCHAR(64) NULL;
                            END IF;
                            IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                                           WHERE table_name='user' AND column_name='last_login_utc') THEN
                                ALTER TABLE "user" ADD COLUMN last_login_utc TIMESTAMP NULL;
                            END IF;
                            -- add index on verification_token if missing
                            BEGIN
                                CREATE INDEX IF NOT EXISTS ix_user_verification_token ON "user"(verification_token);
                            EXCEPTION WHEN OTHERS THEN
                                NULL;
                            END;
                            -- password reset columns
                            IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                                           WHERE table_name='user' AND column_name='password_reset_token') THEN
                                ALTER TABLE "user" ADD COLUMN password_reset_token VARCHAR(64) NULL;
                            END IF;
                            IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                                           WHERE table_name='user' AND column_name='password_reset_expires_utc') THEN
                                ALTER TABLE "user" ADD COLUMN password_reset_expires_utc TIMESTAMP NULL;
                            END IF;
                            -- auth_attempt table
                            CREATE TABLE IF NOT EXISTS auth_attempt (
                                id SERIAL PRIMARY KEY,
                                timestamp_utc TIMESTAMP NOT NULL DEFAULT now(),
                                email VARCHAR(320) NULL,
                                ip VARCHAR(64) NULL,
                                purpose VARCHAR(32) NOT NULL DEFAULT 'signin',
                                success INTEGER NOT NULL DEFAULT 0
                            );
                            CREATE INDEX IF NOT EXISTS ix_auth_attempt_email ON auth_attempt(email);
                            CREATE INDEX IF NOT EXISTS ix_auth_attempt_ip ON auth_attempt(ip);
                        END$$;
                        """
                    ))
            except Exception:
                pass
            # One-time normalization: set insufficient_history to NULL for symbols
            # that have no snapshots and no recorded insufficient-data events.
            try:
                if engine.dialect.name in ("postgresql", "postgres"):
                    conn.execute(text(
                        """
                        UPDATE symbols ps
                        SET insufficient_history = NULL
                        WHERE ps.insufficient_history = 1
                          AND NOT EXISTS (
                            SELECT 1 FROM cvar_snapshot cs WHERE cs.symbol = ps.symbol
                          )
                          AND NOT EXISTS (
                            SELECT 1 FROM insufficient_data_event ide WHERE ide.symbol = ps.symbol
                          );
                        """
                    ))
                    # Backfill: mark insufficient_history=1 for symbols with recorded
                    # insufficient_data events (or insufficient_history code) and no snapshots
                    conn.execute(text(
                        """
                        UPDATE symbols ps
                        SET insufficient_history = 1
                        WHERE (ps.insufficient_history IS NULL OR ps.insufficient_history <> 0)
                          AND EXISTS (
                            SELECT 1 FROM insufficient_data_event ide
                            WHERE ide.symbol = ps.symbol
                          )
                          AND NOT EXISTS (
                            SELECT 1 FROM cvar_snapshot cs WHERE cs.symbol = ps.symbol
                          );
                        """
                    ))
            except Exception:
                # best effort only
                pass
                
            # Add 'valid' column to symbols for general validity flag
            try:
                ddl_add_valid = """
                DO $$
                BEGIN
                    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                                   WHERE table_name='symbols' AND column_name='valid') THEN
                        ALTER TABLE symbols ADD COLUMN valid INTEGER NULL;
                    END IF;
                END$$;
                """
                conn.execute(text(ddl_add_valid))
            except Exception:
                # Adding column is best-effort
                pass
                
            # Create validation_flags table for detailed knockout tracking
            try:
                ddl_validation_table = """
                CREATE TABLE IF NOT EXISTS validation_flags (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(128) NOT NULL,
                    country VARCHAR(64) NULL,
                    as_of_date DATE NOT NULL,
                    
                    -- Overall validity (syncs with symbols.insufficient_history)
                    valid INTEGER NOT NULL DEFAULT 1,
                    
                    -- History criteria
                    insufficient_total_history INTEGER NOT NULL DEFAULT 0,
                    insufficient_data_after_cleanup INTEGER NOT NULL DEFAULT 0,
                    
                    -- Structural criteria  
                    backward_dates INTEGER NOT NULL DEFAULT 0,
                    zero_or_negative_prices INTEGER NOT NULL DEFAULT 0,
                    extreme_price_jumps INTEGER NOT NULL DEFAULT 0,
                    
                    -- Liquidity criteria
                    critical_years INTEGER NOT NULL DEFAULT 0,
                    multiple_violations_last252 INTEGER NOT NULL DEFAULT 0,
                    multiple_weak_years INTEGER NOT NULL DEFAULT 0,
                    low_liquidity_warning INTEGER NOT NULL DEFAULT 0,
                    
                    -- Anomaly criteria
                    robust_outliers INTEGER NOT NULL DEFAULT 0,
                    price_discontinuities INTEGER NOT NULL DEFAULT 0,
                    long_plateaus INTEGER NOT NULL DEFAULT 0,
                    illiquid_spikes INTEGER NOT NULL DEFAULT 0,
                    
                    -- Analytics data
                    liquidity_metrics JSONB NULL,
                    anomaly_details JSONB NULL,
                    validation_summary JSONB NULL,
                    
                    created_at TIMESTAMP NOT NULL DEFAULT now(),
                    updated_at TIMESTAMP NOT NULL DEFAULT now()
                );
                """
                conn.execute(text(ddl_validation_table))
                
                # Create unique constraint and indexes
                ddl_constraints = """
                DO $$
                BEGIN
                    -- Add unique constraint if not exists
                    IF NOT EXISTS (SELECT 1 FROM information_schema.table_constraints 
                                   WHERE table_name='validation_flags' 
                                     AND constraint_name='uq_validation_flags_symbol_country_date') THEN
                        ALTER TABLE validation_flags 
                        ADD CONSTRAINT uq_validation_flags_symbol_country_date 
                        UNIQUE (symbol, country, as_of_date);
                    END IF;
                END$$;
                """
                conn.execute(text(ddl_constraints))
                
                # Create indexes for better performance
                ddl_indexes = """
                CREATE INDEX IF NOT EXISTS ix_validation_flags_symbol ON validation_flags(symbol);
                CREATE INDEX IF NOT EXISTS ix_validation_flags_country ON validation_flags(country);
                CREATE INDEX IF NOT EXISTS ix_validation_flags_as_of_date ON validation_flags(as_of_date);
                CREATE INDEX IF NOT EXISTS ix_validation_flags_valid ON validation_flags(valid);
                """
                conn.execute(text(ddl_indexes))
                
            except Exception as e:
                # Validation table creation is best-effort
                pass
                
            # SQLite compatibility: add alpha column if missing
            try:
                if engine.dialect.name == "sqlite":
                    # Check existing columns
                    rows_snap = conn.exec_driver_sql(
                        "PRAGMA table_info('cvar_snapshot')"
                    ).fetchall()
                    cols = {str(r[1]).lower() for r in rows_snap}
                    if "alpha" not in cols:
                        conn.exec_driver_sql(
                            "ALTER TABLE cvar_snapshot ADD COLUMN alpha REAL"
                        )
                    if "return_as_of" not in cols:
                        conn.exec_driver_sql(
                            "ALTER TABLE cvar_snapshot ADD COLUMN return_as_of REAL"
                        )
                    if "return_annual" not in cols:
                        conn.exec_driver_sql(
                            "ALTER TABLE cvar_snapshot ADD COLUMN return_annual REAL"
                        )
                    # Add instrument_id nullable; SQLite will not enforce FK retroactively
                    if "instrument_id" not in cols:
                        conn.exec_driver_sql(
                            "ALTER TABLE cvar_snapshot ADD COLUMN instrument_id INTEGER"
                        )
                    # Add new symbols columns if missing
                    rows_ps = conn.exec_driver_sql(
                        "PRAGMA table_info('symbols')"
                    ).fetchall()
                    pcols = {str(r[1]).lower() for r in rows_ps}
                    if "alternative_names" not in pcols:
                        # SQLite stores JSON as TEXT; clients should write JSON strings
                        conn.exec_driver_sql(
                            "ALTER TABLE symbols ADD COLUMN alternative_names TEXT"
                        )
                    if "country" not in pcols:
                        conn.exec_driver_sql(
                            "ALTER TABLE symbols ADD COLUMN country TEXT"
                        )
                    if "exchange" not in pcols:
                        conn.exec_driver_sql(
                            "ALTER TABLE symbols ADD COLUMN exchange TEXT"
                        )
                    if "currency" not in pcols:
                        conn.exec_driver_sql(
                            "ALTER TABLE symbols ADD COLUMN currency TEXT"
                        )
                    if "type" not in pcols:
                        conn.exec_driver_sql(
                            "ALTER TABLE symbols ADD COLUMN type TEXT"
                        )
                    if "isin" not in pcols:
                        conn.exec_driver_sql(
                            "ALTER TABLE symbols ADD COLUMN isin TEXT"
                        )
                    if "insufficient_history" not in pcols:
                        conn.exec_driver_sql(
                            "ALTER TABLE symbols ADD COLUMN insufficient_history INTEGER"
                        )
                    if "five_stars" not in pcols:
                        conn.exec_driver_sql(
                            "ALTER TABLE symbols ADD COLUMN five_stars INTEGER NOT NULL DEFAULT 0"
                        )
                    if "dropped_points_recent" not in pcols:
                        conn.exec_driver_sql(
                            "ALTER TABLE symbols ADD COLUMN dropped_points_recent INTEGER"
                        )
                    if "has_dropped_points_recent" not in pcols:
                        conn.exec_driver_sql(
                            "ALTER TABLE symbols ADD COLUMN has_dropped_points_recent INTEGER NOT NULL DEFAULT 0"
                        )
                    # Ensure user auth columns exist
                    rows_user = conn.exec_driver_sql(
                        "PRAGMA table_info('user')"
                    ).fetchall()
                    # Create hot-path indexes (SQLite)
                    try:
                        conn.exec_driver_sql("CREATE INDEX IF NOT EXISTS ix_cvar_snapshot_symbol_asof ON cvar_snapshot(symbol, as_of_date)")
                    except Exception:
                        pass
                    try:
                        conn.exec_driver_sql("CREATE INDEX IF NOT EXISTS ix_cvar_snapshot_alpha ON cvar_snapshot(alpha_label)")
                    except Exception:
                        pass
                    try:
                        conn.exec_driver_sql("CREATE INDEX IF NOT EXISTS ix_catalogue_snapshot_asof ON catalogue_snapshot(as_of_utc)")
                    except Exception:
                        pass
                    try:
                        conn.exec_driver_sql("CREATE INDEX IF NOT EXISTS ix_catalogue_entry_snapshot ON catalogue_snapshot_entry(snapshot_id)")
                    except Exception:
                        pass
                    try:
                        conn.exec_driver_sql("CREATE INDEX IF NOT EXISTS ix_sur_model_snapshot ON sur_model(snapshot_id)")
                    except Exception:
                        pass
                    try:
                        conn.exec_driver_sql("CREATE INDEX IF NOT EXISTS ix_digest_delivery_user_status ON digest_delivery(user_id, status)")
                    except Exception:
                        pass
                    try:
                        conn.exec_driver_sql("CREATE INDEX IF NOT EXISTS ix_digest_delivery_unsub ON digest_delivery(unsubscribe_token)")
                    except Exception:
                        pass
                    try:
                        conn.exec_driver_sql("CREATE INDEX IF NOT EXISTS ix_auth_attempt_email_time ON auth_attempt(email, timestamp_utc)")
                    except Exception:
                        pass
                    ucols = {str(r[1]).lower() for r in rows_user}
                    if "password_hash" not in ucols:
                        conn.exec_driver_sql(
                            "ALTER TABLE 'user' ADD COLUMN password_hash TEXT"
                        )
                    if "email_verified" not in ucols:
                        conn.exec_driver_sql(
                            "ALTER TABLE 'user' ADD COLUMN email_verified INTEGER NOT NULL DEFAULT 0"
                        )
                    if "verification_token" not in ucols:
                        conn.exec_driver_sql(
                            "ALTER TABLE 'user' ADD COLUMN verification_token TEXT"
                        )
                    if "last_login_utc" not in ucols:
                        conn.exec_driver_sql(
                            "ALTER TABLE 'user' ADD COLUMN last_login_utc TEXT"
                        )
                    if "password_reset_token" not in ucols:
                        conn.exec_driver_sql(
                            "ALTER TABLE 'user' ADD COLUMN password_reset_token TEXT"
                        )
                    if "password_reset_expires_utc" not in ucols:
                        conn.exec_driver_sql(
                            "ALTER TABLE 'user' ADD COLUMN password_reset_expires_utc TEXT"
                        )
                    # Ensure auth_attempt table exists
                    conn.exec_driver_sql(
                        "CREATE TABLE IF NOT EXISTS auth_attempt (\n"
                        "  id INTEGER PRIMARY KEY AUTOINCREMENT,\n"
                        "  timestamp_utc TEXT NOT NULL,\n"
                        "  email TEXT NULL,\n"
                        "  ip TEXT NULL,\n"
                        "  purpose TEXT NOT NULL DEFAULT 'signin',\n"
                        "  success INTEGER NOT NULL DEFAULT 0\n"
                        ")"
                    )
                    # Ensure exchange table exists (SQLite)
                    conn.exec_driver_sql(
                        "CREATE TABLE IF NOT EXISTS exchange (\n"
                        "  id INTEGER PRIMARY KEY AUTOINCREMENT,\n"
                        "  code TEXT NOT NULL UNIQUE,\n"
                        "  name TEXT NULL,\n"
                        "  operating_mic TEXT NULL,\n"
                        "  country TEXT NULL,\n"
                        "  currency TEXT NULL,\n"
                        "  country_iso2 TEXT NULL,\n"
                        "  country_iso3 TEXT NULL,\n"
                        "  created_at TEXT NOT NULL,\n"
                        "  updated_at TEXT NOT NULL\n"
                        ")"
                    )
                    # Create indexes for exchange table
                    try:
                        conn.exec_driver_sql("CREATE UNIQUE INDEX IF NOT EXISTS uq_exchange_code ON exchange(code)")
                    except Exception:
                        pass
                    try:
                        conn.exec_driver_sql("CREATE INDEX IF NOT EXISTS ix_exchange_country ON exchange(country)")
                    except Exception:
                        pass
            except Exception:
                # best-effort only
                pass
        return True
    except Exception:
        return False


## duplicate removed
def upsert_price_last(
    *,
    symbol: str,
    as_of_date: date,
    price_close: float,
    currency: str | None = None,
    source: str | None = None,
    session=None,
) -> bool:
    """Insert/update last price for (symbol, as_of_date).

    If Postgres is detected, uses ON CONFLICT to avoid extra SELECT.
    """
    own_session = False
    if session is None:
        session = get_db_session()
        own_session = True
    if session is None:
        return False
    try:
        sym = (symbol or "").strip()
        if not sym:
            return False
        try:
            eng = session.get_bind()  # type: ignore[attr-defined]
            if eng is not None and getattr(eng.dialect, "name", "") in ("postgresql", "postgres"):
                sql = (
                    """
                    INSERT INTO price_last (symbol, as_of_date, price_close, currency, source, created_at)
                    VALUES (:symbol, :as_of_date, :price_close, :currency, :source, now())
                    ON CONFLICT (symbol, as_of_date) DO UPDATE
                    SET price_close = EXCLUDED.price_close,
                        currency = EXCLUDED.currency,
                        source = EXCLUDED.source
                    """
                )
                session.execute(
                    text(sql),
                    {
                        "symbol": sym,
                        "as_of_date": as_of_date,
                        "price_close": float(price_close),
                        "currency": currency,
                        "source": source,
                    },
                )
            else:
                rec = (
                    session.query(PriceLast)
                    .filter(
                        PriceLast.symbol == sym,
                        PriceLast.as_of_date == as_of_date,
                    )
                    .one_or_none()
                )
                if rec is None:
                    rec = PriceLast(
                        symbol=sym,
                        as_of_date=as_of_date,
                        price_close=float(price_close),
                        currency=currency,
                        source=source,
                    )
                else:
                    rec.price_close = float(price_close)
                    rec.currency = currency
                    rec.source = source
                session.merge(rec)
        except Exception:
            try:
                rec = (
                    session.query(PriceLast)
                    .filter(
                        PriceLast.symbol == sym,
                        PriceLast.as_of_date == as_of_date,
                    )
                    .one_or_none()
                )
                if rec is None:
                    rec = PriceLast(
                        symbol=sym,
                        as_of_date=as_of_date,
                        price_close=float(price_close),
                        currency=currency,
                        source=source,
                    )
                else:
                    rec.price_close = float(price_close)
                    rec.currency = currency
                    rec.source = source
                session.merge(rec)
            except Exception:
                pass
        if own_session:
            session.commit()
        return True
    except SQLAlchemyError:
        if own_session:
            try:
                session.rollback()
            except Exception:
                pass
        return False
    except Exception:
        return False
    finally:
        if own_session:
            try:
                session.close()
            except Exception:
                pass


def upsert_symbols_item(
    *,
    code: str,
    name: str | None = None,
    country: str | None = None,
    exchange: str | None = None,
    currency: str | None = None,
    instrument_type: str | None = None,
    isin: str | None = None,
) -> None:
    """Insert or update a single Symbols row by (symbol, country).

    - code maps to Symbols.symbol
    - name maps to Symbols.name
    - uniqueness is enforced on (symbol, country)
    """
    session = get_db_session()
    if session is None:
        return
    try:
        sym = (code or "").strip()
        if not sym:
            return
        # Normalize simple country labels
        def _canon_country(v: object | None) -> str | None:
            try:
                s = str(v).strip().lower() if v is not None else ""
            except Exception:
                s = ""
            if not s:
                return None
            # map common placeholders to NULL
            if s in ("none", "null", "n/a", "na", "-", "*"):
                return None
            if s in ("us", "usa", "united states", "united states of america"):
                return "US"
            if s in ("ca", "can", "canada"):
                return "Canada"
            return str(v).strip() if v is not None else None

        def _infer_country(
            country_in: object | None,
            exchange_in: object | None,
            symbol_in: str,
        ) -> str:
            # 1) explicit country wins if valid
            c = _canon_country(country_in)
            if c:
                return c
            # 2) exchange mapping
            try:
                ex = str(exchange_in or "").strip().upper()
            except Exception:
                ex = ""
            if ex in ("TSX", "TSXV", "CSE", "NEO", "TO", "V"):
                return "Canada"
            if ex in ("NYSE", "NASDAQ", "ARCA", "BATS", "OTC", "NYSEMKT"):
                return "US"
            # 3) symbol suffix hints
            try:
                s = (symbol_in or "").upper().strip()
            except Exception:
                s = symbol_in or ""
            for suf in (".TO", ".V", ".CN", ".NE", ":TO", ":V", ":CN", ":NE"):
                if s.endswith(suf):
                    return "Canada"
            for suf in (".US", ":US"):
                if s.endswith(suf):
                    return "US"
            # 4) default to US (never leave empty)
            return "US"

        inc_country = _infer_country(country, exchange, code)
        rec = (
            session.query(Symbols)
            .filter(Symbols.symbol == sym, Symbols.country == inc_country)
            .one_or_none()
        )
        from datetime import datetime as _dt
        now = _dt.utcnow()
        if rec is None:
            rec = Symbols(
                symbol=sym,
                name=(name or sym),
                country=inc_country,
                exchange=exchange,
                currency=currency,
                instrument_type=_canon_type(instrument_type),
                isin=isin,
                created_at=now,
                updated_at=now,
            )
        else:
            # Update fields if changed; preserve existing data when new data is missing
            if name:
                rec.name = name
            rec.country = inc_country
            # Only update if new value is not None/empty - preserve existing data
            def _has_value(val):
                """Check if value is not None and not empty string"""
                return val is not None and str(val).strip() != ""
            
            if _has_value(exchange):
                rec.exchange = exchange
            if _has_value(currency):
                rec.currency = currency
            if _has_value(instrument_type):
                rec.instrument_type = _canon_type(instrument_type)
            if _has_value(isin):
                rec.isin = isin
            rec.updated_at = now
        session.merge(rec)
        session.commit()
    except SQLAlchemyError:
        session.rollback()
    except Exception:
        pass
    finally:
        try:
            session.close()
        except Exception:
            pass


def insert_insufficient_data_event(entry: dict) -> None:
    """Persist a single insufficient-data raw event for diagnostics.

    Expected entry structure:
      {"symbol": str, "as_of": str|None, "raw": dict, "diag": dict|None}
    """
    session = get_db_session()
    if session is None:
        return
    try:
        symbol = str(entry.get("symbol") or "").strip()
        as_of_raw = entry.get("as_of")
        as_of_dt = _coerce_date(as_of_raw) if as_of_raw else None
        raw_payload = (
            entry.get("raw") if isinstance(entry.get("raw"), dict) else None
        )
        diag_payload = (
            entry.get("diag") if isinstance(entry.get("diag"), dict) else None
        )
        code = None
        error_msg = None
        source = None
        correlation_id = None
        try:
            if isinstance(raw_payload, dict):
                code = str(raw_payload.get("code") or "insufficient_data")
                error_msg = raw_payload.get("error")
                source = raw_payload.get("source")
                correlation_id = raw_payload.get("correlation_id")
        except Exception:
            pass
        # Normalize a code for storage (map insufficient_history into code if present)
        try:
            code_norm = str(entry.get("code") or "")
            if not code_norm and isinstance(entry.get("raw"), dict):
                code_norm = str(entry["raw"].get("code") or "")
            if not code_norm:
                code_norm = "insufficient_data"
        except Exception:
            code_norm = "insufficient_data"

        rec = InsufficientDataEvent(
            symbol=symbol or None,
            as_of_date=as_of_dt,
            code=code_norm,
            error=(str(error_msg) if error_msg is not None else None),
            source=(str(source) if source is not None else None),
            correlation_id=(
                str(correlation_id) if correlation_id is not None else None
            ),
            raw=raw_payload,
            diag=diag_payload,
        )
        session.add(rec)
        session.commit()
    except SQLAlchemyError:
        try:
            session.rollback()
        except Exception:
            pass
    except Exception:
        pass
    finally:
        try:
            session.close()
        except Exception:
            pass


def upsert_symbols_bulk(items: list[dict]) -> int:
    """Upsert many symbol rows. Returns count processed.

    Expected keys per item: Code, Name, Country, Exchange, Currency, Type, Isin
    """
    session = get_db_session()
    if session is None:
        return 0
    processed = 0
    try:
        import logging as _log
        _logger = _log.getLogger("symbols_import")
        from datetime import datetime as _dt
        now = _dt.utcnow()

        def _canon_country(v: Any) -> str | None:
            try:
                s = str(v).strip().lower()
            except Exception:
                return None
            if not s:
                return None
            if s in ("none", "null", "n/a", "na", "-", "*"):
                return None
            if s in ("us", "usa", "united states", "united states of america"):
                return "US"
            if s in ("ca", "can", "canada"):
                return "Canada"
            return str(v).strip()
        def _infer_country_bulk(country_in: Any, exchange_in: Any, symbol_in: str) -> str:
            c = _canon_country(country_in)
            if c:
                return c
            try:
                ex = str(exchange_in or "").strip().upper()
            except Exception:
                ex = ""
            if ex in ("TSX", "TSXV", "CSE", "NEO", "TO", "V"):
                return "Canada"
            if ex in ("NYSE", "NASDAQ", "ARCA", "BATS", "OTC", "NYSEMKT"):
                return "US"
            try:
                s = (symbol_in or "").upper().strip()
            except Exception:
                s = symbol_in or ""
            for suf in (".TO", ".V", ".CN", ".NE", ":TO", ":V", ":CN", ":NE"):
                if s.endswith(suf):
                    return "Canada"
            for suf in (".US", ":US"):
                if s.endswith(suf):
                    return "US"
            return "US"
        for it in items:
            try:
                code = str(it.get("Code") or "").strip()
                if not code:
                    continue
                name = (str(it.get("Name")) if it.get("Name") is not None else code).strip()
                inc_country = _infer_country_bulk(
                    it.get("Country"), it.get("Exchange"), code
                )
                rec = (
                    session.query(Symbols)
                    .filter(Symbols.symbol == code, Symbols.country == inc_country)
                    .one_or_none()
                )
                mark_star = False
                try:
                    # Accept either boolean or truthy string
                    v = it.get("five_stars") if "five_stars" in it else it.get("FiveStars")
                    if isinstance(v, str):
                        mark_star = v.strip().lower() in ("1", "true", "yes")
                    elif isinstance(v, (int, float)):
                        mark_star = bool(v)
                    elif isinstance(v, bool):
                        mark_star = v
                except Exception:
                    mark_star = False

                if rec is None:
                    rec = Symbols(
                        symbol=code,
                        name=name or code,
                        country=inc_country,
                        exchange=(
                            str(it.get("Exchange")) if it.get("Exchange") is not None else None
                        ),
                        currency=(
                            str(it.get("Currency")) if it.get("Currency") is not None else None
                        ),
                        instrument_type=_canon_type(
                            str(it.get("Type")) if it.get("Type") is not None else None
                        ),
                        isin=(
                            str(it.get("Isin")) if it.get("Isin") is not None else None
                        ),
                        created_at=now,
                        updated_at=now,
                    )
                else:
                    # We are updating the row for the specific (symbol, country)
                    allow = True
                    # Update human name cautiously
                    try:
                        if allow or not rec.name:
                            rec.name = name or rec.name
                        elif name and name != rec.name:
                            alts = rec.alternative_names or []
                            if isinstance(alts, list) and name not in alts:
                                alts.append(name)
                                rec.alternative_names = alts
                    except Exception:
                        pass
                    if allow:
                        rec.country = inc_country
                        
                        def _has_value(val):
                            """Check if value is not None and not empty string"""
                            return val is not None and str(val).strip() != ""
                        
                        # Only update if new value is not None/empty - preserve existing data
                        exchange_val = it.get("Exchange")
                        if _has_value(exchange_val):
                            rec.exchange = str(exchange_val)
                        
                        currency_val = it.get("Currency")
                        if _has_value(currency_val):
                            rec.currency = str(currency_val)
                        
                        type_val = it.get("Type")
                        if _has_value(type_val):
                            rec.instrument_type = _canon_type(str(type_val))
                        
                        isin_val = it.get("Isin")
                        if _has_value(isin_val):
                            rec.isin = str(isin_val)
                    rec.updated_at = now
                # Set five_stars if provided
                try:
                    if mark_star:
                        rec.five_stars = 1
                except Exception:
                    pass
                session.merge(rec)
                processed += 1
                # Commit periodically to reduce transaction size
                if (processed % 500) == 0:
                    session.commit()
            except Exception:
                continue
        session.commit()
        return processed
    except SQLAlchemyError:
        try:
            session.rollback()
        except Exception:
            pass
        return processed
    finally:
        try:
            session.close()
        except Exception:
            pass


def save_cvar_result(symbol: str, payload: Dict[str, Any]) -> None:
    """Persist CVaR results (for a single symbol computation) if DB configured.

    Expects payload from CVaRCalculator.get_cvar_data().
    Writes three rows per compute (alpha_label in {50,95,99}).
    """
    # Normalize symbol for DB storage (strip :US/.US etc.)
    try:
        symbol = _db_base_symbol(symbol)
    except Exception:
        symbol = (symbol or "").strip().upper()
    session = get_db_session()
    if session is None:
        return
    try:
        as_of = _coerce_date(payload.get("as_of_date"))
        start_date = _coerce_date(payload.get("start_date"))
        wrote_any = False
        # Try to extract observed years from payload (data_summary)
        years_val: int = 1
        try:
            y = payload.get("data_summary", {}).get("years")
            if y is not None:
                years_val = int(round(float(y)))
        except Exception:
            years_val = 1

        # Resolve instrument_id for this symbol if possible
        instrument_id_val: int | None = None
        try:
            row_ids = (
                session.query(Symbols.id, Symbols.country)
                .filter(Symbols.symbol == symbol)
                .all()
            )
            if row_ids:
                if len(row_ids) == 1:
                    instrument_id_val = int(row_ids[0][0])
                else:
                    # prefer US when ambiguous
                    us_row = [r for r in row_ids if (str(r[1] or "").upper() in ("US", "USA", "UNITED STATES"))]
                    instrument_id_val = int((us_row[0][0] if us_row else row_ids[0][0]))
        except Exception:
            instrument_id_val = None

        for label in (50, 95, 99):
            block = payload.get(f"cvar{label}") or {}
            annual = block.get("annual") or {}
            # Skip persisting labels that contain only nulls
            try:
                _n = float(annual.get("nig"))  # type: ignore[arg-type]
            except Exception:
                _n = float("nan")
            try:
                _g = float(annual.get("ghst"))  # type: ignore[arg-type]
            except Exception:
                _g = float("nan")
            try:
                _e = float(annual.get("evar"))  # type: ignore[arg-type]
            except Exception:
                _e = float("nan")
            if not any(math.isfinite(x) for x in (_n, _g, _e)):
                continue
            q = (
                session.query(CvarSnapshot)
                .filter(
                    CvarSnapshot.symbol == symbol,
                    CvarSnapshot.as_of_date == as_of,
                    CvarSnapshot.alpha_label == label,
                )
            )
            if instrument_id_val is not None:
                q = q.filter(CvarSnapshot.instrument_id == instrument_id_val)  # type: ignore
            rec = q.one_or_none()
            if rec is None:
                rec = CvarSnapshot(
                    symbol=symbol,
                    as_of_date=as_of,
                    start_date=start_date,
                    years=years_val,
                    alpha_label=label,
                )
                try:
                    rec.instrument_id = instrument_id_val  # type: ignore[attr-defined]
                except Exception:
                    pass
            else:
                # Update years/start_date if present
                try:
                    rec.years = years_val
                except Exception:
                    pass
                if start_date is not None:
                    rec.start_date = start_date
            # Persist both dynamic/fixed confidence alpha and model triples
            try:
                rec.alpha = _safe_float(block.get("alpha"))
            except Exception:
                pass
            rec.cvar_nig = _safe_float(annual.get("nig"))
            rec.cvar_ghst = _safe_float(annual.get("ghst"))
            rec.cvar_evar = _safe_float(annual.get("evar"))
            rec.cached = 1 if payload.get("cached") else 0
            # Store limited extra info (e.g., lambert benchmarks)
            extra = {}
            if isinstance(block.get("lambert"), dict):
                extra["lambert"] = block["lambert"]
            rec.extra = extra or None
            try:
                if instrument_id_val is not None:
                    rec.instrument_id = instrument_id_val  # type: ignore[attr-defined]
            except Exception:
                pass
            session.merge(rec)
            wrote_any = True
        if wrote_any:
            session.commit()
        # Save anomalies report if present (one row per as_of/symbol)
        # Allow disabling via env NIR_SAVE_ANOMALIES=0|off|false|no
        try:
            import os as _os
            _save_anom = (_os.getenv("NIR_SAVE_ANOMALIES", "1") or "").lower() not in (
                "0",
                "off",
                "false",
                "no",
            )
        except Exception:
            _save_anom = True
        if not _save_anom:
            return
        try:
            rep = payload.get("anomalies_report") if isinstance(payload, dict) else None
            if isinstance(rep, dict):
                as_of = _coerce_date(payload.get("as_of_date"))
                if as_of is not None:
                    ar = (
                        session.query(AnomalyReport)
                        .filter(AnomalyReport.symbol == symbol, AnomalyReport.as_of_date == as_of)
                        .one_or_none()
                    )
                    if ar is None:
                        ar = AnomalyReport(symbol=symbol, as_of_date=as_of)
                    # payload keys: report(list), summary(dict), policy/asset_class(optional)
                    try:
                        _rep_obj = rep.get("report")
                        _sum_obj = rep.get("summary")
                        if isinstance(_rep_obj, str):
                            try:
                                _rep_obj = json.loads(_rep_obj)
                            except Exception:
                                pass
                        if isinstance(_sum_obj, str):
                            try:
                                _sum_obj = json.loads(_sum_obj)
                            except Exception:
                                pass
                        ar.report = _rep_obj
                        ar.summary = _sum_obj
                        ar.policy = rep.get("policy")
                        ar.asset_class = rep.get("asset_class")
                    except Exception:
                        pass
                    session.merge(ar)
                    session.commit()
        except Exception:
            # best-effort only
            pass
    except SQLAlchemyError:
        session.rollback()
    except Exception:
        # Avoid surfacing DB errors to the app flow
        pass
    finally:
        try:
            session.close()
        except Exception:
            pass


def upsert_snapshot_row(
    *,
    symbol: str,
    as_of_date: date,
    alpha_label: int,
    alpha_conf: float | None = None,
    years: int | float | None,
    cvar_nig: Any,
    cvar_ghst: Any,
    cvar_evar: Any,
    start_date: Any | None = None,
    source: str | None = None,
    return_as_of: float | None = None,
    return_annual: float | None = None,
    instrument_id: int | None = None,
    session=None,
) -> bool:
    """Insert/update a single CvarSnapshot row (idempotent by unique key)."""
    _u_log = _log.getLogger("cvar_upsert.db")
    own_session = False
    if session is None:
        session = get_db_session()
        own_session = True
    if session is None:
        return False
    try:
        # Normalize to base symbol (AAPL, not AAPL:US) for storage key
        try:
            symbol = _db_base_symbol(symbol)
        except Exception:
            symbol = (symbol or "").strip().upper()
        inst_id: int | None = instrument_id
        if inst_id is None:
            try:
                rows = (
                    session.query(Symbols.id, Symbols.country)
                    .filter(Symbols.symbol == symbol)
                    .all()
                )
                if rows:
                    if len(rows) == 1:
                        inst_id = int(rows[0][0])
                    else:
                        us_row = [r for r in rows if (str(r[1] or "").upper() in ("US", "USA", "UNITED STATES"))]
                        inst_id = int((us_row[0][0] if us_row else rows[0][0]))
            except Exception:
                inst_id = None
        try:
            eng = session.get_bind()  # type: ignore[attr-defined]
            is_pg = eng is not None and getattr(eng.dialect, "name", "") in ("postgresql", "postgres")
        except Exception:
            is_pg = False

        years_val = int(years) if years is not None else 1
        alpha_val = float(alpha_conf) if alpha_conf is not None else None
        ra_val = _safe_float(return_as_of)
        rann_val = _safe_float(return_annual)
        sd_val = _coerce_date(start_date) if start_date is not None else None
        extra_val = ({"source": source} if source else None)

        if is_pg and inst_id is not None:
            sql = (
                """
                INSERT INTO cvar_snapshot (
                    symbol, as_of_date, alpha_label, years, alpha,
                    cvar_nig, cvar_ghst, cvar_evar,
                    return_as_of, return_annual, start_date,
                    extra, instrument_id, cached, updated_at, created_at
                )
                VALUES (
                    :symbol, :as_of_date, :alpha_label, :years, :alpha,
                    :cvar_nig, :cvar_ghst, :cvar_evar,
                    :return_as_of, :return_annual, :start_date,
                    :extra, :instrument_id, :cached, now(), now()
                )
                ON CONFLICT (symbol, as_of_date, alpha_label, instrument_id) DO UPDATE
                SET years = EXCLUDED.years,
                    alpha = EXCLUDED.alpha,
                    cvar_nig = EXCLUDED.cvar_nig,
                    cvar_ghst = EXCLUDED.cvar_ghst,
                    cvar_evar = EXCLUDED.cvar_evar,
                    return_as_of = EXCLUDED.return_as_of,
                    return_annual = EXCLUDED.return_annual,
                    start_date = COALESCE(EXCLUDED.start_date, cvar_snapshot.start_date),
                    extra = COALESCE(EXCLUDED.extra, cvar_snapshot.extra),
                    cached = EXCLUDED.cached,
                    updated_at = now();
                """
            )
            _u_log.info(
                "db upsert try(pg): symbol=%s as_of=%s label=%d nig=%s ghst=%s evar=%s inst_id=%s",
                symbol,
                as_of_date,
                int(alpha_label),
                _safe_float(cvar_nig),
                _safe_float(cvar_ghst),
                _safe_float(cvar_evar),
                int(inst_id),
            )
            stmt = text(sql).bindparams(bindparam("extra", type_=JSON))
            session.execute(
                stmt,
                {
                    "symbol": symbol,
                    "as_of_date": as_of_date,
                    "alpha_label": int(alpha_label),
                    "years": years_val,
                    "alpha": alpha_val,
                    "cvar_nig": _safe_float(cvar_nig),
                    "cvar_ghst": _safe_float(cvar_ghst),
                    "cvar_evar": _safe_float(cvar_evar),
                    "return_as_of": ra_val,
                    "return_annual": rann_val,
                    "start_date": sd_val,
                    "extra": extra_val,
                    "instrument_id": int(inst_id),
                    "cached": 0,
                },
            )
        else:
            _u_log.info(
                "db upsert try(orm): symbol=%s as_of=%s label=%d nig=%s ghst=%s evar=%s inst_id=%s",
                symbol,
                as_of_date,
                int(alpha_label),
                _safe_float(cvar_nig),
                _safe_float(cvar_ghst),
                _safe_float(cvar_evar),
                (int(inst_id) if inst_id is not None else None),
            )
            q = (
                session.query(CvarSnapshot)
                .filter(
                    CvarSnapshot.symbol == symbol,
                    CvarSnapshot.as_of_date == as_of_date,
                    CvarSnapshot.alpha_label == alpha_label,
                )
            )
            if inst_id is not None:
                q = q.filter(CvarSnapshot.instrument_id == inst_id)  # type: ignore
            rec = q.one_or_none()
            if rec is None:
                rec = CvarSnapshot(
                    symbol=symbol,
                    as_of_date=as_of_date,
                    years=years_val,
                    alpha_label=alpha_label,
                )
                try:
                    rec.instrument_id = inst_id  # type: ignore[attr-defined]
                except Exception:
                    pass
            else:
                if years is not None:
                    try:
                        rec.years = years_val
                    except Exception:
                        pass
            if sd_val is not None:
                try:
                    rec.start_date = sd_val
                except Exception:
                    pass
            if alpha_val is not None:
                try:
                    rec.alpha = alpha_val
                except Exception:
                    pass
            rec.cvar_nig = _safe_float(cvar_nig)
            rec.cvar_ghst = _safe_float(cvar_ghst)
            rec.cvar_evar = _safe_float(cvar_evar)
            try:
                rec.return_as_of = ra_val
                rec.return_annual = rann_val
            except Exception:
                pass
            rec.cached = 0
            try:
                from datetime import datetime as _dt
                rec.updated_at = _dt.utcnow()
            except Exception:
                pass
            if source:
                rec.extra = {"source": source}
            try:
                if inst_id is not None:
                    rec.instrument_id = inst_id  # type: ignore[attr-defined]
            except Exception:
                pass
            session.merge(rec)
        if own_session:
            session.commit()
        return True
    except SQLAlchemyError as _e:
        try:
            _u_log.exception(
                "db upsert error(sqlalchemy): symbol=%s as_of=%s label=%d err=%s",
                symbol,
                as_of_date,
                int(alpha_label),
                str(_e),
            )
        except Exception:
            pass
        if own_session:
            try:
                session.rollback()
            except Exception:
                pass
        return False
    except Exception as _e:
        try:
            _u_log.exception(
                "db upsert error(other): symbol=%s as_of=%s label=%d err=%s",
                symbol,
                as_of_date,
                int(alpha_label),
                str(_e),
            )
        except Exception:
            pass
        return False
    finally:
        if own_session:
            try:
                session.close()
            except Exception:
                pass


def _safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        return v
    except Exception:
        return None


def _coerce_date(v: Any) -> Optional[date]:
    if isinstance(v, date):
        return v
    if isinstance(v, str):
        try:
            return datetime.strptime(v, "%Y-%m-%d").date()
        except Exception:
            try:
                # try ISO full datetime
                return datetime.fromisoformat(v).date()
            except Exception:
                return None
    return None


def upsert_annual_violation(
    *,
    symbol: str,
    year: int,
    as_of_date: date,
    next_year: int,
    next_return: Optional[float],
    cvar99_nig: Optional[float],
    cvar99_ghst: Optional[float],
    cvar99_evar: Optional[float],
    cvar99_worst: Optional[float],
    violated99: bool,
) -> None:
    """Insert or update a single (symbol, year) annual violation row.

    Values are positive loss fractions for CVaR fields,
    next_return is signed simple return.
    """
    session = get_db_session()
    if session is None:
        return
    try:
        rec = (
            session.query(AnnualCvarViolation)
            .filter(
                AnnualCvarViolation.symbol == symbol,
                AnnualCvarViolation.year == year,
            )
            .one_or_none()
        )
        if rec is None:
            rec = AnnualCvarViolation(
                symbol=symbol,
                year=year,
                as_of_date=as_of_date,
                next_year=next_year,
            )
        rec.as_of_date = as_of_date
        rec.next_year = next_year
        rec.next_return = _safe_float(next_return)
        rec.cvar99_nig = _safe_float(cvar99_nig)
        rec.cvar99_ghst = _safe_float(cvar99_ghst)
        rec.cvar99_evar = _safe_float(cvar99_evar)
        rec.cvar99_worst = _safe_float(cvar99_worst)
        rec.violated99 = 1 if violated99 else 0
        session.merge(rec)
        session.commit()
    except SQLAlchemyError:
        session.rollback()
    except Exception:
        pass
    finally:
        try:
            session.close()
        except Exception:
            pass


def get_existing_annual_violation_years(
    symbol: str,
    start_year: int,
    end_year: int,
) -> set[int]:
    """Return set of years already present for a symbol within the range.

    If the DB is not configured or in case of any error, returns an empty set.
    """
    session = get_db_session()
    if session is None:
        return set()
    try:
        rows = (
            session.query(AnnualCvarViolation.year)
            .filter(
                AnnualCvarViolation.symbol == symbol,
                AnnualCvarViolation.year >= start_year,
                AnnualCvarViolation.year <= end_year,
            )
            .all()
        )
        return {int(y[0]) for y in rows if y and y[0] is not None}
    except Exception:
        return set()
    finally:
        try:
            session.close()
        except Exception:
            pass


def list_annual_violations(
    symbol: str, start_year: int, end_year: int
) -> list[AnnualCvarViolation]:
    """Return rows from annual_cvar_violation for a symbol and year range.

    Empty list if DB unavailable or on error.
    """
    session = get_db_session()
    if session is None:
        return []
    try:
        rows = (
            session.query(AnnualCvarViolation)
            .filter(
                AnnualCvarViolation.symbol == symbol,
                AnnualCvarViolation.year >= start_year,
                AnnualCvarViolation.year <= end_year,
            )
            .order_by(AnnualCvarViolation.year.asc())
            .all()
        )
        return list(rows)
    except Exception:
        return []
    finally:
        try:
            session.close()
        except Exception:
            pass


def bootstrap_annual_violations_from_csv(csv_path: str) -> int:
    """Load CSV rows into annual_cvar_violation if the table is empty.

    Returns number of rows inserted. Safe to call multiple times: does nothing
    if the table already has data.
    CSV columns expected (header, case-sensitive):
      symbol,year,as_of_date,next_year,next_return,cvar99_nig,cvar99_ghst,cvar99_evar,cvar99_worst,violated99
    """
    session = get_db_session()
    if session is None:
        return 0
    try:
        # Check emptiness quickly
        has_any = session.query(AnnualCvarViolation.id).limit(1).all()
        if has_any:
            return 0
        import csv
        from datetime import datetime as _dt
        inserted = 0
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    symbol = str(row.get("symbol") or "").strip()
                    if not symbol:
                        continue
                    y_raw = row.get("year")
                    year = int(str(y_raw))
                    as_of_raw = row.get("as_of_date") or f"{year}-12-31"
                    try:
                        as_of_date = _dt.strptime(
                            as_of_raw, "%Y-%m-%d"
                        ).date()
                    except Exception:
                        as_of_date = _dt.fromisoformat(
                            str(as_of_raw)
                        ).date()
                    ny_raw = row.get("next_year")
                    next_year = int(str(ny_raw))

                    def _f(name: str):
                        try:
                            v = row.get(name)
                            return (
                                float(v) if v is not None and v != "" else None
                            )
                        except Exception:
                            return None
                    next_return = _f("next_return")
                    c_nig = _f("cvar99_nig")
                    c_ghst = _f("cvar99_ghst")
                    c_evar = _f("cvar99_evar")
                    c_worst = _f("cvar99_worst")
                    violated = str(row.get("violated99") or "0").strip()
                    violated99 = violated in (
                        "1",
                        "true",
                        "True",
                        "YES",
                        "yes",
                    )
                    rec = AnnualCvarViolation(
                        symbol=symbol,
                        year=year,
                        as_of_date=as_of_date,
                        next_year=next_year,
                        next_return=next_return,
                        cvar99_nig=c_nig,
                        cvar99_ghst=c_ghst,
                        cvar99_evar=c_evar,
                        cvar99_worst=c_worst,
                        violated99=1 if violated99 else 0,
                    )
                    session.add(rec)
                    inserted += 1
                except Exception:
                    continue
        session.commit()
        return inserted
    except Exception:
        try:
            session.rollback()
        except Exception:
            pass
        return 0
    finally:
        try:
            session.close()
        except Exception:
            pass


