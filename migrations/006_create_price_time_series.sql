BEGIN;

CREATE TABLE IF NOT EXISTS price_time_series (
    id SERIAL PRIMARY KEY,
    symbol_id INTEGER NOT NULL REFERENCES symbols (id) ON DELETE CASCADE,
    date DATE NOT NULL,
    price NUMERIC(20, 6) NOT NULL,
    volume BIGINT NULL,
    source_type VARCHAR(20) NOT NULL CHECK (source_type IN ('raw','computed')),
    version_id VARCHAR(128) NOT NULL,
    created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_pts_symbol_id ON price_time_series(symbol_id);
CREATE INDEX IF NOT EXISTS ix_pts_date ON price_time_series(date);
CREATE INDEX IF NOT EXISTS ix_pts_symbol_date ON price_time_series(symbol_id, date);

CREATE UNIQUE INDEX IF NOT EXISTS uq_pts_symbol_date_version ON price_time_series(symbol_id, date, version_id);

COMMIT;


