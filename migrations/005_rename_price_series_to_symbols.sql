-- Migration: Rename price_series table to symbols and add flag columns
-- This migration safely renames the table while preserving all data and relationships

BEGIN;

-- Step 1: Add new flag columns to existing table
DO $$
BEGIN
    -- Check if flag columns don't exist, then add them
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                  WHERE table_name='price_series' AND column_name='five_stars') THEN
        -- five_stars already exists, but ensure it's not null
        UPDATE price_series SET five_stars = 0 WHERE five_stars IS NULL;
        ALTER TABLE price_series ALTER COLUMN five_stars SET NOT NULL;
        ALTER TABLE price_series ALTER COLUMN five_stars SET DEFAULT 0;
    END IF;
    
    -- Add new flag columns based on folder structure
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                  WHERE table_name='price_series' AND column_name='dow_jones_industrial_average') THEN
        ALTER TABLE price_series ADD COLUMN dow_jones_industrial_average INTEGER NOT NULL DEFAULT 0;
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                  WHERE table_name='price_series' AND column_name='nasdaq_100') THEN
        ALTER TABLE price_series ADD COLUMN nasdaq_100 INTEGER NOT NULL DEFAULT 0;
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                  WHERE table_name='price_series' AND column_name='sp_500') THEN
        ALTER TABLE price_series ADD COLUMN sp_500 INTEGER NOT NULL DEFAULT 0;
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                  WHERE table_name='price_series' AND column_name='russell_1000') THEN
        ALTER TABLE price_series ADD COLUMN russell_1000 INTEGER NOT NULL DEFAULT 0;
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                  WHERE table_name='price_series' AND column_name='sp_midcap_400') THEN
        ALTER TABLE price_series ADD COLUMN sp_midcap_400 INTEGER NOT NULL DEFAULT 0;
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                  WHERE table_name='price_series' AND column_name='sp_smallcap_600') THEN
        ALTER TABLE price_series ADD COLUMN sp_smallcap_600 INTEGER NOT NULL DEFAULT 0;
    END IF;
    
    -- Add extensible flags for future use
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                  WHERE table_name='price_series' AND column_name='ftse_100') THEN
        ALTER TABLE price_series ADD COLUMN ftse_100 INTEGER NOT NULL DEFAULT 0;
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                  WHERE table_name='price_series' AND column_name='tsx_60') THEN
        ALTER TABLE price_series ADD COLUMN tsx_60 INTEGER NOT NULL DEFAULT 0;
    END IF;
    
    -- Generic flags for Harvard universe categories
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                  WHERE table_name='price_series' AND column_name='etf_flag') THEN
        ALTER TABLE price_series ADD COLUMN etf_flag INTEGER NOT NULL DEFAULT 0;
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                  WHERE table_name='price_series' AND column_name='mutual_fund_flag') THEN
        ALTER TABLE price_series ADD COLUMN mutual_fund_flag INTEGER NOT NULL DEFAULT 0;
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                  WHERE table_name='price_series' AND column_name='common_stock_flag') THEN
        ALTER TABLE price_series ADD COLUMN common_stock_flag INTEGER NOT NULL DEFAULT 0;
    END IF;
END $$;

-- Step 2: Create indexes on new flag columns for performance
DO $$
BEGIN
    -- Create indexes if they don't exist
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'ix_price_series_five_stars') THEN
        CREATE INDEX ix_price_series_five_stars ON price_series(five_stars) WHERE five_stars = 1;
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'ix_price_series_sp_500') THEN
        CREATE INDEX ix_price_series_sp_500 ON price_series(sp_500) WHERE sp_500 = 1;
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'ix_price_series_nasdaq_100') THEN
        CREATE INDEX ix_price_series_nasdaq_100 ON price_series(nasdaq_100) WHERE nasdaq_100 = 1;
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'ix_price_series_dow_jones') THEN
        CREATE INDEX ix_price_series_dow_jones ON price_series(dow_jones_industrial_average) WHERE dow_jones_industrial_average = 1;
    END IF;
END $$;

-- Step 3: Rename the table to symbols
DO $$
BEGIN
    -- Check if symbols table doesn't already exist
    IF NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name='symbols') THEN
        ALTER TABLE price_series RENAME TO symbols;
        
        -- Update constraint names
        ALTER TABLE symbols RENAME CONSTRAINT uq_price_series_symbol_country TO uq_symbols_symbol_country;
        
        -- Update sequence name if it exists
        IF EXISTS (SELECT 1 FROM information_schema.sequences WHERE sequence_name = 'price_series_id_seq') THEN
            ALTER SEQUENCE price_series_id_seq RENAME TO symbols_id_seq;
        END IF;
        
        -- Update any existing foreign key constraints in other tables
        -- This will be handled by SQLAlchemy model updates
    END IF;
END $$;

-- Step 4: Create a view for backward compatibility (temporary)
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.views WHERE table_name='price_series_compat') THEN
        CREATE VIEW price_series_compat AS SELECT * FROM symbols;
    END IF;
END $$;

-- Step 5: Add comments for documentation
COMMENT ON TABLE symbols IS 'Financial instrument symbols and metadata (formerly price_series)';
COMMENT ON COLUMN symbols.five_stars IS 'Morningstar 5-star rated funds (1=yes, 0=no)';
COMMENT ON COLUMN symbols.sp_500 IS 'S&P 500 index constituent (1=yes, 0=no)';
COMMENT ON COLUMN symbols.nasdaq_100 IS 'NASDAQ 100 index constituent (1=yes, 0=no)';
COMMENT ON COLUMN symbols.dow_jones_industrial_average IS 'Dow Jones Industrial Average constituent (1=yes, 0=no)';
COMMENT ON COLUMN symbols.russell_1000 IS 'Russell 1000 index constituent (1=yes, 0=no)';
COMMENT ON COLUMN symbols.sp_midcap_400 IS 'S&P MidCap 400 index constituent (1=yes, 0=no)';
COMMENT ON COLUMN symbols.sp_smallcap_600 IS 'S&P SmallCap 600 index constituent (1=yes, 0=no)';
COMMENT ON COLUMN symbols.ftse_100 IS 'FTSE 100 index constituent (1=yes, 0=no)';
COMMENT ON COLUMN symbols.tsx_60 IS 'TSX 60 index constituent (1=yes, 0=no)';
COMMENT ON COLUMN symbols.etf_flag IS 'Exchange-traded fund classification (1=yes, 0=no)';
COMMENT ON COLUMN symbols.mutual_fund_flag IS 'Mutual fund classification (1=yes, 0=no)';
COMMENT ON COLUMN symbols.common_stock_flag IS 'Common stock classification (1=yes, 0=no)';

COMMIT;
