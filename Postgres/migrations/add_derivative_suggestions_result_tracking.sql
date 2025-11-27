-- Migration: Add result tracking fields to derivative_suggestions table
-- This migration adds fields to track the actual outcomes of derivative suggestions

-- Add result tracking columns if they don't exist
DO $$ 
BEGIN
    -- Add status column
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_schema = 'my_schema' 
                   AND table_name = 'derivative_suggestions' 
                   AND column_name = 'status') THEN
        ALTER TABLE my_schema.derivative_suggestions 
        ADD COLUMN status VARCHAR(20) DEFAULT 'PENDING';
    END IF;
    
    -- Add executed_at column
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_schema = 'my_schema' 
                   AND table_name = 'derivative_suggestions' 
                   AND column_name = 'executed_at') THEN
        ALTER TABLE my_schema.derivative_suggestions 
        ADD COLUMN executed_at TIMESTAMP;
    END IF;
    
    -- Add exit_date column
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_schema = 'my_schema' 
                   AND table_name = 'derivative_suggestions' 
                   AND column_name = 'exit_date') THEN
        ALTER TABLE my_schema.derivative_suggestions 
        ADD COLUMN exit_date DATE;
    END IF;
    
    -- Add exit_price column
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_schema = 'my_schema' 
                   AND table_name = 'derivative_suggestions' 
                   AND column_name = 'exit_price') THEN
        ALTER TABLE my_schema.derivative_suggestions 
        ADD COLUMN exit_price DOUBLE PRECISION;
    END IF;
    
    -- Add actual_profit column
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_schema = 'my_schema' 
                   AND table_name = 'derivative_suggestions' 
                   AND column_name = 'actual_profit') THEN
        ALTER TABLE my_schema.derivative_suggestions 
        ADD COLUMN actual_profit DOUBLE PRECISION;
    END IF;
    
    -- Add actual_loss column
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_schema = 'my_schema' 
                   AND table_name = 'derivative_suggestions' 
                   AND column_name = 'actual_loss') THEN
        ALTER TABLE my_schema.derivative_suggestions 
        ADD COLUMN actual_loss DOUBLE PRECISION;
    END IF;
    
    -- Add actual_pnl column
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_schema = 'my_schema' 
                   AND table_name = 'derivative_suggestions' 
                   AND column_name = 'actual_pnl') THEN
        ALTER TABLE my_schema.derivative_suggestions 
        ADD COLUMN actual_pnl DOUBLE PRECISION;
    END IF;
    
    -- Add outcome column
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_schema = 'my_schema' 
                   AND table_name = 'derivative_suggestions' 
                   AND column_name = 'outcome') THEN
        ALTER TABLE my_schema.derivative_suggestions 
        ADD COLUMN outcome VARCHAR(20);
    END IF;
    
    -- Add notes column
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_schema = 'my_schema' 
                   AND table_name = 'derivative_suggestions' 
                   AND column_name = 'notes') THEN
        ALTER TABLE my_schema.derivative_suggestions 
        ADD COLUMN notes TEXT;
    END IF;
    
    -- Add updated_at column
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_schema = 'my_schema' 
                   AND table_name = 'derivative_suggestions' 
                   AND column_name = 'updated_at') THEN
        ALTER TABLE my_schema.derivative_suggestions 
        ADD COLUMN updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;
    END IF;
END $$;

-- Create indexes for result tracking fields
CREATE INDEX IF NOT EXISTS idx_deriv_sugg_status ON my_schema.derivative_suggestions(status);
CREATE INDEX IF NOT EXISTS idx_deriv_sugg_outcome ON my_schema.derivative_suggestions(outcome);
CREATE INDEX IF NOT EXISTS idx_deriv_sugg_exit_date ON my_schema.derivative_suggestions(exit_date);

-- Update existing records to have PENDING status if status is NULL
UPDATE my_schema.derivative_suggestions 
SET status = 'PENDING' 
WHERE status IS NULL;

