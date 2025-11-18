-- Performance indexes (idempotent)
CREATE INDEX IF NOT EXISTS idx_holdings_run_date ON my_schema.holdings(run_date);
CREATE INDEX IF NOT EXISTS idx_mf_holdings_run_date ON my_schema.mf_holdings(run_date);
CREATE INDEX IF NOT EXISTS idx_positions_run_date_type ON my_schema.positions(run_date, position_type);
CREATE INDEX IF NOT EXISTS idx_rt_price_scrip_date ON my_schema.rt_intraday_price(scrip_id, price_date DESC);
CREATE INDEX IF NOT EXISTS idx_ticks_inst_ts ON my_schema.ticks(instrument_token, timestamp DESC);
CREATE SCHEMA my_schema;

CREATE TABLE IF NOT EXISTS my_schema.raw_ticks (
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        instrument_token INTEGER,
        ltp FLOAT,
        high FLOAT,
        low FLOAT,
        open FLOAT,
        close FLOAT,
        volume INTEGER,
        run_date DATE DEFAULT CURRENT_DATE
    );

    CREATE TABLE IF NOT EXISTS my_schema.bars (
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        instrument_token INTEGER,
        open FLOAT,
        high FLOAT,
        low FLOAT,
        close FLOAT,
        volume INTEGER,
        PRIMARY KEY (timestamp, instrument_token),
        run_date DATE DEFAULT CURRENT_DATE
    );
    
    CREATE TABLE IF NOT EXISTS my_schema.market_structure (
        session_date DATE,
        instrument_token INTEGER,
        poc FLOAT,
        vah FLOAT,
        val FLOAT,
        day_type VARCHAR(50),
        opening_type VARCHAR(50),
        ib_high FLOAT,
        ib_low FLOAT,
        single_prints JSONB,
        poor_high BOOLEAN,
        poor_low BOOLEAN,
        overnight_high FLOAT,
        overnight_low FLOAT,
        PRIMARY KEY (session_date, instrument_token),
        run_date DATE DEFAULT CURRENT_DATE,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS my_schema.sessions (
        session_date DATE,
        instrument_token INTEGER,
        overnight_high FLOAT,
        overnight_low FLOAT,
        PRIMARY KEY (session_date, instrument_token),
        run_date DATE DEFAULT CURRENT_DATE,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );


CREATE TABLE my_schema.ticks (
    id SERIAL PRIMARY KEY,
    instrument_token BIGINT NOT NULL,
    last_price DECIMAL(15, 2),
    volume BIGINT,
    oi BIGINT,
    open DECIMAL(15, 2),
    high DECIMAL(15, 2),
    low DECIMAL(15, 2),
    close DECIMAL(15, 2),
    run_date DATE DEFAULT CURRENT_DATE,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);  

CREATE TABLE my_schema.market_depth (
    id SERIAL PRIMARY KEY,
    tick_id INTEGER,
    instrument_token BIGINT NOT NULL,
    side VARCHAR(4) NOT NULL CHECK (side IN ('buy', 'sell')),
    price DECIMAL(15, 2) NOT NULL,
    quantity INTEGER NOT NULL,
    orders INTEGER NOT NULL,
    run_date DATE DEFAULT CURRENT_DATE,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS my_schema.profile (
    fetch_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    run_date DATE DEFAULT CURRENT_DATE,
    user_id VARCHAR(50),
    user_name VARCHAR(100),
    email VARCHAR(100),
    user_type VARCHAR(50),
    broker VARCHAR(10),
    products TEXT[],
    order_types TEXT[],
    exchanges TEXT[],
    CONSTRAINT profile_key UNIQUE (user_id)
);

-- Derivative suggestions history (persist strategy suggestions for audit and analysis)
CREATE TABLE IF NOT EXISTS my_schema.derivative_suggestions (
    id SERIAL PRIMARY KEY,
    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    analysis_date DATE,
    source VARCHAR(20) DEFAULT 'TPO',
    strategy_type VARCHAR(40),
    strategy_name VARCHAR(80),
    instrument VARCHAR(100),
    instrument_token BIGINT,
    direction VARCHAR(10),
    quantity INT,
    lot_size INT,
    entry_price DOUBLE PRECISION,
    strike_price DOUBLE PRECISION,
    expiry DATE,
    total_premium DOUBLE PRECISION,
    total_premium_income DOUBLE PRECISION,
    margin_required DOUBLE PRECISION,
    hedge_value DOUBLE PRECISION,
    coverage_percentage DOUBLE PRECISION,
    portfolio_value DOUBLE PRECISION,
    beta DOUBLE PRECISION,
    rationale TEXT,
    tpo_context JSONB,
    diagnostics JSONB,
    potential_profit DOUBLE PRECISION,
    max_potential_profit DOUBLE PRECISION,
    max_profit DOUBLE PRECISION,
    max_loss DOUBLE PRECISION,
    risk_reward_ratio DOUBLE PRECISION,
    probability_of_profit DOUBLE PRECISION,
    breakeven DOUBLE PRECISION,
    payoff_chart TEXT,
    payoff_sparkline TEXT,
    run_date DATE DEFAULT CURRENT_DATE
);

CREATE INDEX IF NOT EXISTS idx_deriv_sugg_generated_at ON my_schema.derivative_suggestions(generated_at);
CREATE INDEX IF NOT EXISTS idx_deriv_sugg_strategy ON my_schema.derivative_suggestions(strategy_type);
CREATE INDEX IF NOT EXISTS idx_deriv_sugg_instrument ON my_schema.derivative_suggestions(instrument);

-- Options Back-testing Results (store back-testing runs and their results)
CREATE TABLE IF NOT EXISTS my_schema.options_backtest_results (
    id SERIAL PRIMARY KEY,
    backtest_name VARCHAR(255),
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    strategy_type VARCHAR(50),
    timeframe_minutes INTEGER,
    show_only_profitable BOOLEAN DEFAULT FALSE,
    min_profit DOUBLE PRECISION,
    total_trades INTEGER,
    win_count INTEGER,
    loss_count INTEGER,
    win_rate DOUBLE PRECISION,
    total_profit_loss DOUBLE PRECISION,
    avg_profit_loss DOUBLE PRECISION,
    gross_profit DOUBLE PRECISION,
    gross_loss DOUBLE PRECISION,
    avg_win DOUBLE PRECISION,
    avg_loss DOUBLE PRECISION,
    max_profit DOUBLE PRECISION,
    max_loss DOUBLE PRECISION,
    profit_factor DOUBLE PRECISION,
    sharpe_ratio DOUBLE PRECISION,
    max_drawdown DOUBLE PRECISION,
    avg_holding_period DOUBLE PRECISION,
    confidence_score DOUBLE PRECISION,
    data_quality JSONB,
    metrics JSONB,
    summary JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    run_date DATE DEFAULT CURRENT_DATE
);

CREATE INDEX IF NOT EXISTS idx_backtest_start_date ON my_schema.options_backtest_results(start_date);
CREATE INDEX IF NOT EXISTS idx_backtest_end_date ON my_schema.options_backtest_results(end_date);
CREATE INDEX IF NOT EXISTS idx_backtest_created_at ON my_schema.options_backtest_results(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_backtest_strategy ON my_schema.options_backtest_results(strategy_type);

-- Options Back-testing Trades (individual trades from back-testing runs)
CREATE TABLE IF NOT EXISTS my_schema.options_backtest_trades (
    id SERIAL PRIMARY KEY,
    backtest_result_id INTEGER REFERENCES my_schema.options_backtest_results(id) ON DELETE CASCADE,
    entry_date DATE NOT NULL,
    exit_date DATE NOT NULL,
    symbol VARCHAR(100),
    option_type VARCHAR(2),
    strike_price DOUBLE PRECISION,
    expiry DATE,
    entry_price DOUBLE PRECISION,
    exit_price DOUBLE PRECISION,
    profit_loss DOUBLE PRECISION,
    exit_reason VARCHAR(50),
    holding_period INTEGER,
    is_generated BOOLEAN DEFAULT FALSE,
    data_source VARCHAR(20),
    trade_details JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_backtest_trades_result_id ON my_schema.options_backtest_trades(backtest_result_id);
CREATE INDEX IF NOT EXISTS idx_backtest_trades_entry_date ON my_schema.options_backtest_trades(entry_date);
CREATE INDEX IF NOT EXISTS idx_backtest_trades_exit_date ON my_schema.options_backtest_trades(exit_date);
CREATE INDEX IF NOT EXISTS idx_backtest_trades_profit_loss ON my_schema.options_backtest_trades(profit_loss);
CREATE INDEX IF NOT EXISTS idx_backtest_trades_is_generated ON my_schema.options_backtest_trades(is_generated);

CREATE TABLE my_schema.orders (
    order_id VARCHAR(50) NOT NULL, -- Adjust length based on actual data
    parent_order_id VARCHAR(50),   -- Nullable, as it can be None
    exchange_order_id VARCHAR(50), -- Nullable
    status VARCHAR(20),            -- e.g., 'COMPLETE'
    status_message TEXT,           -- Nullable, for longer messages
    order_type VARCHAR(20),        -- e.g., 'MARKET'
    transaction_type VARCHAR(20),  -- e.g., 'SELL'
    exchange VARCHAR(10),          -- e.g., 'NSE'
    trading_symbol VARCHAR(50),    -- e.g., 'SETF10GILT'
    instrument_token BIGINT,       -- e.g., 4453121
    quantity INTEGER,              -- e.g., 5
    price NUMERIC(15, 2),         -- e.g., 0 (for MARKET orders)
    trigger_price NUMERIC(15, 2), -- e.g., 0
    average_price NUMERIC(15, 2), -- e.g., 258.5
    order_timestamp TIMESTAMP,     -- e.g., '2025-10-20 09:34:26'
    exchange_timestamp TIMESTAMP,  -- e.g., '2025-10-20 09:34:26'
    run_date date DEFAULT CURRENT_DATE,
    CONSTRAINT orders_order_id_key UNIQUE (order_id)
);
--CREATE UNIQUE INDEX orders_order_id_idx ON my_schema.orders (order_id);

CREATE TABLE IF NOT EXISTS my_schema.trades (
    fetch_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    run_date DATE DEFAULT CURRENT_DATE,
    trade_id VARCHAR(50),
    order_id VARCHAR(50),
    exchange_order_id VARCHAR(50),
    exchange VARCHAR(20),
    trading_symbol VARCHAR(50),
    instrument_token INTEGER,
    transaction_type VARCHAR(20),
    quantity INTEGER,
    average_price FLOAT,
    trade_timestamp TIMESTAMP,
    exchange_timestamp TIMESTAMP,
    CONSTRAINT trades_unique_key UNIQUE (trade_id)
);

CREATE TABLE IF NOT EXISTS my_schema.positions (
    fetch_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    run_date DATE DEFAULT CURRENT_DATE,
    user_id VARCHAR(50),
    position_type VARCHAR(10), -- 'net' or 'day'
    trading_symbol VARCHAR(50),
    instrument_token INTEGER,
    exchange VARCHAR(20),
    product VARCHAR(20),
    quantity INTEGER,
    buy_qty INTEGER,
    sell_qty INTEGER,
    buy_price FLOAT,
    sell_price FLOAT,
    average_price FLOAT,
    last_price FLOAT,
    pnl FLOAT,
    m2m FLOAT, -- Mark-to-market
    realised FLOAT,
    unrealised FLOAT,
    value FLOAT,
    CONSTRAINT positions_unique_key UNIQUE (position_type, instrument_token)
);

CREATE TABLE IF NOT EXISTS my_schema.holdings (
    fetch_timestamp TIMESTAMP DEFAULT current_timestamp,
    run_date DATE DEFAULT CURRENT_DATE,
    trading_symbol VARCHAR(50),
    instrument_token INTEGER,
    isin VARCHAR(12),
    quantity INTEGER,
    t1_quantity INTEGER,
    authorised_quantity INTEGER,
    average_price FLOAT,
    close_price FLOAT,
    last_price FLOAT,
    pnl FLOAT,
    collateral_quantity INTEGER,
    collateral_type VARCHAR(20),
    CONSTRAINT holdings_unique_key UNIQUE (instrument_token, run_date)
);

CREATE TABLE IF NOT EXISTS my_schema.mf_holdings (
    fetch_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    run_date DATE DEFAULT CURRENT_DATE,
    folio VARCHAR(100),
    fund VARCHAR(200),
    tradingsymbol VARCHAR(50),
    isin VARCHAR(12),
    quantity FLOAT,
    average_price FLOAT,
    last_price FLOAT,
    invested_amount FLOAT,
    current_value FLOAT,
    pnl FLOAT,
    net_change_percentage FLOAT,
    day_change_percentage FLOAT,
    CONSTRAINT mf_holdings_unique_key UNIQUE (folio, tradingsymbol, run_date)
);

-- Indexes for MF holdings table
CREATE INDEX IF NOT EXISTS idx_mf_holdings_run_date ON my_schema.mf_holdings(run_date);
CREATE INDEX IF NOT EXISTS idx_mf_holdings_tradingsymbol ON my_schema.mf_holdings(tradingsymbol);

-- Irrational gains/losses analysis table
CREATE TABLE IF NOT EXISTS my_schema.holdings_irrational_analysis (
    id SERIAL PRIMARY KEY,
    trading_symbol VARCHAR(50) NOT NULL,
    instrument_token INTEGER,
    analysis_date DATE DEFAULT CURRENT_DATE,
    run_date DATE DEFAULT CURRENT_DATE,
    
    -- Current metrics
    pnl_pct_change FLOAT,
    today_pnl_pct FLOAT,
    current_price FLOAT,
    average_price FLOAT,
    
    -- Analysis flags
    is_statistical_outlier BOOLEAN DEFAULT FALSE,
    z_score FLOAT,
    portfolio_avg_pnl FLOAT,
    portfolio_std_dev FLOAT,
    
    is_market_mismatch BOOLEAN DEFAULT FALSE,
    market_change_pct FLOAT,
    correlation_score FLOAT,
    
    is_technical_mismatch BOOLEAN DEFAULT FALSE,
    rsi FLOAT,
    macd_divergence BOOLEAN DEFAULT FALSE,
    volume_anomaly BOOLEAN DEFAULT FALSE,
    volume_ratio FLOAT,
    
    is_time_anomaly BOOLEAN DEFAULT FALSE,
    days_since_large_move INTEGER,
    move_velocity FLOAT, -- % change per day
    
    is_fundamental_mismatch BOOLEAN DEFAULT FALSE,
    pe_ratio FLOAT,
    prophet_prediction_diff FLOAT,
    news_sentiment_diff FLOAT,
    
    -- Overall assessment
    irrational_score FLOAT, -- 0-100, higher = more irrational
    irrational_type VARCHAR(50), -- 'gain' or 'loss'
    exit_recommendation VARCHAR(20), -- 'STRONG', 'MODERATE', 'WEAK', 'NONE'
    exit_reason TEXT,
    
    -- Metadata
    analysis_details JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT holdings_irrational_unique UNIQUE (trading_symbol, analysis_date)
);

-- Indexes for irrational analysis table
CREATE INDEX IF NOT EXISTS idx_irrational_analysis_date ON my_schema.holdings_irrational_analysis(analysis_date);
CREATE INDEX IF NOT EXISTS idx_irrational_score ON my_schema.holdings_irrational_analysis(irrational_score DESC);
CREATE INDEX IF NOT EXISTS idx_irrational_exit_recommendation ON my_schema.holdings_irrational_analysis(exit_recommendation);
CREATE INDEX IF NOT EXISTS idx_mf_holdings_folio ON my_schema.mf_holdings(folio);

-- Comments on MF holdings table
COMMENT ON TABLE my_schema.mf_holdings IS 'Stores Mutual Fund holdings data fetched from Kite API';
COMMENT ON COLUMN my_schema.mf_holdings.folio IS 'MF Folio number';
COMMENT ON COLUMN my_schema.mf_holdings.fund IS 'Mutual Fund name';
COMMENT ON COLUMN my_schema.mf_holdings.invested_amount IS 'Total amount invested in the MF';
COMMENT ON COLUMN my_schema.mf_holdings.current_value IS 'Current value of the holding';
COMMENT ON COLUMN my_schema.mf_holdings.net_change_percentage IS 'Overall change percentage from invested amount';
COMMENT ON COLUMN my_schema.mf_holdings.day_change_percentage IS 'Change percentage for the day';

-- MF NAV History table
CREATE TABLE IF NOT EXISTS my_schema.mf_nav_history (
    mf_symbol VARCHAR(50) NOT NULL,
    scheme_code VARCHAR(20),  -- AMFI scheme code
    fund_name VARCHAR(200),
    yahoo_symbol VARCHAR(50),  -- Yahoo Finance symbol (may differ from Zerodha symbol)
    nav_date DATE NOT NULL,
    nav_value FLOAT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
	run_date date default current_date,
    CONSTRAINT mf_nav_history_pk PRIMARY KEY (mf_symbol, nav_date)
);

-- Add yahoo_symbol column if it doesn't exist (for existing installations)
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'my_schema' 
        AND table_name = 'mf_nav_history' 
        AND column_name = 'yahoo_symbol'
    ) THEN
        ALTER TABLE my_schema.mf_nav_history ADD COLUMN yahoo_symbol VARCHAR(50);
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_mf_nav_symbol_date ON my_schema.mf_nav_history(mf_symbol, nav_date DESC);
CREATE INDEX IF NOT EXISTS idx_mf_nav_scheme_code ON my_schema.mf_nav_history(scheme_code);

-- Comments on MF NAV history table
COMMENT ON TABLE my_schema.mf_nav_history IS 'Stores historical NAV data for Mutual Funds from AMFI or Yahoo Finance';
COMMENT ON COLUMN my_schema.mf_nav_history.mf_symbol IS 'Mutual Fund trading symbol';
COMMENT ON COLUMN my_schema.mf_nav_history.scheme_code IS 'AMFI scheme code for the mutual fund';
COMMENT ON COLUMN my_schema.mf_nav_history.fund_name IS 'Mutual Fund name';
COMMENT ON COLUMN my_schema.mf_nav_history.nav_date IS 'Date of NAV value';
COMMENT ON COLUMN my_schema.mf_nav_history.nav_value IS 'Net Asset Value on the given date';

-- MF Portfolio Holdings table (constituent stocks in each MF)
CREATE TABLE IF NOT EXISTS my_schema.mf_portfolio_holdings (
    id SERIAL PRIMARY KEY,
    mf_symbol VARCHAR(50) NOT NULL,
    scheme_code VARCHAR(20),
    stock_symbol VARCHAR(50) NOT NULL,
    stock_name VARCHAR(200),
    weight_pct FLOAT,
    quantity FLOAT,
    value FLOAT,
    sector VARCHAR(100),
    portfolio_date DATE NOT NULL,
    fetch_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT mf_portfolio_unique UNIQUE (mf_symbol, stock_symbol, portfolio_date)
);

CREATE INDEX IF NOT EXISTS idx_mf_portfolio_mf_symbol ON my_schema.mf_portfolio_holdings(mf_symbol, portfolio_date DESC);
CREATE INDEX IF NOT EXISTS idx_mf_portfolio_stock_symbol ON my_schema.mf_portfolio_holdings(stock_symbol);
CREATE INDEX IF NOT EXISTS idx_mf_portfolio_date ON my_schema.mf_portfolio_holdings(portfolio_date DESC);

-- Comments on MF portfolio holdings table
COMMENT ON TABLE my_schema.mf_portfolio_holdings IS 'Stores constituent stock holdings for Mutual Funds';
COMMENT ON COLUMN my_schema.mf_portfolio_holdings.mf_symbol IS 'Mutual Fund trading symbol';
COMMENT ON COLUMN my_schema.mf_portfolio_holdings.scheme_code IS 'AMFI scheme code for the mutual fund';
COMMENT ON COLUMN my_schema.mf_portfolio_holdings.stock_symbol IS 'Stock trading symbol (constituent)';
COMMENT ON COLUMN my_schema.mf_portfolio_holdings.weight_pct IS 'Percentage weight of stock in MF portfolio';
COMMENT ON COLUMN my_schema.mf_portfolio_holdings.portfolio_date IS 'Date of portfolio snapshot';

CREATE TABLE IF NOT EXISTS my_schema.margins (
    fetch_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    run_date DATE DEFAULT CURRENT_DATE,
    margin_type VARCHAR(20), -- 'equity' or 'commodity'
    enabled BOOLEAN,
    net FLOAT,
    available_adhoc_margin FLOAT,
    available_cash FLOAT,
    available_opening_balance FLOAT,
    available_live_balance FLOAT,
    available_collateral FLOAT,
    available_intraday_payin FLOAT,
    utilised_debits FLOAT,
    utilised_exposure FLOAT,
    utilised_m2m_realised FLOAT,
    utilised_m2m_unrealised FLOAT,
    utilised_option_premium FLOAT,
    utilised_payout FLOAT,
    utilised_span FLOAT,
    utilised_holding_sales FLOAT,
    utilised_turnover FLOAT,
    utilised_liquid_collateral FLOAT,
    utilised_stock_collateral FLOAT,
    utilised_equity FLOAT,
    utilised_delivery FLOAT,
    CONSTRAINT margins_unique_key UNIQUE (margin_type)
);

-- public.master_scrips definition

-- Drop table

-- DROP TABLE public.master_scrips;

CREATE TABLE my_schema.master_scrips (
	scrip_id varchar(10) NULL,
	scrip_screener_code int4 NULL,
	sector_code varchar(10) NULL,
	created_at timestamp DEFAULT now() NULL,
	yahoo_code varchar NULL,
	scrip_group varchar NULL,
	scrip_mcap int4 NULL,
	scrip_country varchar NULL,
	ownership varchar NULL,
	fno bool NULL,
	hist_roce float4 NULL,
	debt_to_equity float4 NULL,
	ps_ratio float4 NULL,
	updated_at timestamp DEFAULT now() NULL
);
CREATE INDEX master_scrips_scrip_country_idx ON my_schema.master_scrips USING btree (scrip_country);
CREATE UNIQUE INDEX master_scrips_scrip_id_idx ON my_schema.master_scrips USING btree (scrip_id);
CREATE UNIQUE INDEX master_scrips_scrip_id_idx1 ON my_schema.master_scrips USING btree (scrip_id, scrip_country);
CREATE INDEX master_scrips_sector_code_idx ON my_schema.master_scrips USING btree (sector_code);

-- public.rt_intraday_price definition

-- Drop table

-- DROP TABLE public.rt_intraday_price;

CREATE TABLE my_schema.rt_intraday_price (
	scrip_id varchar NOT NULL,
	insert_date date NULL,
	price_open float8 NULL,
	price_high float8 NULL,
	price_low float8 NULL,
	price_close float8 NULL,
	price_lc float8 NULL,
	price_uc float8 NULL,
	volume int8 NULL,
	delivery int8 NULL,
	created_at timestamp NULL,
	dma50 float8 NULL,
	dma200 float8 NULL,
	price_date varchar NOT NULL,
	country varchar NULL,
	CONSTRAINT rt_intraday_price_pk1 PRIMARY KEY (scrip_id, price_date)
);
CREATE INDEX rt_intraday_price2_price_date2_idx ON my_schema.rt_intraday_price USING btree (price_date, country);
CREATE INDEX rt_intraday_price2_price_date_idx ON my_schema.rt_intraday_price USING btree (price_date);
CREATE INDEX rt_intraday_price2_volume_idx ON my_schema.rt_intraday_price USING btree (volume, price_date, country);


-- my_schema.instruments definition

-- Drop table

-- DROP TABLE my_schema.instruments;

CREATE TABLE my_schema.instruments (
	instrument_token int4 NOT NULL,
	exchange_token int4 NULL,
	tradingsymbol varchar NOT NULL,
	"name" varchar NULL,
	last_price float4 NULL,
	expiry date NULL,
	strike float4 NULL,
	tick_size float4 NULL,
	lot_size int4 NULL,
	instrument_type varchar NULL,
	segment varchar NULL,
	exchange varchar NULL,
	"timestamp" timestamp DEFAULT CURRENT_TIMESTAMP NULL,
	run_date date DEFAULT CURRENT_DATE NULL,
	CONSTRAINT instruments_pk1 PRIMARY KEY (instrument_token, expiry, strike)
);


CREATE TABLE my_schema.futures_ticks (
    id SERIAL PRIMARY KEY,
    instrument_token BIGINT NOT NULL,
    timestamp TIMESTAMP default current_timestamp,
    run_date date default current_date,
    last_trade_time TIMESTAMP,
    last_price DOUBLE PRECISION,
    last_quantity INT,
    buy_quantity BIGINT,
    sell_quantity BIGINT,
    volume BIGINT,
    average_price DOUBLE PRECISION,
    oi BIGINT,
    oi_day_high BIGINT,
    oi_day_low BIGINT,
    net_change DOUBLE PRECISION,
    lower_circuit_limit DOUBLE PRECISION,
    upper_circuit_limit DOUBLE PRECISION
);

CREATE TABLE my_schema.futures_tick_ohlc (
    tick_id INT REFERENCES my_schema.ticks(id) ON DELETE CASCADE,
    open DOUBLE PRECISION,
    high DOUBLE PRECISION,
    low DOUBLE PRECISION,
    close DOUBLE PRECISION,
    PRIMARY KEY (tick_id),
    timestamp TIMESTAMP default current_timestamp,
    run_date date default current_date
);

CREATE TABLE my_schema.futures_tick_depth (
    id SERIAL PRIMARY KEY,
    tick_id INT REFERENCES my_schema.ticks(id) ON DELETE CASCADE,
    side VARCHAR(4) CHECK (side IN ('buy', 'sell')),
    price DOUBLE PRECISION,
    quantity BIGINT,
    orders INT,
    timestamp TIMESTAMP default current_timestamp,
    run_date date default current_date
);

-- Options ticks table (similar to futures_ticks but with strike, option_type, expiry)
CREATE TABLE my_schema.options_ticks (
    id SERIAL PRIMARY KEY,
    instrument_token BIGINT NOT NULL,
    timestamp TIMESTAMP default current_timestamp,
    run_date date default current_date,
    last_trade_time TIMESTAMP,
    last_price DOUBLE PRECISION,
    last_quantity INT,
    buy_quantity BIGINT,
    sell_quantity BIGINT,
    volume BIGINT,
    average_price DOUBLE PRECISION,
    oi BIGINT,
    oi_day_high BIGINT,
    oi_day_low BIGINT,
    net_change DOUBLE PRECISION,
    lower_circuit_limit DOUBLE PRECISION,
    upper_circuit_limit DOUBLE PRECISION,
    strike_price DOUBLE PRECISION,
    option_type VARCHAR(2) CHECK (option_type IN ('CE', 'PE')),
    expiry DATE,
    tradingsymbol VARCHAR(100)
);

CREATE TABLE my_schema.options_tick_ohlc (
    tick_id INT REFERENCES my_schema.options_ticks(id) ON DELETE CASCADE,
    open DOUBLE PRECISION,
    high DOUBLE PRECISION,
    low DOUBLE PRECISION,
    close DOUBLE PRECISION,
    PRIMARY KEY (tick_id),
    timestamp TIMESTAMP default current_timestamp,
    run_date date default current_date
);

CREATE TABLE my_schema.options_tick_depth (
    id SERIAL PRIMARY KEY,
    tick_id INT REFERENCES my_schema.options_ticks(id) ON DELETE CASCADE,
    side VARCHAR(4) CHECK (side IN ('buy', 'sell')),
    price DOUBLE PRECISION,
    quantity BIGINT,
    orders INT,
    timestamp TIMESTAMP default current_timestamp,
    run_date date default current_date
);

-- Indexes for options_ticks table
CREATE INDEX IF NOT EXISTS idx_options_ticks_instrument_token ON my_schema.options_ticks(instrument_token);
CREATE INDEX IF NOT EXISTS idx_options_ticks_expiry ON my_schema.options_ticks(expiry);
CREATE INDEX IF NOT EXISTS idx_options_ticks_strike_price ON my_schema.options_ticks(strike_price);
CREATE INDEX IF NOT EXISTS idx_options_ticks_option_type ON my_schema.options_ticks(option_type);
CREATE INDEX IF NOT EXISTS idx_options_ticks_timestamp ON my_schema.options_ticks(timestamp);
CREATE INDEX IF NOT EXISTS idx_options_ticks_run_date ON my_schema.options_ticks(run_date);
CREATE INDEX IF NOT EXISTS idx_options_ticks_expiry_strike_type ON my_schema.options_ticks(expiry, strike_price, option_type);

-- IV History Table for IV Rank calculation
CREATE TABLE IF NOT EXISTS my_schema.iv_history (
    instrument_token BIGINT NOT NULL,
    strike_price DOUBLE PRECISION NOT NULL,
    option_type VARCHAR(2) CHECK (option_type IN ('CE', 'PE')) NOT NULL,
    expiry DATE NOT NULL,
    iv DOUBLE PRECISION NOT NULL,
    price_date DATE NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (instrument_token, strike_price, option_type, expiry, price_date)
);

CREATE INDEX IF NOT EXISTS idx_iv_history_instrument_token ON my_schema.iv_history(instrument_token);
CREATE INDEX IF NOT EXISTS idx_iv_history_expiry ON my_schema.iv_history(expiry);
CREATE INDEX IF NOT EXISTS idx_iv_history_price_date ON my_schema.iv_history(price_date DESC);
CREATE INDEX IF NOT EXISTS idx_iv_history_strike_type_expiry ON my_schema.iv_history(strike_price, option_type, expiry);

-- Indexes for options_tick_depth table
CREATE INDEX IF NOT EXISTS idx_options_tick_depth_tick_id ON my_schema.options_tick_depth(tick_id);

                CREATE TABLE IF NOT EXISTS my_schema.tpo_analysis (
                    analysis_date DATE,
                    instrument_token INTEGER,
                    session_type VARCHAR(20),
                    poc FLOAT,
                    value_area_high FLOAT,
                    value_area_low FLOAT,
                    initial_balance_high FLOAT,
                    initial_balance_low FLOAT,
                    confidence_score FLOAT,
                    created_at TIMESTAMP DEFAULT current_timestamp,
                    CONSTRAINT tpo_analysis_unique_key UNIQUE (analysis_date, instrument_token, session_type)
            	);

				                CREATE TABLE IF NOT EXISTS my_schema.holdings (
                    fetch_timestamp TIMESTAMP DEFAULT current_timestamp,
                    run_date DATE DEFAULT CURRENT_DATE,
                    trading_symbol VARCHAR(50),
                    instrument_token INTEGER,
                    isin VARCHAR(12),
                    quantity INTEGER,
                    t1_quantity INTEGER,
                    authorised_quantity INTEGER,
                    average_price FLOAT,
                    close_price FLOAT,
                    last_price FLOAT,
                    pnl FLOAT,
                    collateral_quantity INTEGER,
                    collateral_type VARCHAR(20),
                    CONSTRAINT holdings_unique_key UNIQUE (instrument_token, run_date)
                );


-- Table to store GTT (Good Till Triggered) transactions and responses
CREATE TABLE IF NOT EXISTS my_schema.gtt_transactions (
    id SERIAL PRIMARY KEY,
    trigger_id INTEGER,
    trading_symbol VARCHAR(50) NOT NULL,
    exchange VARCHAR(20) NOT NULL,
    instrument_token INTEGER,
    quantity INTEGER NOT NULL,
    trigger_price FLOAT NOT NULL,
    last_price FLOAT NOT NULL,
    order_price FLOAT NOT NULL,
    order_type VARCHAR(20) DEFAULT 'LIMIT',
    transaction_type VARCHAR(10) DEFAULT 'SELL',
    product VARCHAR(10) DEFAULT 'CNC',
    status VARCHAR(20) DEFAULT 'ACTIVE',
    gtt_type VARCHAR(20) DEFAULT 'SINGLE',
    response JSONB,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    cancelled_at TIMESTAMP,
    run_date DATE DEFAULT CURRENT_DATE,
    notes TEXT
);

-- Indexes for GTT transactions table
CREATE INDEX IF NOT EXISTS idx_gtt_transactions_trigger_id ON my_schema.gtt_transactions(trigger_id);
CREATE INDEX IF NOT EXISTS idx_gtt_transactions_trading_symbol ON my_schema.gtt_transactions(trading_symbol);
CREATE INDEX IF NOT EXISTS idx_gtt_transactions_status ON my_schema.gtt_transactions(status);
CREATE INDEX IF NOT EXISTS idx_gtt_transactions_run_date ON my_schema.gtt_transactions(run_date);
CREATE INDEX IF NOT EXISTS idx_gtt_transactions_created_at ON my_schema.gtt_transactions(created_at);

-- Comments on GTT transactions table
COMMENT ON TABLE my_schema.gtt_transactions IS 'Stores all GTT (Good Till Triggered) order transactions and their responses from Kite API';
COMMENT ON COLUMN my_schema.gtt_transactions.trigger_id IS 'Kite API trigger ID for the GTT order';
COMMENT ON COLUMN my_schema.gtt_transactions.status IS 'Current status: ACTIVE, CANCELLED, TRIGGERED, etc.';
COMMENT ON COLUMN my_schema.gtt_transactions.response IS 'Full JSON response from Kite API';
COMMENT ON COLUMN my_schema.gtt_transactions.notes IS 'Additional notes or metadata about the GTT order';

-- Table to store Swing Trade Recommendations (persist recommendations for audit and analysis)
CREATE TABLE IF NOT EXISTS my_schema.swing_trade_suggestions (
    id SERIAL PRIMARY KEY,
    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    analysis_date DATE,
    run_date DATE DEFAULT CURRENT_DATE,
    scrip_id VARCHAR(10),
    instrument_token BIGINT,
    pattern_type VARCHAR(50),
    direction VARCHAR(10) DEFAULT 'BUY',
    entry_price DOUBLE PRECISION,
    target_price DOUBLE PRECISION,
    stop_loss DOUBLE PRECISION,
    potential_gain_pct DOUBLE PRECISION,
    risk_reward_ratio DOUBLE PRECISION,
    confidence_score DOUBLE PRECISION,
    holding_period_days INT,
    current_price DOUBLE PRECISION,
    sma_20 DOUBLE PRECISION,
    sma_50 DOUBLE PRECISION,
    sma_200 DOUBLE PRECISION,
    rsi_14 DOUBLE PRECISION,
    macd DOUBLE PRECISION,
    macd_signal DOUBLE PRECISION,
    atr_14 DOUBLE PRECISION,
    volume_trend VARCHAR(20),
    support_level DOUBLE PRECISION,
    resistance_level DOUBLE PRECISION,
    rationale TEXT,
    technical_context JSONB,
    diagnostics JSONB,
    filtering_criteria JSONB,
    status VARCHAR(20) DEFAULT 'ACTIVE'
);

-- Indexes for swing trade suggestions table
CREATE INDEX IF NOT EXISTS idx_swing_trade_analysis_date ON my_schema.swing_trade_suggestions(analysis_date);
CREATE INDEX IF NOT EXISTS idx_swing_trade_scrip_id ON my_schema.swing_trade_suggestions(scrip_id);
CREATE INDEX IF NOT EXISTS idx_swing_trade_pattern ON my_schema.swing_trade_suggestions(pattern_type);
CREATE INDEX IF NOT EXISTS idx_swing_trade_confidence ON my_schema.swing_trade_suggestions(confidence_score DESC);
CREATE INDEX IF NOT EXISTS idx_swing_trade_run_date ON my_schema.swing_trade_suggestions(run_date);

-- Comments on swing trade suggestions table
COMMENT ON TABLE my_schema.swing_trade_suggestions IS 'Stores swing trade recommendations generated from technical analysis';
COMMENT ON COLUMN my_schema.swing_trade_suggestions.analysis_date IS 'Date when the analysis was performed';
COMMENT ON COLUMN my_schema.swing_trade_suggestions.filtering_criteria IS 'JSON object containing the filtering criteria used to generate this recommendation';

-- Table to store Prophet price predictions (30-day forecasts)
CREATE TABLE IF NOT EXISTS my_schema.prophet_predictions (
    id SERIAL PRIMARY KEY,
    scrip_id VARCHAR(10) NOT NULL,
    run_date DATE DEFAULT CURRENT_DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    prediction_days INTEGER DEFAULT 30,
    current_price DOUBLE PRECISION,
    predicted_price_30d DOUBLE PRECISION,
    predicted_price_change_pct DOUBLE PRECISION,
    prediction_confidence DOUBLE PRECISION,
    prediction_details JSONB,
    status VARCHAR(20) DEFAULT 'ACTIVE',
    UNIQUE(scrip_id, run_date, prediction_days)
);

-- Indexes for prophet predictions table
CREATE INDEX IF NOT EXISTS idx_prophet_run_date ON my_schema.prophet_predictions(run_date);
CREATE INDEX IF NOT EXISTS idx_prophet_scrip_id ON my_schema.prophet_predictions(scrip_id);
CREATE INDEX IF NOT EXISTS idx_prophet_price_change ON my_schema.prophet_predictions(predicted_price_change_pct DESC);
CREATE INDEX IF NOT EXISTS idx_prophet_status ON my_schema.prophet_predictions(status);

-- Comments on prophet predictions table
COMMENT ON TABLE my_schema.prophet_predictions IS 'Stores Prophet-based 30-day price predictions for stocks';
COMMENT ON COLUMN my_schema.prophet_predictions.predicted_price_30d IS 'Predicted price after 30 days';
COMMENT ON COLUMN my_schema.prophet_predictions.predicted_price_change_pct IS 'Predicted percentage change from current price to 30-day target';
COMMENT ON COLUMN my_schema.prophet_predictions.prediction_confidence IS 'Confidence score of the prediction (0-100)';
COMMENT ON COLUMN my_schema.prophet_predictions.prediction_details IS 'JSON object containing detailed prediction data including daily forecasts';

-- Fundamental Data and Sentiment Analysis Tables

-- Fundamental data table to store metrics from screener.in
CREATE TABLE IF NOT EXISTS my_schema.fundamental_data (
    id SERIAL PRIMARY KEY,
    scrip_id VARCHAR(10) NOT NULL,
    fetch_date DATE DEFAULT CURRENT_DATE,
    pe_ratio DOUBLE PRECISION,
    pb_ratio DOUBLE PRECISION,
    debt_to_equity DOUBLE PRECISION,
    roe DOUBLE PRECISION,
    roce DOUBLE PRECISION,
    current_ratio DOUBLE PRECISION,
    quick_ratio DOUBLE PRECISION,
    eps DOUBLE PRECISION,
    revenue_growth DOUBLE PRECISION,
    profit_growth DOUBLE PRECISION,
    dividend_yield DOUBLE PRECISION,
    market_cap DOUBLE PRECISION,
    raw_data JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(scrip_id, fetch_date)
);

-- Indexes for fundamental_data table
CREATE INDEX IF NOT EXISTS idx_fundamental_scrip_id ON my_schema.fundamental_data(scrip_id);
CREATE INDEX IF NOT EXISTS idx_fundamental_fetch_date ON my_schema.fundamental_data(fetch_date);
CREATE INDEX IF NOT EXISTS idx_fundamental_scrip_date ON my_schema.fundamental_data(scrip_id, fetch_date DESC);
CREATE INDEX IF NOT EXISTS idx_fundamental_date_scrip ON my_schema.fundamental_data(fetch_date DESC, scrip_id);

-- News sentiment table to store news articles and sentiment scores
CREATE TABLE IF NOT EXISTS my_schema.news_sentiment (
    id SERIAL PRIMARY KEY,
    scrip_id VARCHAR(10) NOT NULL,
    article_date DATE NOT NULL,
    source VARCHAR(200),
    title TEXT,
    content TEXT,
    sentiment_score DOUBLE PRECISION,  -- -1 to 1 range
    sentiment_label VARCHAR(20),  -- positive/negative/neutral
    raw_data JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(scrip_id, article_date, source, title)
);

-- Indexes for news_sentiment table
CREATE INDEX IF NOT EXISTS idx_news_sentiment_scrip_id ON my_schema.news_sentiment(scrip_id);
CREATE INDEX IF NOT EXISTS idx_news_sentiment_article_date ON my_schema.news_sentiment(article_date);

-- Accumulation/Distribution Analysis Tables
-- my_schema.accumulation_distribution definition

-- Drop table

-- DROP TABLE my_schema.accumulation_distribution;

CREATE TABLE my_schema.accumulation_distribution (
	id serial4 NOT NULL,
	scrip_id varchar(10) NOT NULL,
	analysis_date date NOT NULL,
	run_date date DEFAULT CURRENT_DATE NULL,
	state varchar(20) NULL,
	start_date date NULL,
	days_in_state int4 NULL,
	obv_value float8 NULL,
	ad_value float8 NULL,
	momentum_score float8 NULL,
	pattern_detected varchar(50) NULL,
	volume_analysis jsonb NULL,
	confidence_score float8 NULL,
	technical_context jsonb NULL,
	created_at timestamp DEFAULT CURRENT_TIMESTAMP NULL,
	CONSTRAINT accumulation_distribution_pkey PRIMARY KEY (id),
	CONSTRAINT accumulation_distribution_scrip_id_analysis_date_key UNIQUE (scrip_id, analysis_date)
);
CREATE INDEX idx_accumulation_distribution_analysis_date ON my_schema.accumulation_distribution USING btree (analysis_date);
CREATE INDEX idx_accumulation_distribution_scrip_date ON my_schema.accumulation_distribution USING btree (scrip_id, analysis_date DESC);
CREATE INDEX idx_accumulation_distribution_scrip_id ON my_schema.accumulation_distribution USING btree (scrip_id);
CREATE INDEX idx_accumulation_distribution_start_date ON my_schema.accumulation_distribution USING btree (start_date);
CREATE INDEX idx_accumulation_distribution_state ON my_schema.accumulation_distribution USING btree (state);

-- my_schema.accumulation_distribution_history definition

-- Drop table

-- DROP TABLE my_schema.accumulation_distribution_history;

CREATE TABLE my_schema.accumulation_distribution_history (
	id serial4 NOT NULL,
	scrip_id varchar(10) NOT NULL,
	state varchar(20) NULL,
	start_date date NOT NULL,
	end_date date NULL,
	duration_days int4 NULL,
	created_at timestamp DEFAULT CURRENT_TIMESTAMP NULL,
	CONSTRAINT accumulation_distribution_history_pkey PRIMARY KEY (id)
);
CREATE INDEX idx_accumulation_distribution_history_scrip_id ON my_schema.accumulation_distribution_history USING btree (scrip_id);
CREATE INDEX idx_accumulation_distribution_history_scrip_start ON my_schema.accumulation_distribution_history USING btree (scrip_id, start_date DESC);
CREATE INDEX idx_accumulation_distribution_history_start_date ON my_schema.accumulation_distribution_history USING btree (start_date);
CREATE INDEX idx_accumulation_distribution_history_state ON my_schema.accumulation_distribution_history USING btree (state);


-- Table to store Accumulation/Distribution analysis results

CREATE INDEX IF NOT EXISTS idx_news_sentiment_sentiment_score ON my_schema.news_sentiment(sentiment_score);

-- Combined sentiment table to store aggregated sentiment scores per stock
CREATE TABLE IF NOT EXISTS my_schema.combined_sentiment (
    id SERIAL PRIMARY KEY,
    scrip_id VARCHAR(10) NOT NULL,
    calculation_date DATE DEFAULT CURRENT_DATE,
    news_sentiment_score DOUBLE PRECISION,  -- -1 to 1 range
    fundamental_sentiment_score DOUBLE PRECISION,  -- -1 to 1 range
    combined_sentiment_score DOUBLE PRECISION,  -- -1 to 1 range
    news_weight DOUBLE PRECISION DEFAULT 0.5,
    fundamental_weight DOUBLE PRECISION DEFAULT 0.5,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(scrip_id, calculation_date)
);

-- Indexes for combined_sentiment table
CREATE INDEX IF NOT EXISTS idx_combined_sentiment_scrip_id ON my_schema.combined_sentiment(scrip_id);
CREATE INDEX IF NOT EXISTS idx_combined_sentiment_calculation_date ON my_schema.combined_sentiment(calculation_date);

-- Add sentiment columns to prophet_predictions table
ALTER TABLE my_schema.prophet_predictions 
ADD COLUMN IF NOT EXISTS news_sentiment_score DOUBLE PRECISION,
ADD COLUMN IF NOT EXISTS fundamental_sentiment_score DOUBLE PRECISION,
ADD COLUMN IF NOT EXISTS combined_sentiment_score DOUBLE PRECISION,
ADD COLUMN IF NOT EXISTS enhanced_predicted_price_change_pct DOUBLE PRECISION,
ADD COLUMN IF NOT EXISTS enhanced_prediction_confidence DOUBLE PRECISION,
ADD COLUMN IF NOT EXISTS sentiment_metadata JSONB;

-- Indexes for new sentiment columns
CREATE INDEX IF NOT EXISTS idx_prophet_combined_sentiment ON my_schema.prophet_predictions(combined_sentiment_score);
CREATE INDEX IF NOT EXISTS idx_prophet_enhanced_price_change ON my_schema.prophet_predictions(enhanced_predicted_price_change_pct DESC);



-- public.rt_scrip_actions definition

-- Drop table

-- DROP TABLE public.rt_scrip_actions;

CREATE TABLE public.rt_scrip_actions (
	scrip_id varchar NOT NULL,
	run_date varchar DEFAULT CURRENT_DATE NOT NULL,
	dividend float4 NULL,
	split float4 NULL,
	created_at timestamp DEFAULT CURRENT_TIMESTAMP NOT NULL,
	trans_date date NOT NULL,
	CONSTRAINT rt_scrip_actions_pk PRIMARY KEY (scrip_id, trans_date)
);

-- public.rt_quarterly_income definition

-- Drop table

-- DROP TABLE public.rt_quarterly_income;

CREATE TABLE  if not exists public.rt_quarterly_income (
	scrip_id varchar NOT NULL,
	run_date date DEFAULT CURRENT_DATE NOT NULL,
	trans_date date NOT NULL,
	trans_tag varchar NOT NULL,
	trans_value varchar NOT NULL,
	created_at timestamp DEFAULT CURRENT_TIMESTAMP NOT NULL,
	CONSTRAINT rt_quarterly_income_pk PRIMARY KEY (scrip_id, trans_date, trans_tag, trans_value)
);

-- public.rt_insider_trans definition

-- Drop table

-- DROP TABLE public.rt_insider_trans;

CREATE TABLE if not exists public.rt_insider_trans (
	scrip_id varchar NULL,
	run_date date DEFAULT CURRENT_DATE NOT NULL,
	shares varchar NULL,
	value varchar NULL,
	"text" varchar NULL,
	insider varchar NULL,
	"position" varchar NULL,
	trans_date date NULL,
	ownership varchar NULL,
	created_at timestamp DEFAULT CURRENT_TIMESTAMP NOT NULL
);

 CREATE TABLE IF NOT EXISTS my_schema.system_flags (
                    flag_key VARCHAR(100) PRIMARY KEY,
                    value VARCHAR(10),
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

-- Accumulation/Distribution Analysis Tables
CREATE TABLE IF NOT EXISTS my_schema.accumulation_distribution (
    scrip_id VARCHAR(50) NOT NULL,
    analysis_date DATE NOT NULL,
    run_date DATE DEFAULT CURRENT_DATE,
    state VARCHAR(20) NOT NULL, -- 'ACCUMULATION', 'DISTRIBUTION', 'NEUTRAL'
    start_date DATE,
    days_in_state INTEGER,
    obv_value FLOAT,
    ad_value FLOAT,
    momentum_score FLOAT,
    pattern_detected VARCHAR(100),
    volume_analysis JSONB,
    confidence_score FLOAT,
    technical_context JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (scrip_id, analysis_date)
);

-- History table for state transitions
CREATE TABLE IF NOT EXISTS my_schema.accumulation_distribution_history (
    id SERIAL PRIMARY KEY,
    scrip_id VARCHAR(50) NOT NULL,
    state VARCHAR(20) NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE,
    duration_days INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for accumulation/distribution tables
CREATE INDEX IF NOT EXISTS idx_acc_dist_scrip_date ON my_schema.accumulation_distribution(scrip_id, analysis_date DESC);
CREATE INDEX IF NOT EXISTS idx_acc_dist_state ON my_schema.accumulation_distribution(state);
CREATE INDEX IF NOT EXISTS idx_acc_dist_history_scrip ON my_schema.accumulation_distribution_history(scrip_id, start_date DESC);

-- Supertrend values table (calculated once per day)
CREATE TABLE IF NOT EXISTS my_schema.supertrend_values (
    scrip_id VARCHAR(50) NOT NULL,
    calculation_date DATE NOT NULL,
    supertrend_value FLOAT,
    supertrend_direction INTEGER, -- -1 if price is below supertrend, 1 if price is above supertrend
    close_price FLOAT,
    days_below_supertrend INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (scrip_id, calculation_date)
);

-- Index for supertrend values table
CREATE INDEX IF NOT EXISTS idx_supertrend_scrip_date ON my_schema.supertrend_values(scrip_id, calculation_date DESC);

-- Stock price fetch error logging table
CREATE TABLE IF NOT EXISTS my_schema.stock_price_fetch_errors (
    id SERIAL PRIMARY KEY,
    scrip_id VARCHAR(50),
    error_type VARCHAR(100),
    error_message TEXT,
    function_name VARCHAR(200),
    stack_trace TEXT,
    fetch_date DATE DEFAULT CURRENT_DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for stock_price_fetch_errors table
CREATE INDEX IF NOT EXISTS idx_price_fetch_errors_scrip_id ON my_schema.stock_price_fetch_errors(scrip_id);
CREATE INDEX IF NOT EXISTS idx_price_fetch_errors_fetch_date ON my_schema.stock_price_fetch_errors(fetch_date DESC);
CREATE INDEX IF NOT EXISTS idx_price_fetch_errors_created_at ON my_schema.stock_price_fetch_errors(created_at DESC);