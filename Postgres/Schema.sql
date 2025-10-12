CREATE SCHEMA my_schema;
CREATE TABLE my_schema.users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(255) UNIQUE
);
CREATE TABLE my_schema.orders (
    order_id SERIAL PRIMARY KEY,
    user_id INT REFERENCES my_schema.users(id),
    order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
-- Insert sample data
INSERT INTO my_schema.users (name, email) VALUES
('Alice', 'alice@example.com'),
('Bob', 'bob@example.com');
INSERT INTO my_schema.orders (user_id, order_date) VALUES
(1, CURRENT_TIMESTAMP),
(2, CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS my_schema.raw_ticks (
        timestamp TIMESTAMP,
        instrument_token INTEGER,
        ltp FLOAT,
        high FLOAT,
        low FLOAT,
        open FLOAT,
        close FLOAT,
        volume INTEGER
    );
    CREATE TABLE IF NOT EXISTS my_schema.bars (
        timestamp TIMESTAMP,
        instrument_token INTEGER,
        open FLOAT,
        high FLOAT,
        low FLOAT,
        close FLOAT,
        volume INTEGER,
        PRIMARY KEY (timestamp, instrument_token)
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
        PRIMARY KEY (session_date, instrument_token)
    );
    CREATE TABLE IF NOT EXISTS my_schema.sessions (
        session_date DATE,
        instrument_token INTEGER,
        overnight_high FLOAT,
        overnight_low FLOAT,
        PRIMARY KEY (session_date, instrument_token)
    );


CREATE TABLE my_schema.ticks (
    id SERIAL PRIMARY KEY,
    instrument_token BIGINT NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    last_price DECIMAL(15, 2),
    volume BIGINT,
    oi BIGINT,
    open DECIMAL(15, 2),
    high DECIMAL(15, 2),
    low DECIMAL(15, 2),
    close DECIMAL(15, 2)
);

CREATE TABLE my_schema.market_depth (
    id SERIAL PRIMARY KEY,
    tick_id INTEGER REFERENCES ticks(id),
    instrument_token BIGINT NOT NULL,
    side VARCHAR(4) NOT NULL CHECK (side IN ('buy', 'sell')),
    price DECIMAL(15, 2) NOT NULL,
    quantity INTEGER NOT NULL,
    orders INTEGER NOT NULL
);

CREATE INDEX idx_ticks_instrument_token ON my_schema.ticks(instrument_token);
CREATE INDEX idx_ticks_timestamp ON my_schema.ticks(timestamp);
CREATE INDEX idx_depth_tick_id ON my_schema.market_depth(tick_id);