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