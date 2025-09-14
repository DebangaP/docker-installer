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