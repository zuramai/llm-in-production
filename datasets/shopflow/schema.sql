-- ShopFlow schema. Idempotent: drops and recreates everything in `public`.
-- Conventions (deliberate, and a classic LLM pitfall):
--   * all money is INTEGER CENTS (divide by 100 for USD)
--   * all timestamps are timestamptz, stored in UTC; the business runs on shop_settings.timezone

DROP SCHEMA IF EXISTS public CASCADE;
CREATE SCHEMA public;
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE categories (
    id   serial PRIMARY KEY,
    name text NOT NULL UNIQUE
);

CREATE TABLE products (
    id          serial PRIMARY KEY,
    sku         text NOT NULL UNIQUE,
    name        text NOT NULL,
    category_id int  NOT NULL REFERENCES categories(id),
    price_cents int  NOT NULL,
    cost_cents  int  NOT NULL,
    active      bool NOT NULL DEFAULT true
);

CREATE TABLE customers (
    id         serial PRIMARY KEY,
    email      text NOT NULL UNIQUE,
    full_name  text NOT NULL,
    country    text NOT NULL,
    city       text NOT NULL,
    segment    text NOT NULL CHECK (segment IN ('consumer', 'business', 'vip')),
    created_at timestamptz NOT NULL
);

CREATE TABLE orders (
    id             serial PRIMARY KEY,
    customer_id    int  NOT NULL REFERENCES customers(id),
    status         text NOT NULL CHECK (status IN ('pending', 'paid', 'shipped', 'delivered',
                                                   'cancelled', 'refunded')),
    channel        text NOT NULL CHECK (channel IN ('web', 'ios', 'android')),
    currency       text NOT NULL DEFAULT 'USD',
    subtotal_cents int  NOT NULL,
    discount_cents int  NOT NULL,
    shipping_cents int  NOT NULL,
    total_cents    int  NOT NULL,
    placed_at      timestamptz NOT NULL
);

CREATE TABLE order_items (
    id               serial PRIMARY KEY,
    order_id         int NOT NULL REFERENCES orders(id),
    product_id       int NOT NULL REFERENCES products(id),
    quantity         int NOT NULL,
    unit_price_cents int NOT NULL
);

CREATE TABLE payments (
    id           serial PRIMARY KEY,
    order_id     int  NOT NULL REFERENCES orders(id),
    provider     text NOT NULL CHECK (provider IN ('stripe', 'paypal')),
    status       text NOT NULL CHECK (status IN ('succeeded', 'failed')),
    amount_cents int  NOT NULL,
    created_at   timestamptz NOT NULL
);

CREATE TABLE refunds (
    id           serial PRIMARY KEY,
    order_id     int  NOT NULL REFERENCES orders(id),
    amount_cents int  NOT NULL,
    reason       text NOT NULL,
    created_at   timestamptz NOT NULL
);

CREATE TABLE shipments (
    id           serial PRIMARY KEY,
    order_id     int  NOT NULL REFERENCES orders(id),
    carrier      text NOT NULL,
    status       text NOT NULL,
    shipped_at   timestamptz,
    delivered_at timestamptz
);

CREATE TABLE inventory (
    product_id int  NOT NULL REFERENCES products(id),
    warehouse  text NOT NULL,
    on_hand    int  NOT NULL,
    PRIMARY KEY (product_id, warehouse)
);

CREATE TABLE support_tickets (
    id             serial PRIMARY KEY,
    customer_id    int  NOT NULL REFERENCES customers(id),
    order_id       int  REFERENCES orders(id),
    channel        text NOT NULL,
    subject        text NOT NULL,
    body           text NOT NULL,
    created_at     timestamptz NOT NULL,
    label_category text NOT NULL,
    label_priority text NOT NULL
);

CREATE TABLE shop_settings (
    key   text PRIMARY KEY,
    value text NOT NULL
);

CREATE INDEX ON products (category_id);
CREATE INDEX ON orders (customer_id);
CREATE INDEX ON orders (placed_at);
CREATE INDEX ON order_items (order_id);
CREATE INDEX ON order_items (product_id);
CREATE INDEX ON payments (order_id);
CREATE INDEX ON refunds (order_id);
CREATE INDEX ON shipments (order_id);
CREATE INDEX ON inventory (product_id);
CREATE INDEX ON support_tickets (customer_id);
CREATE INDEX ON support_tickets (order_id);

-- Read-only role for LLM-generated queries.
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'shopflow_ro') THEN
        CREATE ROLE shopflow_ro LOGIN PASSWORD 'shopflow_ro';
    END IF;
END
$$;

DO $$
BEGIN
    EXECUTE format('GRANT CONNECT ON DATABASE %I TO shopflow_ro', current_database());
END
$$;
GRANT USAGE ON SCHEMA public TO shopflow_ro;
GRANT SELECT ON categories, products, orders, order_items, payments, refunds, shipments,
                inventory, support_tickets, shop_settings TO shopflow_ro;
-- PII: email is deliberately NOT readable by the read-only role.
GRANT SELECT (id, full_name, country, city, segment, created_at) ON customers TO shopflow_ro;
ALTER ROLE shopflow_ro SET default_transaction_read_only = on;
ALTER ROLE shopflow_ro SET statement_timeout = '5s';
