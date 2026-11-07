import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlbot.guard import UnsafeSQLError, validate_sql

TABLES = {"orders", "order_items", "customers", "payments", "refunds"}


@pytest.mark.parametrize(
    ("sql", "rule"),
    [
        ("DELETE FROM orders WHERE status = 'cancelled'", "select_only"),
        ("DROP TABLE orders", "select_only"),
        ("UPDATE orders SET status = 'paid'", "select_only"),
        ("SELECT 1; DROP TABLE orders", "single_statement"),
        ("SELECT * FROM pg_shadow", "allowed_tables"),
        ("SELECT * FROM information_schema.tables", "allowed_tables"),
        ("SELECT * FROM products", "allowed_tables"),
        ("SELECT pg_sleep(10)", "blocked_function"),
        ("SELECT * INTO x FROM orders", "no_select_into"),
        ("WITH d AS (DELETE FROM orders RETURNING *) SELECT count(*) FROM d", "select_only"),
    ],
)
def test_rejects_unsafe_sql(sql, rule):
    with pytest.raises(UnsafeSQLError, match=f"rule={rule}"):
        validate_sql(sql, TABLES)


def test_accepts_cte_select():
    sql = """
        WITH today AS (SELECT * FROM orders WHERE placed_at >= date_trunc('day', now()))
        SELECT count(*) FROM today
    """
    out = validate_sql(sql, TABLES)
    assert out.startswith("WITH today AS")
    assert out.endswith("LIMIT 1000")


def test_injects_limit_when_missing():
    assert validate_sql("SELECT id FROM orders", TABLES, max_limit=50).endswith("LIMIT 50")


def test_keeps_existing_smaller_limit():
    assert validate_sql("SELECT id FROM orders LIMIT 5", TABLES).endswith("LIMIT 5")


def test_caps_larger_limit():
    assert validate_sql("SELECT id FROM orders LIMIT 100000", TABLES).endswith("LIMIT 1000")


def test_union_of_selects_gets_outer_limit():
    out = validate_sql("SELECT id FROM orders UNION ALL SELECT id FROM refunds", TABLES)
    assert out.endswith("LIMIT 1000")
