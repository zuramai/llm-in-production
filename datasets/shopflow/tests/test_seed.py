"""Invariants of the ShopFlow seed.

These tests re-run `seed.py --reset`, which drops the whole `public` schema, including tables
that chapters create (e.g. `ch04_doc_chunks`). They are excluded from a plain `uv run pytest`;
run them on purpose with `uv run pytest -m reseed datasets`, then re-run any chapter setup.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import psycopg
import pytest
from dotenv import find_dotenv, load_dotenv

pytestmark = [pytest.mark.docker, pytest.mark.reseed]

SEED = Path(__file__).resolve().parents[1] / "seed.py"
TODAY_ORDERS = 37  # deliberately not the default, to prove the flag is honoured
TZ = "America/New_York"

load_dotenv(find_dotenv(usecwd=True))
ADMIN_URL = os.environ.get("DATABASE_URL", "postgresql://shopflow:shopflow@localhost:5433/shopflow")
RO_URL = os.environ.get(
    "DATABASE_URL_RO", "postgresql://shopflow_ro:shopflow_ro@localhost:5433/shopflow"
)


def run_seed(*extra: str) -> None:
    subprocess.run(
        [sys.executable, str(SEED), "--reset", "--seed", "42", "--timezone", TZ, *extra],
        check=True,
        capture_output=True,
    )


def connect_in_tz() -> psycopg.Connection:
    conn = psycopg.connect(ADMIN_URL)
    conn.execute("SELECT set_config('TimeZone', %s, false)", (TZ,))
    return conn


def historical_sum() -> int:
    with connect_in_tz() as conn:
        return conn.execute(
            "SELECT sum(total_cents) FROM orders WHERE placed_at < date_trunc('day', now())"
        ).fetchone()[0]


@pytest.fixture(scope="module", autouse=True)
def restore_default_seed():
    yield
    run_seed()  # leave the DB in the state every chapter expects


def test_today_order_count_matches_flag_in_business_tz():
    run_seed("--today-orders", str(TODAY_ORDERS))
    with connect_in_tz() as conn:
        n = conn.execute(
            "SELECT count(*) FROM orders WHERE placed_at >= date_trunc('day', now())"
        ).fetchone()[0]
    assert n == TODAY_ORDERS


def test_same_seed_is_deterministic():
    run_seed()
    first = historical_sum()
    run_seed()
    assert historical_sum() == first


def test_readonly_role_cannot_read_email():
    with psycopg.connect(RO_URL) as conn, pytest.raises(
        psycopg.errors.InsufficientPrivilege, match="permission denied"
    ):
        conn.execute("SELECT email FROM customers LIMIT 1")


def test_readonly_role_cannot_write_even_after_disabling_readonly_default():
    with psycopg.connect(RO_URL) as conn:
        # Layer 1: the role's sessions default to read-only transactions...
        with pytest.raises(psycopg.errors.ReadOnlySqlTransaction):
            conn.execute("DELETE FROM orders WHERE id = 1")
        conn.rollback()
        # ...but any session can switch that default off. Layer 2, the missing grant, holds.
        conn.execute("SET default_transaction_read_only = off")
        conn.commit()
        with pytest.raises(psycopg.errors.InsufficientPrivilege, match="permission denied"):
            conn.execute("DELETE FROM orders WHERE id = 1")
