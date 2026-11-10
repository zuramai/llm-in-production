"""Idempotent setup for Chapter 10: Postgres schema `ch10`, the refund ledger, and LangGraph's
checkpoint tables (created by `PostgresSaver.setup()` inside the same schema).

    uv run python ch10_setup.py            # create anything missing
    uv run python ch10_setup.py --reset    # also empty the ledger and all checkpoints
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from contextlib import contextmanager

import psycopg
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg.rows import dict_row

from llmprod import connect, get_settings

SCHEMA = "ch10"
REASONS = ("damaged", "defective", "wrong_item", "not_delivered", "late", "changed_mind", "other")

DDL = f"""
CREATE SCHEMA IF NOT EXISTS {SCHEMA};
-- Stand-in for the payment provider: a row here means money left the building.
CREATE TABLE IF NOT EXISTS {SCHEMA}.refund_ledger (
    id               serial PRIMARY KEY,
    order_id         int         NOT NULL REFERENCES public.orders(id),
    amount_cents     int         NOT NULL CHECK (amount_cents > 0),
    reason           text        NOT NULL CHECK (reason IN ({", ".join(f"'{r}'" for r in REASONS)})),
    -- Handbook: refunds are idempotent per order + reason.
    idempotency_key  text        NOT NULL UNIQUE,
    thread_id        text        NOT NULL,
    tool_call_id     text        NOT NULL,
    approved_by_role text,       -- NULL only when the ungated v0 agent wrote it
    approver_note    text,
    created_at       timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS refund_ledger_order ON {SCHEMA}.refund_ledger (order_id);
GRANT USAGE ON SCHEMA {SCHEMA} TO shopflow_ro;
GRANT SELECT ON {SCHEMA}.refund_ledger TO shopflow_ro;
"""


def checkpoint_conn() -> psycopg.Connection:
    """A connection for PostgresSaver: autocommit + dict rows (both required by the library),
    with `search_path` pinned to our schema so its unqualified table names land in `ch10`."""
    return psycopg.connect(
        get_settings().database_url,
        autocommit=True,
        prepare_threshold=0,
        row_factory=dict_row,
        options=f"-c search_path={SCHEMA}",
    )


@contextmanager
def checkpointer() -> Iterator[PostgresSaver]:
    with checkpoint_conn() as conn:
        yield PostgresSaver(conn)


def setup() -> None:
    with connect() as conn:
        conn.execute(DDL)
    with checkpointer() as saver:
        saver.setup()  # runs LangGraph's own migrations, idempotently


def reset() -> None:
    with connect() as conn:
        conn.execute(f"TRUNCATE {SCHEMA}.refund_ledger RESTART IDENTITY")
        conn.execute(
            f"TRUNCATE {SCHEMA}.checkpoints, {SCHEMA}.checkpoint_blobs, {SCHEMA}.checkpoint_writes"
        )


def clear_orders(order_ids: list[int]) -> None:
    """Delete ledger rows for some orders (the eval resets only the orders it uses)."""
    with connect() as conn:
        conn.execute(f"DELETE FROM {SCHEMA}.refund_ledger WHERE order_id = ANY(%s)", (order_ids,))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reset", action="store_true", help="empty ledger and checkpoints")
    args = parser.parse_args()
    setup()
    if args.reset:
        reset()
    print(f"schema {SCHEMA} ready" + (" (emptied)" if args.reset else ""))
