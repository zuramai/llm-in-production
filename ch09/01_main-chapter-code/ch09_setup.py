"""Idempotent setup for the chapter's only table: `ch09_pending_refunds`.

`datasets/shopflow/seed.py --reset` drops the whole `public` schema, so this script is
safe to re-run any time (CREATE ... IF NOT EXISTS) and the notebook calls it first.

    uv run python ch09_setup.py            # create if missing
    uv run python ch09_setup.py --reset    # also delete all refund requests (demo cleanup)
"""

from __future__ import annotations

import argparse

from llmprod import connect

REASONS = ("damaged", "defective", "wrong_item", "not_delivered", "late", "changed_mind", "other")

DDL = f"""
CREATE TABLE IF NOT EXISTS ch09_pending_refunds (
    id                serial PRIMARY KEY,
    order_id          int         NOT NULL REFERENCES orders(id),
    amount_cents      int         NOT NULL CHECK (amount_cents > 0),
    reason            text        NOT NULL CHECK (reason IN ({", ".join(f"'{r}'" for r in REASONS)})),
    idempotency_key   text        NOT NULL UNIQUE,
    status            text        NOT NULL DEFAULT 'pending'
                                  CHECK (status IN ('pending', 'approved', 'rejected')),
    required_approver text        NOT NULL CHECK (required_approver IN ('agent', 'team_lead', 'finance')),
    requested_by      text        NOT NULL DEFAULT 'support-assistant',
    created_at        timestamptz NOT NULL DEFAULT now(),
    decided_at        timestamptz,
    decided_by_role   text,
    decision_note     text
);
-- Handbook rule: "refunds are idempotent per order + reason". Enforced for open requests, so a
-- model that invents a fresh idempotency key on retry still cannot create a second refund.
CREATE UNIQUE INDEX IF NOT EXISTS ch09_pending_refunds_open_order_reason
    ON ch09_pending_refunds (order_id, reason) WHERE status IN ('pending', 'approved');
CREATE INDEX IF NOT EXISTS ch09_pending_refunds_order ON ch09_pending_refunds (order_id);
GRANT SELECT ON ch09_pending_refunds TO shopflow_ro;
"""


def setup_tables() -> None:
    with connect() as conn:
        conn.execute(DDL)


def reset_refunds() -> None:
    with connect() as conn:
        conn.execute("TRUNCATE ch09_pending_refunds RESTART IDENTITY")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reset", action="store_true", help="delete all refund requests")
    args = parser.parse_args()
    setup_tables()
    if args.reset:
        reset_refunds()
    print("ch09_pending_refunds ready" + (" (emptied)" if args.reset else ""))
