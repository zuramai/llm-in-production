"""Refund domain logic: approval limits and idempotent refund *requests*.

Copied from ch09/01_main-chapter-code/ch09_refunds.py (request side only; the human approval
functions aren't needed here). Requests go to the chapter's own `ch13_pending_refunds` table,
so the attack suite can count what an injected agent managed to queue. The model only proposes.
Limits come from datasets/shopflow/docs/handbook/refund-approval-limits.md.
"""

from __future__ import annotations

from typing import Any

from psycopg.rows import dict_row

from llmprod import connect

AGENT_LIMIT_CENTS = 10_000  # up to $100.00: support agent
LEAD_LIMIT_CENTS = 50_000  # $100.01 to $500.00: team lead; above: Finance
ROLE_RANK = {"agent": 1, "team_lead": 2, "finance": 3}
REFUNDABLE_STATUSES = {"paid", "shipped", "delivered"}


class RefundError(ValueError):
    """An anticipated failure whose message is safe, and useful, to show to the model."""


def usd(cents: int) -> str:
    return f"${cents / 100:,.2f}"


def required_approver(combined_cents: int) -> str:
    """Who must approve a refund whose per-order total is `combined_cents`."""
    if combined_cents <= AGENT_LIMIT_CENTS:
        return "agent"
    if combined_cents <= LEAD_LIMIT_CENTS:
        return "team_lead"
    return "finance"


def evaluate_refund(
    *,
    status: str,
    total_cents: int,
    refunded_cents: int,
    open_cents: int,
    amount_cents: int,
) -> str:
    """Validate a refund request against the order and return the required approver.

    `refunded_cents` is money already refunded; `open_cents` is money in pending/approved
    requests. The limit applies to the combined per-order total ("do not split a refund").
    """
    if amount_cents <= 0:
        raise RefundError("amount_cents must be a positive integer number of cents")
    if status not in REFUNDABLE_STATUSES:
        raise RefundError(
            f"order status is '{status}'; only paid, shipped or delivered orders can be refunded"
        )
    combined = refunded_cents + open_cents + amount_cents
    if combined > total_cents:
        room = max(total_cents - refunded_cents - open_cents, 0)
        raise RefundError(
            f"refund would exceed the order total: order total is {total_cents} cents, "
            f"{refunded_cents + open_cents} cents are already refunded or pending, "
            f"so at most {room} cents can still be requested"
        )
    return required_approver(combined)


def _public(row: dict[str, Any], *, replay: bool) -> dict[str, Any]:
    status = row["status"]
    if status == "pending":
        msg = f"Submitted for approval by {row['required_approver']}. NOT refunded yet."
    elif status == "approved":
        msg = "Already approved."
    else:
        msg = "This request was rejected."
    return {
        "refund_request_id": row["id"],
        "order_id": row["order_id"],
        "amount_cents": row["amount_cents"],
        "amount_usd": usd(row["amount_cents"]),
        "reason": row["reason"],
        "status": status,
        "required_approver": row["required_approver"],
        "idempotent_replay": replay,
        "message": msg,
    }


def request_refund(
    order_id: int, amount_cents: int, reason: str, idempotency_key: str
) -> dict[str, Any]:
    """Queue a refund for human approval. Safe to call again with the same key."""
    if not idempotency_key or len(idempotency_key) > 120:
        raise RefundError("idempotency_key must be 1-120 characters")
    with connect() as conn, conn.transaction(), conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT id, status, total_cents FROM orders WHERE id = %s FOR UPDATE", (order_id,)
        )
        order = cur.fetchone()
        if order is None:
            raise RefundError(f"order {order_id} not found")

        # 1) Same key: replay the stored result, but refuse a different payload (Stripe's rule).
        cur.execute(
            "SELECT * FROM ch13_pending_refunds WHERE idempotency_key = %s", (idempotency_key,)
        )
        prior = cur.fetchone()
        if prior is not None:
            same = (prior["order_id"], prior["amount_cents"], prior["reason"]) == (
                order_id,
                amount_cents,
                reason,
            )
            if not same:
                raise RefundError(
                    "idempotency_key was already used with different parameters "
                    f"(order {prior['order_id']}, {prior['amount_cents']} cents, "
                    f"reason '{prior['reason']}'); use a new key for a different refund"
                )
            return _public(prior, replay=True)

        # 2) Handbook rule: idempotent per order + reason, even if the key differs.
        cur.execute(
            "SELECT * FROM ch13_pending_refunds "
            "WHERE order_id = %s AND reason = %s AND status IN ('pending', 'approved')",
            (order_id, reason),
        )
        open_same = cur.fetchone()
        if open_same is not None:
            return _public(open_same, replay=True)

        cur.execute(
            "SELECT coalesce(sum(amount_cents), 0) AS s FROM refunds WHERE order_id = %s",
            (order_id,),
        )
        refunded = int(cur.fetchone()["s"])
        cur.execute(
            "SELECT coalesce(sum(amount_cents), 0) AS s FROM ch13_pending_refunds "
            "WHERE order_id = %s AND status IN ('pending', 'approved')",
            (order_id,),
        )
        open_cents = int(cur.fetchone()["s"])

        approver = evaluate_refund(
            status=order["status"],
            total_cents=order["total_cents"],
            refunded_cents=refunded,
            open_cents=open_cents,
            amount_cents=amount_cents,
        )
        cur.execute(
            "INSERT INTO ch13_pending_refunds "
            "(order_id, amount_cents, reason, idempotency_key, required_approver) "
            "VALUES (%s, %s, %s, %s, %s) RETURNING *",
            (order_id, amount_cents, reason, idempotency_key, approver),
        )
        return _public(cur.fetchone(), replay=False)
