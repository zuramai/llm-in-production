"""Chapter 10 tools: Chapter 9's read-only tools, one long-output tool, and the one tool with
a side effect (`issue_refund`), which the graph never runs without a human approval.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Annotated, Any

from psycopg.rows import dict_row
from pydantic import Field

from llmprod import connect


def _load_previous() -> Any:
    name = "ch10_previous_chapters"  # unique: other chapters also ship a previous_chapters.py
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, Path(__file__).with_name("previous_chapters.py")
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


prev = _load_previous()
ToolError = prev.ToolError
RefundError = prev.RefundError
RefundReason = prev.RefundReason
LocalToolbox = prev.LocalToolbox
openai_tool_spec = prev.openai_tool_spec
error_json = prev.error_json
usd = prev.usd
ROLE_RANK = prev.ROLE_RANK
search_orders = prev.search_orders
check_inventory = prev.check_inventory


def idempotency_key(order_id: int, reason: str) -> str:
    """Handbook rule: one refund per order + reason."""
    return f"refund-{order_id}-{reason}"


# ------------------------------------------------------------------------- read tools


def get_order(
    order_id: Annotated[int, Field(gt=0, description="ShopFlow order id, a positive integer")],
) -> dict[str, Any]:
    """Look up one order: status, totals (integer cents), items with SKUs, payments, refunds
    (including refunds issued by this assistant) and shipment dates."""
    order = prev.get_order(order_id)
    with connect(readonly=True) as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT amount_cents, reason, approved_by_role, created_at FROM ch10.refund_ledger "
            "WHERE order_id = %s ORDER BY id",
            (order_id,),
        )
        order["assistant_refunds"] = cur.fetchall()
    return order


def get_customer_history(
    customer_id: Annotated[int, Field(gt=0, description="Customer id (from get_order)")],
) -> dict[str, Any]:
    """Full history of one customer: a summary (order count, refunds, recent refund claims,
    tickets) followed by EVERY order with items and refunds and every support ticket with its
    text. Use it to check fraud flags such as repeated refund claims. Output can be long."""
    with connect(readonly=True) as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT id, full_name, segment, country, created_at FROM customers WHERE id = %s",
            (customer_id,),
        )
        customer = cur.fetchone()
        if customer is None:
            raise ToolError(f"customer {customer_id} not found")
        cur.execute(
            "SELECT o.id, o.status, o.total_cents, o.placed_at, "
            "coalesce((SELECT json_agg(json_build_object('sku', p.sku, 'name', p.name, "
            "  'qty', i.quantity) ORDER BY i.id) FROM order_items i "
            "  JOIN products p ON p.id = i.product_id WHERE i.order_id = o.id), '[]') AS items, "
            "coalesce((SELECT json_agg(json_build_object('amount_cents', r.amount_cents, "
            "  'reason', r.reason, 'created_at', r.created_at) ORDER BY r.id) "
            "  FROM refunds r WHERE r.order_id = o.id), '[]') AS refunds "
            "FROM orders o WHERE o.customer_id = %s ORDER BY o.placed_at DESC",
            (customer_id,),
        )
        orders = cur.fetchall()
        cur.execute(
            "SELECT id, order_id, channel, subject, body, label_category AS category, created_at "
            "FROM support_tickets WHERE customer_id = %s ORDER BY created_at DESC",
            (customer_id,),
        )
        tickets = cur.fetchall()
        cur.execute(
            "SELECT count(*) AS n, coalesce(sum(r.amount_cents), 0) AS cents, "
            "count(*) FILTER (WHERE r.created_at > now() - interval '90 days') AS n_90d "
            "FROM refunds r JOIN orders o ON o.id = r.order_id WHERE o.customer_id = %s",
            (customer_id,),
        )
        ref = cur.fetchone()
    by_category: dict[str, int] = {}
    for t in tickets:
        by_category[t["category"]] = by_category.get(t["category"], 0) + 1
    summary = {
        "orders": len(orders),
        "spent_cents": sum(o["total_cents"] for o in orders if o["status"] != "cancelled"),
        "refunds": ref["n"],
        "refunded_cents": int(ref["cents"]),
        "refunds_last_90_days": ref["n_90d"],
        "tickets": len(tickets),
        "tickets_by_category": by_category,
    }
    return {"customer": customer, "summary": summary, "orders": orders, "tickets": tickets}


READ_TOOLS = {
    f.__name__: f for f in (get_order, search_orders, check_inventory, get_customer_history)
}


# ------------------------------------------------------------------- the side effect


def issue_refund(
    order_id: Annotated[int, Field(gt=0, description="Order to refund")],
    amount_cents: Annotated[
        int, Field(gt=0, description="Refund amount in integer CENTS (e.g. $12.50 = 1250)")
    ],
    reason: Annotated[RefundReason, Field(description="Reason code from the allowed list")],
) -> dict[str, Any]:
    """Issue a refund to the original payment method. MOVES MONEY. The run pauses until a
    human with enough authority approves (agent up to $100.00, team lead up to $500.00,
    Finance above); you get the decision back as this tool's result."""
    raise NotImplementedError("the graph's approval node runs refunds, never the toolbox")


SIDE_EFFECT_TOOLS = {"issue_refund": issue_refund}


FRAUD_RECENT_REFUNDS = 3  # handbook: "multiple refunds on one account in a short period"


def approver_for(
    *, status: str, total_cents: int, refunded_cents: int, amount_cents: int, recent_refunds: int
) -> tuple[str, list[str]]:
    """Required approver plus fraud flags. The handbook's fraud rule lives here, in code: a
    flagged refund always needs at least a team lead, whatever the model thought of it."""
    approver = prev.evaluate_refund(
        status=status,
        total_cents=total_cents,
        refunded_cents=refunded_cents,
        open_cents=0,
        amount_cents=amount_cents,
    )
    flags = []
    if recent_refunds >= FRAUD_RECENT_REFUNDS:
        flags.append(f"customer has {recent_refunds} refunds in the last 90 days")
        if ROLE_RANK[approver] < ROLE_RANK["team_lead"]:
            approver = "team_lead"
    return approver, flags


def _order_money(cur: Any, order_id: int, *, lock: bool) -> dict[str, Any]:
    cur.execute(
        "SELECT id, customer_id, status, total_cents FROM orders WHERE id = %s"
        + (" FOR UPDATE" if lock else ""),
        (order_id,),
    )
    order = cur.fetchone()
    if order is None:
        raise RefundError(f"order {order_id} not found")
    cur.execute(
        "SELECT (SELECT coalesce(sum(amount_cents), 0) FROM refunds WHERE order_id = %s) "
        "+ (SELECT coalesce(sum(amount_cents), 0) FROM ch10.refund_ledger WHERE order_id = %s) "
        "AS refunded",
        (order_id, order_id),
    )
    order["refunded_cents"] = int(cur.fetchone()["refunded"])
    cur.execute(
        "SELECT (SELECT count(*) FROM refunds r JOIN orders o ON o.id = r.order_id "
        "        WHERE o.customer_id = %(c)s AND r.created_at > now() - interval '90 days') "
        "+ (SELECT count(*) FROM ch10.refund_ledger l JOIN orders o ON o.id = l.order_id "
        "        WHERE o.customer_id = %(c)s AND l.created_at > now() - interval '90 days') "
        "AS recent",
        {"c": order["customer_id"]},
    )
    order["recent_refunds"] = int(cur.fetchone()["recent"])
    return order


def _existing(cur: Any, key: str) -> dict[str, Any] | None:
    cur.execute("SELECT * FROM ch10.refund_ledger WHERE idempotency_key = %s", (key,))
    return cur.fetchone()


def validate_refund(order_id: int, amount_cents: int, reason: str) -> dict[str, Any]:
    """Check a proposed refund BEFORE asking a human. Raises RefundError for anything a
    human should never be asked to approve. Returns the proposal shown to the approver."""
    key = idempotency_key(order_id, reason)
    with connect() as conn, conn.cursor(row_factory=dict_row) as cur:
        order = _order_money(cur, order_id, lock=False)
        prior = _existing(cur, key)
    if prior is not None:
        raise RefundError(
            f"a '{reason}' refund of {usd(prior['amount_cents'])} was already issued for order "
            f"{order_id} (refunds are idempotent per order + reason); nothing to do"
        )
    approver, flags = approver_for(
        status=order["status"],
        total_cents=order["total_cents"],
        refunded_cents=order["refunded_cents"],
        amount_cents=amount_cents,
        recent_refunds=order["recent_refunds"],
    )
    return {
        "order_id": order_id,
        "amount_cents": amount_cents,
        "amount_usd": usd(amount_cents),
        "reason": reason,
        "order_status": order["status"],
        "order_total_usd": usd(order["total_cents"]),
        "already_refunded_usd": usd(order["refunded_cents"]),
        "fraud_flags": flags,
        "required_approver": approver,
        "idempotency_key": key,
    }


def execute_refund(
    proposal: dict[str, Any],
    *,
    thread_id: str,
    tool_call_id: str,
    role: str | None,
    note: str = "",
) -> dict[str, Any]:
    """Write the refund to the ledger. Idempotent: running it twice for the same order + reason
    (a retried node, a resumed crash) returns the first row instead of paying twice. Limits are
    re-checked under a row lock, because money may have moved since the human said yes."""
    key = proposal["idempotency_key"]
    with connect() as conn, conn.transaction(), conn.cursor(row_factory=dict_row) as cur:
        order = _order_money(cur, proposal["order_id"], lock=True)
        prior = _existing(cur, key)
        if prior is not None:
            return {"status": "already_issued", "refund_id": prior["id"],
                    "amount_usd": usd(prior["amount_cents"]), "idempotent_replay": True}  # fmt: skip
        needed, _ = approver_for(
            status=order["status"],
            total_cents=order["total_cents"],
            refunded_cents=order["refunded_cents"],
            amount_cents=proposal["amount_cents"],
            recent_refunds=order["recent_refunds"],
        )
        if role is not None and ROLE_RANK[role] < ROLE_RANK[needed]:
            raise RefundError(f"approval by '{role}' is no longer enough: needs '{needed}'")
        cur.execute(
            "INSERT INTO ch10.refund_ledger (order_id, amount_cents, reason, idempotency_key, "
            "thread_id, tool_call_id, approved_by_role, approver_note) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING id",
            (proposal["order_id"], proposal["amount_cents"], proposal["reason"], key,
             thread_id, tool_call_id, role, note),
        )  # fmt: skip
        refund_id = cur.fetchone()["id"]
    return {
        "status": "issued",
        "refund_id": refund_id,
        "amount_usd": usd(proposal["amount_cents"]),
        "approved_by": role,
        "message": "Refund issued to the original payment method; 5-7 business days to arrive.",
    }


def ledger_rows(order_ids: list[int]) -> list[dict[str, Any]]:
    with connect() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT * FROM ch10.refund_ledger WHERE order_id = ANY(%s) ORDER BY id", (order_ids,)
        )
        return cur.fetchall()
