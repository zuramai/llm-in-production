"""The four ShopFlow support tools, plus the registry that exposes them to an LLM.

Each tool is a plain, typed Python function. Two things are derived from that one definition:
  * the OpenAI `tools=[...]` JSON schema (raw SDK loop), and
  * the MCP tool schema (`ch09_mcp_server.py` registers the very same functions).

`Toolbox.call` never raises: every failure becomes a JSON `{"error": ...}` string that goes back
to the model as the tool message, so the model can read it and recover.
"""

from __future__ import annotations

import inspect
import json
import re
from collections.abc import Callable
from typing import Annotated, Any, Literal, Protocol, get_type_hints

import ch09_refunds
from psycopg.rows import dict_row
from pydantic import ConfigDict, Field, ValidationError, create_model

from llmprod import connect

MAX_RESULT_CHARS = 6000
EMAIL_RE = re.compile(r"^[^@\s]{1,64}@[^@\s]{1,255}\.[^@\s]{2,}$")

RefundReason = Literal[
    "damaged", "defective", "wrong_item", "not_delivered", "late", "changed_mind", "other"
]


class ToolError(Exception):
    """A failure the model can act on. The message is returned to it verbatim."""


# --------------------------------------------------------------------------- tools


def get_order(
    order_id: Annotated[int, Field(gt=0, description="ShopFlow order id, a positive integer")],
) -> dict[str, Any]:
    """Look up one order: status, totals (integer cents), items with SKUs, payments, refunds,
    shipment dates and any pending refund requests. Does not return the customer's email."""
    with connect(readonly=True) as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT o.id, o.status, o.channel, o.currency, o.subtotal_cents, o.discount_cents, "
            "o.shipping_cents, o.total_cents, o.placed_at, c.id AS customer_id, "
            "c.full_name AS customer_name, c.segment AS customer_segment "
            "FROM orders o JOIN customers c ON c.id = o.customer_id WHERE o.id = %s",
            (order_id,),
        )
        order = cur.fetchone()
        if order is None:
            raise ToolError(f"order {order_id} not found")
        cur.execute(
            "SELECT p.sku, p.name, i.quantity, i.unit_price_cents FROM order_items i "
            "JOIN products p ON p.id = i.product_id WHERE i.order_id = %s ORDER BY i.id",
            (order_id,),
        )
        order["items"] = cur.fetchall()
        cur.execute(
            "SELECT provider, status, amount_cents FROM payments WHERE order_id = %s ORDER BY id",
            (order_id,),
        )
        order["payments"] = cur.fetchall()
        cur.execute(
            "SELECT amount_cents, reason, created_at FROM refunds WHERE order_id = %s ORDER BY id",
            (order_id,),
        )
        order["refunds"] = cur.fetchall()
        cur.execute(
            "SELECT carrier, status, shipped_at, delivered_at FROM shipments WHERE order_id = %s",
            (order_id,),
        )
        order["shipments"] = cur.fetchall()
        cur.execute(
            "SELECT id AS refund_request_id, amount_cents, reason, status, required_approver "
            "FROM ch09_pending_refunds WHERE order_id = %s ORDER BY id",
            (order_id,),
        )
        order["refund_requests"] = cur.fetchall()
        return order


def search_orders(
    customer_email: Annotated[str, Field(description="The customer's exact email address")],
) -> dict[str, Any]:
    """List a customer's 10 most recent orders (id, status, total in cents, placed_at)
    from their email address. Returns order ids only; never echoes personal data."""
    if not EMAIL_RE.match(customer_email):
        raise ToolError("customer_email does not look like an email address")
    # Why the ADMIN connection: shopflow_ro cannot read customers.email at all, by design.
    # The privilege is held by this function, not by the model. The query is fixed text with one
    # bound parameter, and it only ever returns order columns, never the email itself.
    with connect() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT o.id AS order_id, o.status, o.total_cents, o.placed_at FROM orders o "
            "JOIN customers c ON c.id = o.customer_id WHERE lower(c.email) = lower(%s) "
            "ORDER BY o.placed_at DESC LIMIT 10",
            (customer_email,),
        )
        rows = cur.fetchall()
    if not rows:
        raise ToolError("no customer with that email address, or the customer has no orders")
    return {"orders": rows}


def check_inventory(
    sku: Annotated[str, Field(description="Product SKU such as SF-ELE-0001")],
) -> dict[str, Any]:
    """Stock on hand per warehouse for one product SKU, plus its price in cents and whether it
    is still sold (active)."""
    with connect(readonly=True) as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT id, sku, name, price_cents, active FROM products WHERE sku = %s",
            (sku.strip().upper(),),
        )
        product = cur.fetchone()
        if product is None:
            raise ToolError(f"unknown SKU '{sku}'. SKUs look like SF-ELE-0001")
        cur.execute(
            "SELECT warehouse, on_hand FROM inventory WHERE product_id = %s ORDER BY warehouse",
            (product.pop("id"),),
        )
        product["warehouses"] = cur.fetchall()
        product["total_on_hand"] = sum(w["on_hand"] for w in product["warehouses"])
        return product


def create_refund(
    order_id: Annotated[int, Field(gt=0, description="Order to refund")],
    amount_cents: Annotated[
        int, Field(gt=0, description="Refund amount in integer CENTS (e.g. $12.50 = 1250)")
    ],
    reason: Annotated[RefundReason, Field(description="Reason code from the allowed list")],
    idempotency_key: Annotated[
        str,
        Field(
            min_length=1,
            max_length=120,
            description="Stable key for this logical refund: 'refund-<order_id>-<reason>'. "
            "Reuse the SAME key when retrying; never invent a new one for the same refund.",
        ),
    ],
) -> dict[str, Any]:
    """PROPOSE a refund. This does not move money: it queues a request that a human must
    approve (agent up to $100.00, team lead up to $500.00, Finance above). Safe to retry
    with the same idempotency_key."""
    try:
        return ch09_refunds.request_refund(order_id, amount_cents, reason, idempotency_key)
    except ch09_refunds.RefundError as exc:
        raise ToolError(str(exc)) from exc


TOOL_FUNCTIONS: dict[str, Callable[..., dict[str, Any]]] = {
    f.__name__: f for f in (get_order, search_orders, check_inventory, create_refund)
}
# MCP `annotations`: honest hints so a client can decide what needs a confirmation click.
TOOL_HINTS = {
    "get_order": {"read_only_hint": True},
    "search_orders": {"read_only_hint": True},
    "check_inventory": {"read_only_hint": True},
    "create_refund": {"read_only_hint": False, "destructive_hint": False, "idempotent_hint": True},
}


# --------------------------------------------------------------------------- registry


def _args_model(fn: Callable[..., Any]):
    hints = get_type_hints(fn, include_extras=True)
    fields = {
        name: (hints[name], ... if p.default is inspect.Parameter.empty else p.default)
        for name, p in inspect.signature(fn).parameters.items()
    }
    return create_model(f"{fn.__name__}_args", __config__=ConfigDict(extra="forbid"), **fields)


def _strip_titles(node: Any) -> Any:
    if isinstance(node, dict):
        return {k: _strip_titles(v) for k, v in node.items() if k != "title"}
    if isinstance(node, list):
        return [_strip_titles(v) for v in node]
    return node


def openai_tool_spec(fn: Callable[..., Any]) -> dict[str, Any]:
    schema = _strip_titles(_args_model(fn).model_json_schema())
    schema["additionalProperties"] = False
    return {
        "type": "function",
        "function": {
            "name": fn.__name__,
            "description": " ".join((fn.__doc__ or "").split()),
            "parameters": schema,
        },
    }


class Toolbox(Protocol):
    specs: list[dict[str, Any]]

    def call(self, name: str, raw_arguments: str) -> str: ...


def _clip(text: str) -> str:
    if len(text) <= MAX_RESULT_CHARS:
        return text
    return text[:MAX_RESULT_CHARS] + f"... [truncated, {len(text)} chars total]"


def error_json(message: str) -> str:
    return json.dumps({"error": message})


class LocalToolbox:
    """Runs tool functions in-process. `call` returns a JSON string and NEVER raises."""

    def __init__(self, functions: dict[str, Callable[..., Any]] | None = None) -> None:
        self.functions = dict(TOOL_FUNCTIONS if functions is None else functions)
        self.specs = [openai_tool_spec(f) for f in self.functions.values()]
        self._models = {name: _args_model(f) for name, f in self.functions.items()}

    def call(self, name: str, raw_arguments: str) -> str:
        fn = self.functions.get(name)
        if fn is None:
            return error_json(f"unknown tool '{name}'. Available tools: {sorted(self.functions)}")
        try:
            args = json.loads(raw_arguments or "{}")
        except json.JSONDecodeError as exc:
            return error_json(f"arguments are not valid JSON: {exc.msg} (position {exc.pos})")
        if not isinstance(args, dict):
            return error_json("arguments must be a JSON object")
        try:
            parsed = self._models[name].model_validate(args)
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(str(p) for p in e['loc']) or 'arguments'}: {e['msg']}"
                for e in exc.errors()
            )
            return error_json(f"invalid arguments for {name}: {problems}")
        try:
            result = fn(**{k: getattr(parsed, k) for k in type(parsed).model_fields})
        except ToolError as exc:
            return error_json(str(exc))
        except Exception as exc:  # noqa: BLE001 - a crashing tool must not crash the agent
            return error_json(f"{name} failed unexpectedly ({type(exc).__name__}); try again later")
        return _clip(json.dumps(result, default=str))
