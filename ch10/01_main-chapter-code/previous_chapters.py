"""Code copied from Chapter 9, so this chapter stands alone.

Sources: `ch09/01_main-chapter-code/ch09_tools.py` (the read-only support tools and the
tool registry) and `ch09_refunds.py` (the approval limits). Three changes from Chapter 9:
  * `get_order` no longer reads Chapter 9's `ch09_pending_refunds` table; Chapter 10 adds its
    own refund ledger on top (see `ch10_tools.get_order`);
  * `create_refund` and the request queue are gone: in Chapter 10 the graph itself pauses for
    approval, so the queue table is replaced by a checkpoint;
  * `LocalToolbox` takes its functions explicitly, gains `parse()` (argument validation on its
    own, used to check `issue_refund` calls without running them) and a `clip` switch.

`ch10_tools.py` loads this file under the unique module name `ch10_previous_chapters`, so it
never collides with another chapter's `previous_chapters` in one pytest process.
"""

from __future__ import annotations

import inspect
import json
import re
from collections.abc import Callable
from typing import Annotated, Any, Literal, get_type_hints

from psycopg.rows import dict_row
from pydantic import ConfigDict, Field, ValidationError, create_model

from llmprod import connect

MAX_RESULT_CHARS = 6000
EMAIL_RE = re.compile(r"^[^@\s]{1,64}@[^@\s]{1,255}\.[^@\s]{2,}$")

RefundReason = Literal[
    "damaged", "defective", "wrong_item", "not_delivered", "late", "changed_mind", "other"
]

# ---------------------------------------------------------------- ch09_refunds.py (limits)

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


# ---------------------------------------------------------------- ch09_tools.py (tools)


class ToolError(Exception):
    """A failure the model can act on. The message is returned to it verbatim."""


def get_order(
    order_id: Annotated[int, Field(gt=0, description="ShopFlow order id, a positive integer")],
) -> dict[str, Any]:
    """Look up one order: status, totals (integer cents), items with SKUs, payments, refunds
    and shipment dates. Does not return the customer's email."""
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


# ---------------------------------------------------------------- ch09_tools.py (registry)


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


def _clip(text: str) -> str:
    if len(text) <= MAX_RESULT_CHARS:
        return text
    return text[:MAX_RESULT_CHARS] + f"... [truncated, {len(text)} chars total]"


def error_json(message: str) -> str:
    return json.dumps({"error": message})


class LocalToolbox:
    """Runs tool functions in-process. `call` returns a JSON string and NEVER raises."""

    def __init__(
        self, functions: dict[str, Callable[..., Any]], *, clip: bool = True
    ) -> None:
        self.functions = dict(functions)
        self.clip = clip
        self.specs = [openai_tool_spec(f) for f in self.functions.values()]
        self._models = {name: _args_model(f) for name, f in self.functions.items()}

    def parse(self, name: str, raw_arguments: str) -> dict[str, Any]:
        """Validate arguments; raises ToolError with a message the model can act on."""
        if name not in self.functions:
            raise ToolError(f"unknown tool '{name}'. Available tools: {sorted(self.functions)}")
        try:
            args = json.loads(raw_arguments or "{}")
        except json.JSONDecodeError as exc:
            raise ToolError(
                f"arguments are not valid JSON: {exc.msg} (position {exc.pos})"
            ) from exc
        if not isinstance(args, dict):
            raise ToolError("arguments must be a JSON object")
        try:
            parsed = self._models[name].model_validate(args)
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(str(p) for p in e['loc']) or 'arguments'}: {e['msg']}"
                for e in exc.errors()
            )
            raise ToolError(f"invalid arguments for {name}: {problems}") from exc
        return {k: getattr(parsed, k) for k in type(parsed).model_fields}

    def call(self, name: str, raw_arguments: str) -> str:
        try:
            kwargs = self.parse(name, raw_arguments)
            result = self.functions[name](**kwargs)
        except ToolError as exc:
            return error_json(str(exc))
        except Exception as exc:  # noqa: BLE001 - a crashing tool must not crash the agent
            return error_json(f"{name} failed unexpectedly ({type(exc).__name__}); try again later")
        text = json.dumps(result, default=str)
        return _clip(text) if self.clip else text
