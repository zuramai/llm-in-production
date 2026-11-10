"""I2: every tool schema is resent, and billed, on every call, whether the turn needs it or not."""

from __future__ import annotations

from ch22_common import LUNA, Result, chat, timed

ID, TITLE, LAYER, KIND = "I2", "Tool schemas tax every call", "Cost", "hosted"


def tool(name: str, description: str, **props: str) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object",
                       "properties": {k: {"type": v} for k, v in props.items()},
                       "required": list(props), "additionalProperties": False}}}


TOOLS = [
    tool("get_order_status", "Look up the status, items and tracking of an order by id.",
         order_id="integer"),
    tool("search_orders", "Find a customer's orders by email, newest first.", email="string"),
    tool("issue_refund", "Refund part or all of an order. Requires approval above $200.",
         order_id="integer", amount_cents="integer", reason="string"),
    tool("cancel_order", "Cancel an order that hasn't shipped yet.", order_id="integer",
         reason="string"),
    tool("update_shipping_address", "Change the shipping address of an unshipped order.",
         order_id="integer", address="string"),
    tool("search_faq", "Search the help center and policies for an answer.", query="string"),
    tool("check_inventory", "Stock level of a product across warehouses.", sku="string"),
    tool("escalate_to_human", "Hand the conversation to a human agent with a summary.",
         summary="string", priority="string"),
]
ROUTES = {  # a cheap keyword router (or a small classifier) picks the tools a turn can use
    "policy": ["search_faq", "escalate_to_human"],
    "order": ["get_order_status", "search_orders", "escalate_to_human"],
}


def tools_for(message: str) -> list[dict]:
    """Expose only the tools this kind of turn can need."""
    m = message.lower()
    route = "order" if any(w in m for w in ("order", "package", "tracking")) else "policy"
    return [t for t in TOOLS if t["function"]["name"] in ROUTES[route]]


@timed
def run() -> Result:
    msg = [{"role": "user", "content": "What's your return window for opened skincare?"}]
    counts = {}
    for label, tools in (("no tools", None), (f"all {len(TOOLS)} tools", TOOLS),
                         ("routed tools", tools_for(msg[0]["content"]))):
        kw = {"tools": tools} if tools else {}
        r = chat(LUNA, msg, max_tokens=60, reasoning_effort="none", **kw)
        counts[label] = r.usage.prompt_tokens
    base = counts["no tools"]
    all_tax = counts[f"all {len(TOOLS)} tools"] - base
    routed_tax = counts["routed tools"] - base
    return Result(
        ID, TITLE, LAYER, KIND,
        reproduced=all_tax > 5 * base,
        fixed=routed_tax < all_tax / 2,
        evidence="same 9-word question, prompt tokens: " + ", ".join(
            f"{k} {v}" for k, v in counts.items())
        + f" -> schemas add {all_tax} tokens per call ({all_tax / len(TOOLS):.0f}/tool); routing "
          f"cuts that to {routed_tax}",
        data=counts,
    )
