"""D1: the model fills required tool arguments it doesn't have (order_id=0, 7999, 1...)."""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor

from ch22_common import LUNA, Result, chat, timed

ID, TITLE, LAYER, KIND = "D1", "Tool-argument hallucination", "Agents & tools", "hosted"

TOOLS = [
    {"type": "function", "function": {
        "name": "get_order_status", "description": "Look up an order by id.",
        "parameters": {"type": "object", "properties": {"order_id": {"type": "integer"}},
                       "required": ["order_id"], "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "issue_refund", "description": "Refund an order.",
        "parameters": {"type": "object",
                       "properties": {"order_id": {"type": "integer"},
                                      "amount_cents": {"type": "integer"}},
                       "required": ["order_id", "amount_cents"], "additionalProperties": False}}},
]
SYSTEM = "You are ShopFlow support. Use tools."
USERS = [
    "Where is my order? It's the blue yoga mat I bought last week.",
    "Refund my last order please, the skillet arrived cracked.",
    "I need a refund for the earbuds, they cost $79.99.",
]
ID_ARGS = {"order_id"}


def ungrounded_ids(arguments: dict, transcript: str) -> list[str]:
    """ID arguments whose value never appeared as a standalone number in the conversation.

    Prices don't count: "$79.99" doesn't make order 79 or 99 real.
    """
    seen = set(re.findall(r"(?<![\d.$])\d+(?!\d|\.\d)", transcript))
    return [k for k in ID_ARGS & arguments.keys()
            if arguments[k] is None or str(arguments[k]) not in seen]


def guarded_dispatch(tool_call, transcript: str) -> dict:
    """Refuse to execute a call built on invented ids; tell the model what to do instead."""
    args = json.loads(tool_call.function.arguments or "{}")
    bad = ungrounded_ids(args, transcript)
    if bad:
        return {"executed": False, "tool_result": {
            "error": f"{', '.join(bad)} was not provided by the customer. Do not guess ids: ask "
                     "the customer for it."}}
    return {"executed": True, "tool_result": {"status": "shipped"}}


def _episode(user: str) -> dict:
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    r = chat(LUNA, messages, max_tokens=200, tools=TOOLS, reasoning_effort="none")
    msg = r.choices[0].message
    if not msg.tool_calls:
        return {"called": False, "invented": False, "executed": False, "asked": True}
    tc = msg.tool_calls[0]
    args = json.loads(tc.function.arguments or "{}")
    invented = bool(ungrounded_ids(args, user))
    outcome = guarded_dispatch(tc, user)
    asked = False
    if not outcome["executed"]:
        messages += [msg.model_dump(exclude_none=True),
                     {"role": "tool", "tool_call_id": tc.id,
                      "content": json.dumps(outcome["tool_result"])}]
        r2 = chat(LUNA, messages, max_tokens=200, tools=TOOLS, reasoning_effort="none")
        m2 = r2.choices[0].message
        asked = not m2.tool_calls and "order" in (m2.content or "").lower()
    return {"called": True, "invented": invented, "args": args,
            "executed": outcome["executed"], "asked": asked}


@timed
def run(n: int = 5) -> Result:
    with ThreadPoolExecutor(8) as pool:
        eps = list(pool.map(_episode, [u for u in USERS for _ in range(n)]))
    invented = sum(e["invented"] for e in eps)
    executed_invented = sum(e["invented"] and e["executed"] for e in eps)
    asked = sum(e["asked"] for e in eps)
    examples = sorted({json.dumps(e["args"]) for e in eps if e["invented"]})[:5]
    return Result(
        ID, TITLE, LAYER, KIND,
        reproduced=invented > 0,
        fixed=executed_invented == 0 and asked >= invented * 0.8,
        evidence=f"luna (reasoning_effort=none), {len(eps)} episodes with no order id given: "
                 f"invented id in {invented}, e.g. {', '.join(examples)}; guard executed "
                 f"{executed_invented} of them, model then asked the customer in {asked}",
        data={"episodes": eps},
    )
