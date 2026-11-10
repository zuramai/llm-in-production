"""Bonus: the same refund agent on the OpenAI Agents SDK, pointed at OpenRouter through the
Chat Completions model class.

Mapping from the LangGraph version (../01_main-chapter-code/ch10_graph.py):
  * interrupt() in an approval node   -> `needs_approval=` on the tool; the run stops with
                                         `result.interruptions`
  * Postgres checkpointer per step    -> `RunState.to_json()` snapshot, stored by us in
                                         `ch10.sdk_runs` while a run waits for a human
  * Command(resume={approved, role})  -> `state.approve(item)` / `state.reject(item)`, then
                                         `Runner.run(agent, state)`
  * the approver's role               -> application context, read inside the tool
The tools, the validation and the idempotent ledger write are the chapter's own code.

    uv run python ch10/02_bonus_agents-sdk/ch10_sdk_agent.py --only S06,S09,S17
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Annotated, Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "01_main-chapter-code"))

import ch10_setup
import ch10_tools as t
from agents import (
    Agent,
    AsyncOpenAI,
    ModelSettings,
    OpenAIChatCompletionsModel,
    Runner,
    RunState,
    function_tool,
    set_tracing_disabled,
)
from agents.items import ToolCallItem
from agents.tool_context import ToolContext
from ch10_compaction import shrink_json
from ch10_graph import MAX_ASKS, SYSTEM_PROMPT, Decision
from ch10_scenarios import REJECT, SCENARIOS, grade, pick_context, summarize
from psycopg.types.json import Jsonb
from pydantic import Field

from llmprod import connect, get_settings

# Traces go to OpenAI's servers by default; with an OpenRouter key that upload fails (401).
set_tracing_disabled(True)

DDL = """
CREATE TABLE IF NOT EXISTS ch10.sdk_runs (
    thread_id  text PRIMARY KEY,
    state      jsonb NOT NULL,
    updated_at timestamptz NOT NULL DEFAULT now()
);
"""
TOOL_OUTPUT_CHARS = 2_500  # the SDK has no per-call view, so we compact at tool time


def _run(fn, **kwargs) -> str:
    """Run a chapter tool; errors become a JSON message for the model, as in Chapter 9."""
    try:
        out = json.dumps(fn(**kwargs), default=str)
    except (t.ToolError, t.RefundError) as exc:
        return t.error_json(str(exc))
    return shrink_json(out, TOOL_OUTPUT_CHARS)


@function_tool
async def get_order(order_id: Annotated[int, Field(gt=0)]) -> str:
    """Look up one order: status, totals (integer cents), items, payments, refunds (including
    refunds issued by this assistant) and shipment dates."""
    return await asyncio.to_thread(_run, t.get_order, order_id=order_id)


@function_tool
async def search_orders(customer_email: str) -> str:
    """List a customer's 10 most recent orders from their exact email address."""
    return await asyncio.to_thread(_run, t.search_orders, customer_email=customer_email)


@function_tool
async def check_inventory(sku: str) -> str:
    """Stock on hand per warehouse for one product SKU such as SF-ELE-0001."""
    return await asyncio.to_thread(_run, t.check_inventory, sku=sku)


@function_tool
async def get_customer_history(customer_id: Annotated[int, Field(gt=0)]) -> str:
    """Full history of one customer: a summary (order count, refunds, recent refund claims,
    tickets) followed by every order and ticket. Use it to check fraud flags."""
    return await asyncio.to_thread(_run, t.get_customer_history, customer_id=customer_id)


async def _needs_approval(_ctx: Any, params: dict[str, Any], _call_id: str) -> bool:
    """Ask a human only for proposals that pass validation. Invalid ones run without approval
    and the tool body returns the validation error: no money can move on that path."""
    try:
        await asyncio.to_thread(t.validate_refund, **params)
    except (t.RefundError, TypeError):
        return False
    return True


@function_tool(needs_approval=_needs_approval)
async def issue_refund(
    ctx: ToolContext[dict],
    order_id: Annotated[int, Field(gt=0)],
    amount_cents: Annotated[int, Field(gt=0)],
    reason: t.RefundReason,
) -> str:
    """Issue a refund to the original payment method. MOVES MONEY. The run pauses until a
    human with enough authority approves (agent up to $100.00, team lead up to $500.00,
    Finance above). Amounts are integer cents."""
    try:
        proposal = await asyncio.to_thread(t.validate_refund, order_id, amount_cents, reason)
    except t.RefundError as exc:
        return t.error_json(f"not executed: {exc}")
    # Defense in depth: the SDK's "approved" is a boolean. Our app recorded WHO approved; the
    # tool refuses unless that role is senior enough.
    decision = ctx.context.get("decisions", {}).get(ctx.tool_call_id)
    if not decision:
        return t.error_json("not executed: no approval decision recorded")
    if t.ROLE_RANK[decision["role"]] < t.ROLE_RANK[proposal["required_approver"]]:
        return t.error_json(f"not executed: needs '{proposal['required_approver']}'")
    try:
        out = await asyncio.to_thread(
            t.execute_refund, proposal, thread_id=ctx.context["thread_id"],
            tool_call_id=ctx.tool_call_id, role=decision["role"], note=decision.get("note", ""),
        )  # fmt: skip
    except t.RefundError as exc:
        return t.error_json(f"not executed: {exc}")
    return json.dumps(out)


def build_agent(model: str) -> Agent:
    client = AsyncOpenAI(
        base_url=get_settings().llm_base_url, api_key=get_settings().llm_api_key, timeout=120
    )
    extra = {"reasoning_effort": "none"} if model.startswith("openai/gpt-6") else {}
    return Agent(
        name="ShopFlow support assistant",
        instructions=SYSTEM_PROMPT,
        model=OpenAIChatCompletionsModel(model=model, openai_client=client),
        model_settings=ModelSettings(max_tokens=1024, include_usage=True, extra_args=extra,
                                     preserve_raw_usage=True),  # keeps OpenRouter's `cost`
        tools=[get_order, search_orders, check_inventory, get_customer_history, issue_refund],
    )


def save_state(thread_id: str, state: RunState) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT INTO ch10.sdk_runs (thread_id, state) VALUES (%s, %s) ON CONFLICT (thread_id) "
            "DO UPDATE SET state = excluded.state, updated_at = now()",
            (thread_id, Jsonb(state.to_json())),
        )


def load_state(thread_id: str) -> dict[str, Any]:
    with connect() as conn:
        return conn.execute(
            "SELECT state FROM ch10.sdk_runs WHERE thread_id = %s", (thread_id,)
        ).fetchone()[0]


async def drive(
    agent: Agent,
    thread_id: str,
    history: list[Any] | str,
    humans: list[dict],
    asked: list[dict],
    responses: list[Any],
) -> Any:
    """Run one user turn to the end, answering approval requests from the scripted `humans`.
    While a run waits, its only copy is the JSON snapshot in Postgres."""
    ctx: dict[str, Any] = {"thread_id": thread_id, "decisions": {}}
    result = await Runner.run(agent, history, context=ctx, max_turns=10)
    responses.extend(result.raw_responses)
    while result.interruptions:
        save_state(thread_id, result.to_state())
        state = await RunState.from_json(agent, load_state(thread_id), context_override=ctx)
        for item in state.get_interruptions():
            args = json.loads(item.arguments or "{}")
            proposal = await asyncio.to_thread(t.validate_refund, **args)
            needed = proposal["required_approver"]
            for attempt in range(1, MAX_ASKS + 1):  # same rule as the graph: ask at most twice
                d = Decision.model_validate(humans.pop(0) if humans else REJECT)
                asked.append({**proposal, "attempt": attempt, "approved": d.approved,
                              "role": d.role})  # fmt: skip
                enough = t.ROLE_RANK[d.role] >= t.ROLE_RANK[needed]
                if not d.approved or enough:
                    break
            if d.approved and enough:
                ctx["decisions"][item.call_id] = d.model_dump()
                state.approve(item)
            else:
                why = "rejected by a human" if not d.approved else f"needs '{needed}' approval"
                state.reject(item, rejection_message=f"Not refunded: {why}. Do not retry.")
        result = await Runner.run(agent, state, max_turns=10)
        responses.extend(result.raw_responses)
    return result


async def run_scenario(sc: Any, ctx: dict[str, Any], agent: Agent) -> dict[str, Any]:
    from ch10_scenarios import _preseed  # same seeding as the LangGraph eval

    order_ids = [ctx[k]["id"] for k in sc.order_keys]
    ch10_setup.clear_orders(order_ids)
    _preseed(sc, ctx)
    thread = f"sdk-{sc.id}-{uuid.uuid4().hex[:8]}"
    humans, asked = list(sc.humans), []
    history: list[Any] = []
    calls: list[dict[str, Any]] = []
    responses: list[Any] = []
    result, error = None, None
    t0 = time.perf_counter()
    try:
        for text in sc.turns(ctx):
            history = [*history, {"role": "user", "content": text}]
            result = await drive(agent, thread, history, humans, asked, responses)
            history = result.to_input_list()
            for item in result.new_items:
                if isinstance(item, ToolCallItem):
                    raw = item.raw_item
                    calls.append({"name": getattr(raw, "name", None) or raw.get("name"),
                                  "args": {}})  # fmt: skip
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"[:300]
    latency = int((time.perf_counter() - t0) * 1000)
    answer = str(result.final_output) if result is not None else ""
    g = grade(sc, ctx, calls, asked, t.ledger_rows(order_ids) if order_ids else [], answer)
    if error:
        g.update(passed=False, detail=f"ERROR {error}; {g['detail']}")
    ch10_setup.clear_orders(order_ids)
    cost = sum(float((r.raw_usage or {}).get("cost") or 0) for r in responses)
    prompts = [r.usage.input_tokens for r in responses]
    cached = sum(r.usage.input_tokens_details.cached_tokens or 0 for r in responses)
    return {"id": sc.id, **g, "tools": [c["name"] for c in calls], "approval_requests": len(asked),
            "latency_ms": latency, "cost_usd": cost, "prompt_tokens": sum(prompts),
            "cached_tokens": cached, "max_prompt_tokens": max(prompts, default=0),
            "answer": answer}  # fmt: skip


async def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default="openai/gpt-6-luna")
    p.add_argument("--only", default="")
    args = p.parse_args()
    ch10_setup.setup()
    with connect() as conn:
        conn.execute(DDL)
    ctx = pick_context()
    agent = build_agent(args.model)
    only = [s for s in args.only.split(",") if s]
    rows = []
    for sc in [s for s in SCENARIOS if not only or s.id in only]:
        r = await run_scenario(sc, ctx, agent)
        rows.append(r)
        mark = "PASS" if r["passed"] else "FAIL"
        print(f"{sc.id} {mark} {r['latency_ms'] / 1000:5.1f}s {r['tools']} {r['detail']}",
              flush=True)  # fmt: skip
    summary = summarize(rows)
    out = HERE / "results" / f"sdk-{args.model.replace('/', '_')}{'-subset' if only else ''}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"model": args.model, "summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    asyncio.run(main())
