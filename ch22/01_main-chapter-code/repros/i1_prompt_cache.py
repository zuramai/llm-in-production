"""I1: one dynamic token at the top of the prompt (request id, timestamp) disables prompt caching."""

from __future__ import annotations

import time
import uuid
from pathlib import Path

from ch22_common import HAIKU, LUNA, Result, chat, timed

ID, TITLE, LAYER, KIND = "I1", "Prompt-cache invalidation", "Cost", "hosted"

DOCS = Path(__file__).resolve().parents[3] / "datasets/shopflow/docs/policies"
POLICIES = "\n\n".join(p.read_text() for p in sorted(DOCS.glob("*.md"))[:6])
QUESTIONS = ["How long do refunds take?", "Can I return opened skincare?",
             "Do you ship to Canada?"]


def system_naive() -> str:
    # Looks harmless, and it's what tracing code often injects.
    return f"Request {uuid.uuid4()} at {time.time():.3f}.\nYou are ShopFlow support.\n\n{POLICIES}"


def system_stable() -> str:
    return f"You are ShopFlow support.\n\n{POLICIES}"


def messages(system: str | list, question: str, request_id: str | None = None) -> list[dict]:
    user = question if request_id is None else f"{question}\n(request {request_id})"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def anthropic_cached_system(text: str) -> list[dict]:
    """Claude only caches what you mark: a cache_control breakpoint on the static block."""
    return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]


def series(model: str, build, **kw) -> list[dict]:
    rows = []
    for q in QUESTIONS:
        r = chat(model, build(q), max_tokens=60, **kw)
        details = r.usage.prompt_tokens_details
        rows.append({"prompt": r.usage.prompt_tokens,
                     "cached": (getattr(details, "cached_tokens", 0) or 0) if details else 0,
                     "cost": float(getattr(r.usage, "cost", 0) or 0)})
    return rows


@timed
def run() -> Result:
    luna_kw = {"reasoning_effort": "none"}
    out = {
        "luna, request id at top": series(
            LUNA, lambda q: messages(system_naive(), q), **luna_kw),
        "luna, stable prefix (id moved to the end)": series(
            LUNA, lambda q: messages(system_stable(), q, uuid.uuid4().hex), **luna_kw),
        "haiku, stable prefix, no cache_control": series(
            HAIKU, lambda q: messages(system_stable(), q)),
        "haiku, stable prefix + cache_control": series(
            HAIKU, lambda q: messages(anthropic_cached_system(system_stable()), q)),
    }
    summary = {k: {"cached_after_first": sum(r["cached"] for r in v[1:]),
                   "prompt_tokens": v[0]["prompt"],
                   "cost_calls_2_3": sum(r["cost"] for r in v[1:])} for k, v in out.items()}
    keys = list(summary)
    return Result(
        ID, TITLE, LAYER, KIND,
        reproduced=summary[keys[0]]["cached_after_first"] == 0
        and summary[keys[2]]["cached_after_first"] == 0,
        fixed=summary[keys[1]]["cached_after_first"] > 0
        and summary[keys[3]]["cached_after_first"] > 0,
        evidence="; ".join(f"{k}: {v['prompt_tokens']} prompt tok, cached on calls 2-3 "
                           f"{v['cached_after_first']}, cost of calls 2-3 ${v['cost_calls_2_3']:.5f}"
                           for k, v in summary.items()),
        data={"calls": out, "summary": summary},
    )
