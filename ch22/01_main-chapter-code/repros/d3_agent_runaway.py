"""D3: an agent pages through a search forever, resending a growing context on every step."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from ch22_common import HAIKU, LUNA, Result, chat, timed

ID, TITLE, LAYER, KIND = "D3", "Agent cost runaway loop", "Agents & tools", "hosted"

TOOLS = [{"type": "function", "function": {
    "name": "search_orders",
    "description": "Search orders by customer email. Results are paginated; pass next_page to "
                   "continue.",
    "parameters": {"type": "object",
                   "properties": {"email": {"type": "string"}, "page": {"type": ["string", "null"]}},
                   "required": ["email", "page"], "additionalProperties": False}}}]
SYSTEM = "You are ShopFlow support. Use tools to find the customer's order before answering."
USER = "Can you find my order for the cast iron skillet? My email is jane.roe@example.com"


def fake_page(step: int) -> dict:
    """Every page has 3 orders that aren't the skillet, and there's always a next page."""
    return {"orders": [{"id": 30000 + step * 10 + k, "product": "Yoga mat"} for k in range(3)],
            "next_page": f"p{step + 2}"}


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Guard:
    """Caps an agent run by model calls, dollars, and calls per tool."""

    max_steps: int = 8
    max_usd: float = 0.005
    max_per_tool: int = 4
    steps: int = 0
    usd: float = 0.0
    per_tool: dict = field(default_factory=dict)

    def charge(self, usd: float) -> None:
        self.steps += 1
        self.usd += usd
        if self.steps >= self.max_steps:
            raise BudgetExceeded(f"step cap {self.max_steps}")
        if self.usd >= self.max_usd:
            raise BudgetExceeded(f"${self.usd:.4f} >= ${self.max_usd}")

    def check_call(self, name: str) -> None:
        self.per_tool[name] = self.per_tool.get(name, 0) + 1
        if self.per_tool[name] > self.max_per_tool:
            raise BudgetExceeded(f"{name} called {self.per_tool[name]}x")


def agent(model: str, guard: Guard | None, hard_cap: int) -> dict:
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER}]
    kw = {"reasoning_effort": "none"} if model == LUNA else {}
    trace, stop, answer = [], "answered", ""
    for step in range(hard_cap):
        r = chat(model, messages, max_tokens=150, tools=TOOLS, **kw)
        cost = float(getattr(r.usage, "cost", 0) or 0)
        trace.append({"prompt_tokens": r.usage.prompt_tokens, "cost": cost})
        msg = r.choices[0].message
        if not msg.tool_calls:
            answer = msg.content or ""
            break
        messages.append(msg.model_dump(exclude_none=True))
        answered: set[str] = set()
        try:
            if guard:
                guard.charge(cost)
            for tc in msg.tool_calls:
                if guard:
                    guard.check_call(tc.function.name)
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": json.dumps(fake_page(step))})
                answered.add(tc.id)
        except BudgetExceeded as e:
            stop = f"guard: {e}"
            # Graceful stop: close the open tool calls and ask for a final answer, no tools.
            for tc in msg.tool_calls:
                if tc.id not in answered:
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(
                        {"error": "search budget exhausted; answer with what you have"})})
            r = chat(model, messages, max_tokens=150, tools=TOOLS, tool_choice="none", **kw)
            trace.append({"prompt_tokens": r.usage.prompt_tokens,
                          "cost": float(getattr(r.usage, "cost", 0) or 0)})
            answer = r.choices[0].message.content or ""
            break
    else:
        stop = f"hard cap {hard_cap}"
    return {"steps": len(trace), "stop": stop, "trace": trace, "answer": answer[:120],
            "prompt_tokens": sum(t["prompt_tokens"] for t in trace),
            "cost": sum(t["cost"] for t in trace)}


@timed
def run(hard_cap: int = 20) -> Result:
    runs = {m: agent(m, None, hard_cap) for m in (LUNA, HAIKU)}
    guarded = agent(LUNA, Guard(), hard_cap)
    naive = runs[LUNA]
    growth = [t["prompt_tokens"] for t in naive["trace"]]
    return Result(
        ID, TITLE, LAYER, KIND,
        reproduced=naive["stop"].startswith("hard cap"),
        fixed=guarded["stop"].startswith("guard") and bool(guarded["answer"])
        and guarded["prompt_tokens"] < naive["prompt_tokens"] / 2,
        evidence=f"search never finds it, always has next_page: luna no guard -> "
                 f"{naive['steps']} calls ({naive['stop']}), prompt tokens/call "
                 f"{growth[0]}->{growth[-1]}, {naive['prompt_tokens']:,} tok total; haiku -> "
                 f"{runs[HAIKU]['steps']} calls ({runs[HAIKU]['stop']}); luna + guard -> "
                 f"{guarded['steps']} calls ({guarded['stop']}), {guarded['prompt_tokens']:,} tok, "
                 f"answer {guarded['answer'][:50]!r}",
        data={"luna": naive, "haiku": runs[HAIKU], "guarded": guarded},
    )
