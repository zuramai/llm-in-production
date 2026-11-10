"""Context compaction for long tool outputs.

The graph's state (and so the Postgres checkpoint) always keeps every tool output in full: that
is the audit trail. What we compact is the *view* sent to the model on each call:

  * `raw`     - send everything (the baseline).
  * `clip`    - Chapter 9's rule: cut each tool output at 6,000 characters.
  * `compact` - structure-aware shrinking: lists keep their first items plus an "N more omitted"
                marker, long strings are cut, the JSON stays valid and short top-level fields
                (a tool's `summary`) survive. Outputs the model has already acted on
                ("stale") get a smaller budget than the ones it is about to read ("fresh").
  * `stable`  - like `compact`, but every output keeps its fresh budget forever. Earlier
                messages never change between calls, so the provider's prompt cache (which
                matches on an unchanged prefix) keeps working.

All functions here are pure, so they are unit-tested without a model or a database.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

Mode = Literal["raw", "clip", "compact", "stable"]

# (max list items, max string chars), from gentle to brutal
_SCHEDULE = [(50, 400), (20, 200), (10, 160), (5, 120), (3, 80), (2, 60), (1, 40), (0, 30)]


def approx_tokens(text: str) -> int:
    """~4 characters per token for English and JSON. Only used for budgeting and charts; real
    token counts come from the provider's `usage`."""
    return (len(text) + 3) // 4


def _prune(value: Any, list_cap: int, str_cap: int) -> Any:
    if isinstance(value, dict):
        return {k: _prune(v, list_cap, str_cap) for k, v in value.items()}
    if isinstance(value, list):
        kept = [_prune(v, list_cap, str_cap) for v in value[:list_cap]]
        if len(value) > list_cap:
            kept.append(f"... {len(value) - list_cap} more items omitted")
        return kept
    if isinstance(value, str) and len(value) > str_cap:
        return value[:str_cap] + f"...[+{len(value) - str_cap} chars]"
    return value


def shrink_json(text: str, budget_chars: int) -> str:
    """Shrink a JSON tool output to at most `budget_chars`, keeping it valid JSON whenever the
    pruned structure fits. Non-JSON text is clipped with an explicit marker."""
    if len(text) <= budget_chars:
        return text
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return _clip(text, budget_chars)
    for list_cap, str_cap in _SCHEDULE:
        out = json.dumps(_prune(value, list_cap, str_cap), default=str, separators=(",", ":"))
        if len(out) <= budget_chars:
            return out
    return _clip(out, budget_chars)


def _clip(text: str, budget_chars: int) -> str:
    if len(text) <= budget_chars:
        return text
    marker = f"... [truncated, {len(text)} chars total]"
    return text[: max(budget_chars - len(marker), 0)] + marker


@dataclass(frozen=True)
class Compaction:
    mode: Mode = "stable"  # the chapter's choice: see README, "Compact long tool outputs"
    fresh_chars: int = 2_500  # tool outputs the model has not answered yet
    stale_chars: int = 600  # tool outputs from earlier steps
    clip_chars: int = 6_000  # Chapter 9's MAX_RESULT_CHARS, for mode="clip"

    def view(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """The messages to send to the model. Never mutates `messages`."""
        if self.mode == "raw":
            return list(messages)
        last_call = max(
            (i for i, m in enumerate(messages) if m["role"] == "assistant" and m.get("tool_calls")),
            default=-1,
        )
        out = []
        for i, m in enumerate(messages):
            if m["role"] != "tool":
                out.append(m)
                continue
            if self.mode == "clip":
                content = _clip(m["content"], self.clip_chars)
            else:
                fresh = i > last_call or self.mode == "stable"
                budget = self.fresh_chars if fresh else self.stale_chars
                content = shrink_json(m["content"], budget)
            out.append({**m, "content": content})
        return out


def prompt_chars(messages: list[dict[str, Any]]) -> int:
    """Characters of message content plus tool-call arguments: a cheap size measure."""
    n = 0
    for m in messages:
        n += len(m.get("content") or "")
        for tc in m.get("tool_calls") or []:
            n += len(tc["function"]["arguments"]) + len(tc["function"]["name"])
    return n
