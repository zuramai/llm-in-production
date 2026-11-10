"""A1: finish_reason == "length" truncates structured output mid-string."""

from __future__ import annotations

import json
from typing import Any

from ch22_common import HAIKU, Result, chat, timed

ID, TITLE, LAYER, KIND = "A1", 'finish_reason == "length" truncates JSON', "API & tokens", "hosted"

TICKET = (
    "Classify this support ticket and summarize it in one sentence: 'I was charged twice for "
    "order #4512 and the package still hasn't arrived after 9 days. I need a refund today.'"
)
SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "ticket",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "category": {"type": "string", "enum": ["billing", "shipping", "account", "other"]},
                "priority": {"type": "string", "enum": ["low", "medium", "high", "urgent"]},
                "summary": {"type": "string"},
            },
            "required": ["category", "priority", "summary"],
        },
    },
}


class TruncatedOutput(RuntimeError):
    """The model stopped because it ran out of tokens, not because it was done."""


def parse_structured(choice: Any) -> dict:
    """Parse a structured-output choice, refusing anything that didn't finish normally."""
    if choice.finish_reason != "stop":
        raise TruncatedOutput(f"finish_reason={choice.finish_reason!r}")
    if not choice.message.content:
        raise TruncatedOutput("empty content")
    return json.loads(choice.message.content)


def create_structured(messages: list[dict], model: str, max_tokens: int, *, retries: int = 2):
    """Call, and on a length stop retry with 4x the budget instead of parsing half a JSON."""
    budget = max_tokens
    for attempt in range(retries + 1):
        r = chat(model, messages, max_tokens=budget, response_format=SCHEMA)
        try:
            return parse_structured(r.choices[0]), budget, attempt + 1
        except TruncatedOutput:
            if attempt == retries:
                raise
            budget *= 4
    raise AssertionError("unreachable")


@timed
def run() -> Result:
    messages = [{"role": "user", "content": TICKET}]
    # Naive: what most code does, json.loads() whatever came back.
    r = chat(HAIKU, messages, max_tokens=30, response_format=SCHEMA)
    choice = r.choices[0]
    try:
        json.loads(choice.message.content or "")
        naive_ok = True
    except json.JSONDecodeError:
        naive_ok = False
    data, budget, attempts = create_structured(messages, HAIKU, max_tokens=30)
    return Result(
        ID, TITLE, LAYER, KIND,
        reproduced=choice.finish_reason == "length" and not naive_ok,
        fixed=set(data) == {"category", "priority", "summary"},
        evidence=(
            f"max_tokens=30: finish_reason={choice.finish_reason}, content ends "
            f"{(choice.message.content or '')[-25:]!r}; fix: {attempts} calls, budget {budget}"
        ),
        data={"naive_tail": (choice.message.content or "")[-40:], "fixed": data},
    )
