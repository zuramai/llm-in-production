"""Cheap → strong cascade: answer with a cheap model, let a verifier check it, escalate if needed.

FrugalGPT (Chen et al., 2023) showed that most questions don't need the expensive model. The
hard part is knowing *which* ones do, without asking the expensive model. Here a cheap
verifier call checks the draft against the same FAQ the answerer saw.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .calls import Call

# (model, messages, max_tokens, response_format or None) -> Call
ChatFn = Callable[[str, list[dict[str, Any]], int, dict[str, Any] | None], Call]

# Static instructions + FAQ first (a cacheable prefix), the per-request parts last.
VERIFIER_SYSTEM = """You check a customer-support draft answer against the ShopFlow FAQ.

Reply with JSON only:
- "supported": true if every fact, number, deadline and promise in the draft is stated in the
  FAQ (or the draft correctly says the FAQ doesn't cover it). false if anything is invented,
  wrong, or contradicts the FAQ.
- "answers": true if the draft actually answers what was asked (or, when the FAQ has no answer,
  says so and points to support). false if it dodges, answers a different question, or says
  "I don't know" although the FAQ does answer it.
- "reason": one short sentence.

<faq>
{faq}
</faq>"""

VERIFIER_USER = """Customer question: {question}

Draft answer: {draft}"""

VERDICT_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "verdict",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "supported": {"type": "boolean"},
                "answers": {"type": "boolean"},
                "reason": {"type": "string"},
            },
            "required": ["supported", "answers", "reason"],
            "additionalProperties": False,
        },
    },
}


@dataclass(frozen=True)
class Verdict:
    supported: bool
    answers: bool
    reason: str

    @property
    def ok(self) -> bool:
        return self.supported and self.answers


def parse_verdict(text: str) -> Verdict:
    """Parse the verifier's JSON. Anything unparseable counts as a failed check (escalate)."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        data = json.loads(match.group(0) if match else text)
        # Only a real JSON `true` passes; "true", 1 or a missing key all mean "escalate".
        return Verdict(data["supported"] is True, data["answers"] is True,
                       str(data.get("reason", "")))
    except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
        return Verdict(False, False, f"unparseable verifier output: {text[:80]!r}")


@dataclass
class CascadeResult:
    question: str
    answer: str
    answered_by: str
    escalated: bool
    verdict: Verdict | None
    calls: list[Call] = field(default_factory=list)

    @property
    def cost_usd(self) -> float:
        return sum(c.cost_usd for c in self.calls)

    @property
    def latency_s(self) -> float:
        return sum(c.latency_s for c in self.calls)


def cached_system(text: str) -> dict[str, Any]:
    """A system message with a prompt-cache breakpoint at its end.

    Anthropic models cache only up to an explicit `cache_control` breakpoint; OpenAI models cache
    prefixes automatically and the gateway passes the block through to OpenRouter either way.
    """
    return {"role": "system",
            "content": [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]}


def answer_messages(system: str, question: str) -> list[dict[str, Any]]:
    return [cached_system(system), {"role": "user", "content": question}]


def verify(chat: ChatFn, verifier: str, faq: str, question: str, draft: str) -> tuple[Verdict, Call]:
    messages = [cached_system(VERIFIER_SYSTEM.format(faq=faq)),
                {"role": "user", "content": VERIFIER_USER.format(question=question, draft=draft)}]
    # gpt-6-luna reasons by default; reasoning tokens count against max_tokens. With 30 tokens
    # it returned no content at all for a one-field JSON (finish_reason="length").
    call = chat(verifier, messages, 800, VERDICT_SCHEMA)
    return parse_verdict(call.text), call


def cascade(chat: ChatFn, question: str, system: str, faq: str,
            tiers: Sequence[str], verifier: str, max_tokens: int = 800) -> CascadeResult:
    """Try each tier in order; accept the first draft the verifier passes. The last tier is final.

    The last tier is never verified: there is nothing left to escalate to, and paying for a
    check we can't act on is waste.
    """
    if not tiers:
        raise ValueError("need at least one tier")
    calls: list[Call] = []
    verdict = None
    for i, model in enumerate(tiers):
        draft = chat(model, answer_messages(system, question), max_tokens, None)
        calls.append(draft)
        if i == len(tiers) - 1:
            return CascadeResult(question, draft.text, model, i > 0, verdict, calls)
        verdict, check = verify(chat, verifier, faq, question, draft.text)
        calls.append(check)
        if verdict.ok:
            return CascadeResult(question, draft.text, model, i > 0, verdict, calls)
    raise AssertionError("unreachable")
