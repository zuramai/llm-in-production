"""A2: hidden reasoning tokens eat max_tokens: empty answer, still billed."""

from __future__ import annotations

from typing import Any

from ch22_common import LUNA, QWEN_FLASH, Result, chat, timed

ID, TITLE, LAYER, KIND = "A2", "Reasoning eats max_tokens: empty but billed", "API & tokens", "hosted"

PROMPT = (
    "Classify this ticket as billing, shipping or other, then give a one-sentence summary: "
    "'I was charged twice for order #4512 and need a refund today.'"
)


def visible_answer(response: Any) -> str:
    """Return the answer text, or raise if reasoning used the whole budget."""
    choice = response.choices[0]
    details = getattr(response.usage, "completion_tokens_details", None)
    reasoning = getattr(details, "reasoning_tokens", 0) or 0
    if not choice.message.content:
        raise RuntimeError(
            f"empty answer: finish_reason={choice.finish_reason}, "
            f"{reasoning}/{response.usage.completion_tokens} completion tokens were reasoning"
        )
    return choice.message.content


@timed
def run() -> Result:
    rows = []
    messages = [{"role": "user", "content": PROMPT}]
    for model in (QWEN_FLASH, LUNA):
        for effort in (None, "none"):
            kw = {"reasoning_effort": effort} if effort else {}
            r = chat(model, messages, max_tokens=60, **kw)
            d = r.usage.completion_tokens_details
            rows.append({
                "model": model,
                "reasoning_effort": effort or "default",
                "finish_reason": r.choices[0].finish_reason,
                "content_chars": len(r.choices[0].message.content or ""),
                "reasoning_tokens": getattr(d, "reasoning_tokens", 0) or 0,
                "completion_tokens": r.usage.completion_tokens,
                "cost_usd": float(getattr(r.usage, "cost", 0) or 0),
            })
    empty_billed = [x for x in rows if x["reasoning_effort"] == "default"
                    and x["content_chars"] == 0 and x["cost_usd"] > 0]
    fixed = all(x["content_chars"] > 0 for x in rows if x["reasoning_effort"] == "none")
    return Result(
        ID, TITLE, LAYER, KIND,
        reproduced=len(empty_billed) > 0,
        fixed=fixed,
        evidence="; ".join(
            f"{x['model'].split('/')[1]} effort={x['reasoning_effort']}: {x['finish_reason']}, "
            f"{x['content_chars']} chars, {x['reasoning_tokens']}/{x['completion_tokens']} reasoning"
            for x in rows
        ),
        data={"rows": rows},
    )
