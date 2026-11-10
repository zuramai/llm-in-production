"""B1: `json_object` mode is a hint, not a contract: Claude wraps JSON in a code fence."""

from __future__ import annotations

import json

from ch22_common import HAIKU, Result, chat, timed

ID, TITLE, LAYER, KIND = "B1", "json_object returns fenced/prose JSON", "Structured output", "hosted"

PROMPT = "Return a JSON object with keys category and priority for: 'charged twice, need refund'."


def extract_json(text: str) -> dict:
    """Parse the outermost {...} in a reply: tolerates ```json fences and leading prose."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"no JSON object in {text[:60]!r}")
    return json.loads(text[start:end + 1])


@timed
def run() -> Result:
    n, naive_fail, fixed_ok, sample = 3, 0, 0, ""
    for _ in range(n):
        r = chat(HAIKU, [{"role": "user", "content": PROMPT}], max_tokens=150,
                 response_format={"type": "json_object"})
        text = r.choices[0].message.content or ""
        sample = sample or text
        try:
            json.loads(text)
        except json.JSONDecodeError:
            naive_fail += 1
        try:
            fixed_ok += isinstance(extract_json(text), dict)
        except ValueError:
            pass
    return Result(
        ID, TITLE, LAYER, KIND,
        reproduced=naive_fail > 0,
        fixed=fixed_ok == n,
        evidence=f"json.loads failed {naive_fail}/{n}; extract_json parsed {fixed_ok}/{n}; "
                 f"reply starts {sample[:12]!r}",
        data={"sample": sample},
    )
