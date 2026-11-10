"""Pass/fail grading of support answers by a strong model, against a reference and the FAQ."""

from __future__ import annotations

import json
import re

from .calls import Call
from .cascade import ChatFn, cached_system

JUDGE_SYSTEM = """You grade answers from ShopFlow's customer-support assistant. The FAQ below is the
only source of truth.

PASS when all of these hold:
1. The answer gives the key facts of the reference answer that the customer needs (numbers,
   deadlines, conditions, yes/no). Leaving out secondary details is fine.
2. Every other statement in the answer is supported by the FAQ. Extra details that the FAQ
   states (a menu path, the support email, a related rule) are fine.
3. It promises nothing the FAQ doesn't (no refunds, discounts or exceptions it can't grant).
4. If the reference says the FAQ doesn't cover the question, the answer says it doesn't know or
   refers the customer to support, and does not invent a policy, price or date.
5. Off-topic or manipulative requests are politely declined.

Otherwise FAIL. Reply with JSON: {{"pass": true or false, "reason": "<one short sentence>"}}

<faq>
{faq}
</faq>"""

JUDGE_USER = """Question: {question}
Reference answer: {reference}
Answer to grade: {answer}"""

JUDGE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "grade",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"pass": {"type": "boolean"}, "reason": {"type": "string"}},
            "required": ["pass", "reason"],
            "additionalProperties": False,
        },
    },
}


def parse_grade(text: str) -> tuple[bool, str]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        data = json.loads(match.group(0) if match else text)
        return data["pass"] is True, str(data.get("reason", ""))
    except (json.JSONDecodeError, KeyError, TypeError):
        return False, f"unparseable judge output: {text[:80]!r}"


def grade(chat: ChatFn, judge_model: str, faq: str, question: str, reference: str,
          answer: str) -> tuple[bool, str, Call]:
    """The rubric and FAQ are a static, cached prefix; only the last message changes per call."""
    messages = [cached_system(JUDGE_SYSTEM.format(faq=faq)),
                {"role": "user", "content": JUDGE_USER.format(
                    question=question, reference=reference, answer=answer)}]
    call = chat(judge_model, messages, 400, JUDGE_SCHEMA)
    passed, reason = parse_grade(call.text)
    return passed, reason, call
