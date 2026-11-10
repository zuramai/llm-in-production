"""LLM judge for answers: correct vs a reference, and faithful to the retrieved context.

The request shape is chapter 3's judge (`ch03_evalkit._ask_judge`): strict JSON schema, `reason`
before the verdict, `max_tokens` always capped, truncation is an error, cost read from `usage`.
The judge model is chapter 3's pick: `openai/gpt-6-luna` with a written rubric and a reference.
In chapter 3 it agreed with 40 reference labels written by a blind AI annotator (not a human) at
Cohen's kappa 0.75-0.82 over three runs. It costs $0.10/$0.50 per 1M tokens.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from llmprod import chat_client

JUDGE_MODEL = "openai/gpt-6-luna"
NOT_IN_DOCS = "NOT_IN_DOCS"

SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {"type": "string"},
        "correct": {"type": "boolean"},
        "faithful": {"type": "boolean"},
    },
    "required": ["reason", "correct", "faithful"],
    "additionalProperties": False,
}

RUBRIC = """You grade a documentation assistant's answer. You get the QUESTION, a REFERENCE
answer written by a human, the CONTEXT the assistant was given, and the ANSWER.

correct: true if the ANSWER agrees with the REFERENCE on every key fact (numbers, names,
commands, yes/no). Extra true detail is fine. A contradiction or a missing key fact is false.
If the REFERENCE is exactly NOT_IN_DOCS, the documents do not contain the answer: the ANSWER is
correct only if it declines to answer (e.g. "I can't find that") and invents nothing.

faithful: true if every factual claim in the ANSWER is supported by the CONTEXT. Claims that
appear only in the REFERENCE or in outside knowledge make it false. A refusal is faithful.
Citation tags like [doc:path#N] are not claims; ignore them.

Write `reason` (one sentence) first, then decide."""


@dataclass
class Verdict:
    correct: bool
    faithful: bool
    reason: str
    cost_usd: float


def ask_judge(client, model: str, system: str, user: str, name: str, schema: dict,
              **opts) -> tuple[dict, float]:
    """Chapter 3's judge call: strict structured output, capped, truncation raises."""
    resp = client.chat.completions.create(
        model=model,
        **{"max_tokens": 1000, **opts},
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": name, "strict": True, "schema": schema},
        },
    )
    choice = resp.choices[0]
    if choice.finish_reason == "length":
        raise RuntimeError(f"judge output truncated (finish_reason=length) for {model}")
    cost = (resp.usage.model_extra or {}).get("cost", 0.0) if resp.usage else 0.0
    return json.loads(choice.message.content), float(cost or 0.0)


def judge(question: str, reference: str, context: str, answer: str,
          model: str = JUDGE_MODEL) -> Verdict:
    user = (f"QUESTION: {question}\n\nREFERENCE: {reference}\n\nCONTEXT:\n{context}\n\n"
            f"ANSWER: {answer}")
    data, cost = ask_judge(chat_client(timeout=90), model, RUBRIC, user, "Verdict", SCHEMA)
    return Verdict(bool(data["correct"]), bool(data["faithful"]), str(data["reason"]), cost)
