"""LLM judge for answers: correct vs a reference, and faithful to the retrieved context."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from llmprod import chat_client

JUDGE_MODEL = "anthropic/claude-haiku-5.5"
NOT_IN_DOCS = "NOT_IN_DOCS"

SCHEMA = {
    "type": "object",
    "properties": {
        "correct": {"type": "boolean"},
        "faithful": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["correct", "faithful", "reason"],
    "additionalProperties": False,
}

RUBRIC = """You grade a documentation assistant's answer. You get the question, a REFERENCE
answer written by a human, the CONTEXT the assistant was given, and the ANSWER.

correct: true if the ANSWER agrees with the REFERENCE on every key fact (numbers, names,
commands, yes/no). Extra true detail is fine. A contradiction or a missing key fact is false.
If the REFERENCE is exactly NOT_IN_DOCS, the documents do not contain the answer: the ANSWER is
correct only if it declines to answer (e.g. "I can't find that") and invents nothing.

faithful: true if every factual claim in the ANSWER is supported by the CONTEXT. Claims that
appear only in the REFERENCE or in outside knowledge make it false. A refusal is faithful.

Reply with JSON: {"correct": bool, "faithful": bool, "reason": "<one sentence>"}."""


@dataclass
class Verdict:
    correct: bool
    faithful: bool
    reason: str
    cost_usd: float


def _parse(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)  # Claude may wrap JSON in prose
    return json.loads(m.group(0) if m else text)


def judge(question: str, reference: str, context: str, answer: str,
          model: str = JUDGE_MODEL) -> Verdict:
    # Judging the whole-corpus baseline would send 35k tokens of context per call; for that
    # baseline the caller passes a short context and we judge correctness only through REFERENCE.
    resp = chat_client(timeout=90).chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": RUBRIC},
            {"role": "user", "content": (
                f"QUESTION: {question}\n\nREFERENCE: {reference}\n\nCONTEXT:\n{context}\n\n"
                f"ANSWER: {answer}")},
        ],
        max_tokens=300,
        response_format={"type": "json_schema", "json_schema": {
            "name": "Verdict", "strict": True, "schema": SCHEMA}},
    )
    data = _parse(resp.choices[0].message.content or "{}")
    cost = float(getattr(resp.usage, "cost", 0.0) or 0.0)
    return Verdict(bool(data["correct"]), bool(data["faithful"]), str(data["reason"]), cost)
