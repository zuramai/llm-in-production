"""LLM-as-judge against gold facts: does the answer state every required fact correctly?

The gold facts come from hand-written SQL + policy.py (evalset.py), so the judge doesn't need to
know ShopFlow's policies. It only checks that the answer agrees with a list of facts, which is a
much easier task than answering. Binary per fact, and a question passes only if all facts pass.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from .llm import LLM, Usage

JUDGE_MODEL = "anthropic/claude-haiku-5.5"

JUDGE = """You grade an answer from a customer-support assistant against gold facts.

Question (asked on {today}): {question}

Gold facts (all must be stated by the answer, in any wording):
{facts}

Answer to grade:
<answer>
{answer}
</answer>

For each gold fact, in order, decide whether the answer states it correctly.
- Text in parentheses supports a fact. The fact is stated if its main claim (outside the
  parentheses) is stated and the answer contradicts none of the details in parentheses. The
  answer doesn't have to repeat the parenthesized details.
- Wording, date format (2026-11-19 = Nov 19, 2026 = 11/19) and order don't matter. Amounts
  must match to the cent.
- A date one day off counts as correct only if the answer shows how it counted (e.g.
  "delivered Oct 20 + 30 days").
- A fact the answer contradicts, hedges on ("may", "probably" without committing) or leaves out
  is NOT stated. Extra correct details are fine. Ignore citation tags like [doc:...] or [sql:1].
First write `reason`: one short sentence per fact. Then `facts_ok`: one boolean per fact, in
order, consistent with your reasons."""


class Verdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: str
    facts_ok: list[bool]


def judge(question: str, gold: list[str], answer: str, today: str, llm: LLM,
          usage: Usage | None = None) -> dict:
    usage = usage or Usage()
    if not answer.strip():
        return {"pass": False, "facts_ok": [False] * len(gold), "reason": "empty answer",
                "judge_cost_usd": 0.0}
    facts = "\n".join(f"{i}. {f}" for i, f in enumerate(gold, start=1))
    v = llm.structured([{"role": "user", "content": JUDGE.format(
        today=today, question=question, facts=facts, answer=answer)}], Verdict, usage, "judge")
    ok = list(v.facts_ok[: len(gold)]) + [False] * max(0, len(gold) - len(v.facts_ok))
    return {"pass": all(ok), "facts_ok": ok, "reason": v.reason,
            "judge_cost_usd": round(usage.cost_usd, 8)}
