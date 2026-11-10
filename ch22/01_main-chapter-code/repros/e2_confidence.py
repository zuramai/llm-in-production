"""E2: the model's self-reported confidence doesn't separate its right answers from wrong ones."""

from __future__ import annotations

import datetime as dt
import json
import random
from concurrent.futures import ThreadPoolExecutor

from ch22_common import LUNA, Result, chat, timed

ID, TITLE, LAYER, KIND = "E2", "Self-reported confidence is uncalibrated", "Evals", "hosted"

SCHEMA = {"type": "json_schema", "json_schema": {"name": "a", "strict": True, "schema": {
    "type": "object", "additionalProperties": False,
    "properties": {"answer": {"type": "string"}, "confidence": {"type": "number"}},
    "required": ["answer", "confidence"]}}}
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def questions(n: int = 20, seed: int = 22) -> list[tuple[str, str]]:
    """Questions with checkable answers that a no-reasoning model sometimes gets wrong."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        if i % 2:
            a, b = rng.randint(1000, 9999), rng.randint(1000, 9999)
            out.append((f"What is {a} * {b}? Answer with digits only.", str(a * b)))
        else:
            d = dt.date(1990, 1, 1) + dt.timedelta(days=rng.randint(0, 18000))
            out.append((f"What day of the week was {d:%B %d, %Y}? Answer with the day name.",
                        DAYS[d.weekday()]))
    return out


def correct(answer: str, gold: str) -> bool:
    return answer.replace(",", "").strip().strip(".").lower() == gold.lower()


def ask(q: str) -> tuple[str, float]:
    r = chat(LUNA, [{"role": "user", "content": q + " Also give your confidence from 0 to 1 that "
                     "your answer is correct."}],
             max_tokens=60, response_format=SCHEMA, reasoning_effort="none")
    d = json.loads(r.choices[0].message.content)
    return d["answer"], float(d["confidence"])


def agreement(samples: list[str]) -> float:
    """Self-consistency: share of samples that agree with the most common answer."""
    norm = [s.replace(",", "").strip().lower() for s in samples]
    return max(norm.count(x) for x in set(norm)) / len(norm)


@timed
def run(k: int = 3) -> Result:
    qs = questions()
    with ThreadPoolExecutor(8) as pool:
        samples = list(pool.map(lambda q: [ask(q[0]) for _ in range(k)], qs))
    rows = []
    for (q, gold), s in zip(qs, samples, strict=True):
        answer, conf = s[0]
        rows.append({"q": q, "gold": gold, "answer": answer, "confidence": conf,
                     "ok": correct(answer, gold), "agreement": agreement([a for a, _ in s])})
    right = [r for r in rows if r["ok"]]
    wrong = [r for r in rows if not r["ok"]]

    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    conf_gap = mean([r["confidence"] for r in right]) - mean([r["confidence"] for r in wrong])
    agree_gap = mean([r["agreement"] for r in right]) - mean([r["agreement"] for r in wrong])
    wrong_hi = sum(r["confidence"] >= 0.9 for r in wrong)
    return Result(
        ID, TITLE, LAYER, KIND,
        reproduced=bool(wrong) and wrong_hi / len(wrong) >= 0.5,
        fixed=bool(wrong) and agree_gap > conf_gap,
        evidence=f"luna, {len(rows)} checkable questions: {len(right)} right, {len(wrong)} wrong; "
                 f"mean confidence right {mean([r['confidence'] for r in right]):.2f} vs wrong "
                 f"{mean([r['confidence'] for r in wrong]):.2f} ({wrong_hi}/{len(wrong)} wrong "
                 f"at >=0.9); {k}-sample agreement right "
                 f"{mean([r['agreement'] for r in right]):.2f} vs wrong "
                 f"{mean([r['agreement'] for r in wrong]):.2f}",
        data={"rows": rows},
    )
