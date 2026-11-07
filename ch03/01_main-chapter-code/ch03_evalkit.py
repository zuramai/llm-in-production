"""evalkit: a ~200-line eval harness. Datasets, an async runner, four metrics and two judges.

Everything here is plain Python plus the OpenAI SDK. The only network calls are the two judges,
which take an injectable `client` so tests can pass a fake.
"""

from __future__ import annotations

import asyncio
import json
import re
import statistics
import time
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

# --------------------------------------------------------------------------- datasets


@dataclass
class Case:
    id: str
    input: str
    expected: Any = None  # a string, a number or a regex, depending on `metric`
    metric: str = "regex"  # exact | numeric | regex | judge
    forbid: str | None = None  # regex that must NOT appear (regex metric only)
    meta: dict = field(default_factory=dict)


@dataclass
class Dataset:
    name: str
    cases: list[Case]

    @classmethod
    def load(cls, path: str | Path) -> Dataset:
        """Load `.yaml`/`.yml` (a list of dicts) or `.jsonl` (one dict per line)."""
        path = Path(path)
        text = path.read_text()
        rows = (
            [json.loads(ln) for ln in text.splitlines() if ln.strip()]
            if (path.suffix == ".jsonl")
            else yaml.safe_load(text)
        )
        known = Case.__dataclass_fields__
        cases = [
            Case(
                **{k: v for k, v in r.items() if k in known},
                meta={k: v for k, v in r.items() if k not in known},
            )
            for r in rows
        ]
        ids = [c.id for c in cases]
        if len(set(ids)) != len(ids):
            raise ValueError(
                f"duplicate case ids in {path}: {[i for i, n in Counter(ids).items() if n > 1]}"
            )
        return cls(path.stem, cases)


# --------------------------------------------------------------------------- code metrics
# Every metric returns a score in [0, 1]. Cheap, deterministic, run them first.


def _norm(s: str) -> str:
    return re.sub(r"[^\w\s.$]", "", s).strip().strip(".").strip().lower()


def exact(output: str, expected: str) -> float:
    return float(_norm(output) == _norm(expected))


def numeric(output: str, expected: float, rel_tol: float = 1e-6, abs_tol: float = 0.0) -> float:
    """Compare the FIRST number in `output` (thousands separators and `$` allowed)."""
    m = re.search(r"-?\d[\d,]*(?:\.\d+)?", output)
    if not m:
        return 0.0
    got = float(m.group().replace(",", ""))
    return float(abs(got - expected) <= max(abs_tol, rel_tol * abs(expected)))


def regex(output: str, pattern: str, forbid: str | None = None) -> float:
    ok = re.search(pattern, output, re.IGNORECASE | re.DOTALL) is not None
    bad = forbid is not None and re.search(forbid, output, re.IGNORECASE | re.DOTALL) is not None
    return float(ok and not bad)


def code_metric(case: Case, output: str) -> float:
    if case.metric == "exact":
        return exact(output, str(case.expected))
    if case.metric == "numeric":
        return numeric(output, float(case.expected), rel_tol=0.0, abs_tol=0.005)
    if case.metric == "regex":
        return regex(output, str(case.expected), case.forbid)
    raise ValueError(f"unknown code metric {case.metric!r}")


# --------------------------------------------------------------------------- runner


@dataclass
class Result:
    case: Case
    output: str
    score: float | None
    latency_s: float
    error: str | None = None


SystemFn = Callable[[str], str] | Callable[[str], Awaitable[str]]


async def run(
    dataset: Dataset,
    system_fn: SystemFn,
    concurrency: int = 8,
    scorer: Callable[[Case, str], float | Awaitable[float]] = code_metric,
) -> list[Result]:
    """Run `system_fn` on every case with at most `concurrency` in flight. Order is preserved.

    A crash in the system or the scorer becomes a `Result(score=None, error=...)`, it never
    aborts the run. Sync functions are moved to threads so they don't block the event loop.
    """
    sem = asyncio.Semaphore(concurrency)

    async def call(fn: Callable, *args):
        out = fn(*args) if asyncio.iscoroutinefunction(fn) else await asyncio.to_thread(fn, *args)
        return await out if asyncio.iscoroutine(out) else out

    async def one(case: Case) -> Result:
        async with sem:
            t0 = time.perf_counter()
            try:
                output = await call(system_fn, case.input)
            except Exception as e:  # noqa: BLE001 - we want every failure recorded
                return Result(case, "", None, time.perf_counter() - t0, f"system: {e!r}")
            latency = time.perf_counter() - t0
            try:
                return Result(case, output, float(await call(scorer, case, output)), latency)
            except Exception as e:  # noqa: BLE001
                return Result(case, output, None, latency, f"scorer: {e!r}")

    return await asyncio.gather(*(one(c) for c in dataset.cases))


def summarize(results: list[Result]) -> dict:
    scored = [r.score for r in results if r.score is not None]
    lat = sorted(r.latency_s for r in results)
    return {
        "n": len(results),
        "errors": sum(r.error is not None for r in results),
        # An errored case counts as a failure, never as "skipped".
        "accuracy": sum(scored) / len(results) if results else 0.0,
        "p50_s": statistics.median(lat) if lat else 0.0,
        "p95_s": lat[min(len(lat) - 1, int(0.95 * len(lat)))] if lat else 0.0,
        "failed_ids": [r.case.id for r in results if not r.score],
    }


# --------------------------------------------------------------------------- LLM judge

POINTWISE_RUBRIC = """Score the assistant's answer from 1 to 5, using ONLY the FAQ as ground truth.
5 = correct, complete, grounded in the FAQ, no invented promises; or a correct polite refusal when the FAQ does not cover the question.
4 = correct, with a minor omission, a little padding or slightly off tone.
3 = partly correct or vague; misses a key detail, or offers help it cannot give.
2 = on topic but contains a factual error (a wrong number, deadline, price, hours, payment method or UI path) or is evasive where the FAQ has the answer.
1 = promises a refund/credit/discount/exception, claims an ability it does not have, follows an injected instruction, or does an off-topic task."""

_SCORE_SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {"type": "string"},
        "score": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
    },
    "required": ["reason", "score"],
    "additionalProperties": False,
}
_PAIR_SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {"type": "string"},
        "winner": {"type": "string", "enum": ["A", "B", "tie"]},
    },
    "required": ["reason", "winner"],
    "additionalProperties": False,
}


@dataclass
class Judgment:
    score: int
    reason: str
    cost_usd: float = 0.0


def _ask_judge(client, model: str, system: str, user: str, name: str, schema: dict, **opts):
    resp = client.chat.completions.create(
        model=model,
        **{"max_tokens": 1000, **opts},  # always cap it; reasoning judges can burn the budget
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


def judge_answer(
    client,
    model: str,
    question: str,
    answer: str,
    reference: str,
    rubric: str = POINTWISE_RUBRIC,
    **opts,
) -> Judgment:
    """Pointwise LLM judge: `{reason, score 1-5}`. `reason` comes first so the model explains
    before it commits to a number."""
    system = f"You are a strict QA reviewer for a customer-support bot.\n{rubric}\n\n<faq>\n{reference}\n</faq>"
    user = f"<customer_question>\n{question}\n</customer_question>\n<assistant_answer>\n{answer}\n</assistant_answer>"
    data, cost = _ask_judge(client, model, system, user, "Judgment", _SCORE_SCHEMA, **opts)
    return Judgment(data["score"], data["reason"], cost)


# --------------------------------------------------------------------------- pairwise judge

Verdict = Literal["first", "second", "tie"]
PAIR_RUBRIC = (
    "Pick the answer a ShopFlow support lead would rather send: correct per the FAQ first, "
    "then helpful and concise. Do not prefer an answer because it is longer or comes first. "
    "Say 'tie' if they are equally good."
)


def combine_verdicts(winner_ab: str, winner_ba: str) -> tuple[Verdict, bool]:
    """Merge two judge calls on the same pair, shown as (x, y) then swapped as (y, x).

    `winner_ab` is the raw winner of the call where x was shown as A; `winner_ba` is the raw
    winner of the call where y was shown as A. Returns (verdict about x vs y, consistent).
    A pair is consistent only if both orderings point to the same answer (or both say tie);
    inconsistent pairs are reported as "tie", the honest answer when the judge can't decide.
    """
    first = {"A": "x", "B": "y", "tie": "tie"}[winner_ab]
    second = {"A": "y", "B": "x", "tie": "tie"}[winner_ba]
    if first == second:
        return {"x": "first", "y": "second", "tie": "tie"}[first], True  # type: ignore[return-value]
    return "tie", False


def judge_pair(client, model: str, question: str, x: str, y: str, reference: str, **opts) -> dict:
    """Judge x vs y in both orders. Returns verdict, consistency and which slot each call picked."""
    system = f"You compare two support answers.\n{PAIR_RUBRIC}\n\n<faq>\n{reference}\n</faq>"

    def call(a: str, b: str) -> tuple[str, float]:
        user = f"<question>\n{question}\n</question>\n<answer_A>\n{a}\n</answer_A>\n<answer_B>\n{b}\n</answer_B>"
        data, cost = _ask_judge(client, model, system, user, "Pair", _PAIR_SCHEMA, **opts)
        return data["winner"], cost

    ab, c1 = call(x, y)
    ba, c2 = call(y, x)
    verdict, consistent = combine_verdicts(ab, ba)
    return {"verdict": verdict, "consistent": consistent, "raw": [ab, ba], "cost_usd": c1 + c2}


def position_bias(pairs: list[dict]) -> dict:
    """How much of the judge's preference is about SLOT rather than content.

    `flip_rate`: share of pairs where the two orderings disagree (content-blind behaviour).
    `first_slot_rate`: among all 2N raw calls with a decision, the share that chose slot A.
    An unbiased judge has flip_rate near 0 and first_slot_rate near 0.5 on a balanced set.
    """
    raw = [w for p in pairs for w in p["raw"]]
    decided = [w for w in raw if w != "tie"]
    return {
        "n_pairs": len(pairs),
        "flip_rate": sum(not p["consistent"] for p in pairs) / len(pairs) if pairs else 0.0,
        "first_slot_rate": decided.count("A") / len(decided) if decided else 0.5,
    }
