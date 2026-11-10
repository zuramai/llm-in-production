"""Scoring for the consistency experiment (`eval/consistency.py`): same question, many runs.

Three decisions, made once here and used by the script, the notebook and the tests:

* **What a bot answered** is the number it *stated* in its reply (the first percentage for a rate,
  the first dollar amount for money, the number in front of the metric's label for a count).
  That's what a user reads. The first row of its SQL result is kept too (`row_value`), because the
  two can disagree: qwen3:8b once wrote "7.63%" over a column that integer division had made 0.
* **The reference** is the governed value: the semantic layer's own SQL (`compile_query`), run
  against the database with `now` set to the time of the run. Never a model's answer.
* **Equal** means equal at the precision the bots display: a rate within 0.01 percentage points
  (0.0001 as a fraction), money within a cent, a count exactly.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime

from .compile import compile_query
from .db import run_query
from .model import default_model
from .query import MetricQuery, TimeRange

# Decimals the bots display, per kind of metric (rates as fractions: 3.07% is 0.0307). Two answers
# are the same if they differ by at most one unit in the last displayed place (0.01 percentage
# points, one cent); counts must be equal.
DECIMALS = {"rate": 4, "money": 2, "count": 0}
TOLERANCE = {kind: 10.0 ** -d if d else 0.0 for kind, d in DECIMALS.items()}
_FLOAT_SLACK = 1e-9  # binary floats: 0.0307 - 0.0306 is slightly more than 0.0001

_NUMBER = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
_PERCENT_COLUMN = ("pct", "percent")


@dataclass(frozen=True)
class Question:
    query: MetricQuery    # the governed MetricQuery the question means
    label: str            # how answers name a count ("4,066 active customers")
    keywords: tuple[str, ...]  # substrings that identify the answer column in the bot's SQL rows

    @property
    def metric(self) -> str:
        return self.query.metrics[0]

    @property
    def kind(self) -> str:
        return metric_kind(self.metric)


def metric_kind(metric: str) -> str:
    """'rate', 'money' or 'count', from the metric's unit in metrics.yaml."""
    unit = default_model().metrics[metric].unit
    return {"ratio": "rate", "usd": "money", None: "count"}[unit]


QUESTIONS = {
    "What's our refund rate this month?": Question(
        MetricQuery(metrics=["refund_rate"], time_range=TimeRange(preset="this_month")),
        "refund rate", ("rate",)),
    "How many active customers do we have?": Question(
        MetricQuery(metrics=["active_customers"]), "active customers", ("active", "customer")),
    "What was net revenue last week?": Question(
        MetricQuery(metrics=["net_revenue"], time_range=TimeRange(preset="last_week")),
        "net revenue", ("net",)),
    "What was our average order value last month?": Question(
        MetricQuery(metrics=["aov"], time_range=TimeRange(preset="last_month")),
        "average order value", ("aov", "average", "avg")),
}


def _to_float(text: str) -> float:
    return float(text.replace(",", ""))


def stated_value(answer: str | None, q: Question) -> float | None:
    """The number the reply states (rates as fractions), or None if it states none."""
    if not answer:
        return None
    if q.kind == "rate":
        m = re.search(rf"({_NUMBER})\s*%", answer)
        return _to_float(m.group(1)) / 100 if m else None
    if q.kind == "money":
        m = re.search(rf"\$\s*({_NUMBER})", answer)
        return _to_float(m.group(1)) if m else None
    m = re.search(rf"({_NUMBER})\**\s+\**{re.escape(q.label)}", answer, re.IGNORECASE)
    return _to_float(m.group(1)) if m else None


def row_value(rows: list[dict] | dict | None, q: Question) -> float | None:
    """The first SQL row's value: the first numeric column named like the metric, else the last.

    A rate in a column named `*pct*` or `*percent*` is divided by 100.
    """
    row = rows[0] if isinstance(rows, list) and rows else rows
    if not isinstance(row, dict):
        return None
    numeric = [(k, v) for k, v in row.items()
               if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if not numeric:
        return None
    name, value = next(((k, v) for k, v in numeric
                        if any(word in k.lower() for word in q.keywords)), numeric[-1])
    if q.kind == "rate" and any(word in name.lower() for word in _PERCENT_COLUMN):
        return float(value) / 100
    return float(value)


def same_answer(a: float | None, b: float | None, kind: str) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= TOLERANCE[kind] + _FLOAT_SLACK


def distinct_answers(values: Iterable[float | None], kind: str) -> list[float | None]:
    """One representative per group of equal answers (None, no number, is an answer too)."""
    reps: list[float | None] = []
    for v in values:
        if not any(same_answer(v, r, kind) for r in reps):
            reps.append(v)
    return reps


def governed_value(q: Question, now: datetime,
                   run: Callable[[str], list[dict]] = run_query) -> float:
    """The semantic layer's answer at `now`: compile_query's SQL, run against the database."""
    rows = run(compile_query(q.query, now=now))
    return float(rows[0][q.metric])


def run_time(result: dict) -> datetime:
    """When a saved consistency run happened (`run_at`, UTC, e.g. 20261109T080302)."""
    return datetime.strptime(result["run_at"], "%Y%m%dT%H%M%S").replace(tzinfo=UTC)


def consistency_table(results: Iterable[dict],
                      run: Callable[[str], list[dict]] = run_query) -> list[dict]:
    """Pool saved runs per (model, question, variant) and score the stated answers.

    `results` are parsed `consistency-*.json` files. Each file's runs are compared with the
    governed value at that file's run time.
    """
    pooled: dict[tuple[str, str, str], dict] = {}
    for result in results:
        now = run_time(result)
        governed = {text: governed_value(q, now, run) for text, q in QUESTIONS.items()}
        for r in result["runs"]:
            q = QUESTIONS[r["question"]]
            cell = pooled.setdefault((result["model"], r["question"], r["variant"]),
                                     {"stated": [], "row": [], "governed": set()})
            cell["stated"].append(stated_value(r.get("answer"), q))
            cell["row"].append(row_value(r.get("row"), q))
            cell["governed"].add(governed[r["question"]])
    table = []
    for (model, question, variant), cell in pooled.items():
        kind = QUESTIONS[question].kind
        if len(cell["governed"]) != 1:
            raise ValueError(f"{model} / {question}: governed value changed between runs "
                             f"({sorted(cell['governed'])}); don't pool these files")
        (governed,) = cell["governed"]
        stated = cell["stated"]
        table.append({
            "model": model, "question": question, "variant": variant, "runs": len(stated),
            "governed": governed,
            "distinct": len(distinct_answers(stated, kind)),
            "matches governed": sum(same_answer(v, governed, kind) for v in stated),
            "row ≠ stated": sum(not same_answer(s, w, kind)
                                for s, w in zip(stated, cell["row"])),
            "answers": sorted({round(v, DECIMALS[kind]) for v in stated if v is not None}),
        })
    return table
