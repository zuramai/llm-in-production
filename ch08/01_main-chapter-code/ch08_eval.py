"""Run an extractor over many files, score it against ground truth, and summarize.

The summary separates the four things that matter in accounts payable:
  correct          the whole invoice matches the truth
  caught           wrong, and the validation rules flagged it (a human will look)
  silent_wrong     wrong, and the rules did NOT flag it (this is the expensive one)
  false_alarm      right, but the rules flagged it (costs reviewer time only)
"""

from __future__ import annotations

import math
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
from ch08_extract import Extraction
from ch08_invoice import (
    SCORE_FIELDS,
    Invoice,
    Violation,
    score_invoice,
    validate_invoice,
)


@dataclass
class Result:
    file: str
    extraction: Extraction
    score: dict[str, float]
    violations: list[Violation]

    @property
    def correct(self) -> bool:
        return self.score["all"] == 1.0

    @property
    def flagged(self) -> bool:
        return bool(self.violations)


def judge(file: str, ex: Extraction, truth: Invoice, today: date | None = None) -> Result:
    violations = (validate_invoice(ex.invoice, today) if ex.invoice
                  else [Violation("parse_error", ex.error or "no invoice")])
    return Result(file, ex, score_invoice(ex.invoice, truth), violations)


def run_batch(files: list[Path], extract: Callable[[Path], Extraction],
              truth: dict[str, Invoice], workers: int = 8,
              today: date | None = None) -> list[Result]:
    def one(p: Path) -> Result:
        return judge(p.name, extract(p), truth[p.name], today)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, files))


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    return s[min(len(s) - 1, math.ceil(q * len(s)) - 1)]


def summarize(results: list[Result]) -> dict[str, float]:
    n = len(results)
    row: dict[str, float] = {"n": n}
    for f in SCORE_FIELDS:
        row[f] = 100 * sum(r.score[f] for r in results) / n
    row["line_item_recall"] = 100 * sum(r.score["line_item_recall"] for r in results) / n
    row["all_correct"] = 100 * sum(r.correct for r in results) / n
    row["parse_fail"] = 100 * sum(r.extraction.invoice is None for r in results) / n
    row["caught"] = sum(1 for r in results if not r.correct and r.flagged)
    row["silent_wrong"] = sum(1 for r in results if not r.correct and not r.flagged)
    row["false_alarm"] = sum(1 for r in results if r.correct and r.flagged)
    lat = [r.extraction.latency_s for r in results]
    row["p50_s"] = percentile(lat, 0.5)
    row["p95_s"] = percentile(lat, 0.95)
    row["usd_per_1k"] = 1000 * sum(r.extraction.cost_usd for r in results) / n
    return row


def summary_table(runs: dict[str, list[Result]]) -> pd.DataFrame:
    return pd.DataFrame({name: summarize(res) for name, res in runs.items()}).T
