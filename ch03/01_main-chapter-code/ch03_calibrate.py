"""Judge calibration: run a judge over the human-labelled set and measure agreement."""

from __future__ import annotations

import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from ch03_evalkit import POINTWISE_RUBRIC, Judgment, judge_answer
from sklearn.metrics import cohen_kappa_score, confusion_matrix

HERE = Path(__file__).resolve().parent
CAL = HERE / "calibration"
SCORES = [1, 2, 3, 4, 5]


def load_calibration() -> tuple[list[dict], dict[str, int]]:
    answers = json.loads((CAL / "answers.json").read_text())
    labels = json.loads((CAL / "human_labels.json").read_text())["labels"]
    human = {k: v["score"] for k, v in labels.items()}
    assert [a["id"] for a in answers] == sorted(human), "answers and labels must cover the same ids"
    return answers, human


def run_judge(
    client,
    model: str,
    answers: list[dict],
    faq: str,
    rubric: str = POINTWISE_RUBRIC,
    workers: int = 8,
    **opts,
) -> dict[str, Judgment]:
    """Judge every answer once. Raises on a failed call: a missing score is not a score."""

    def one(a: dict) -> tuple[str, Judgment]:
        return a["id"], judge_answer(client, model, a["question"], a["answer"], faq, rubric, **opts)

    with ThreadPoolExecutor(workers) as pool:
        return dict(pool.map(one, answers))


def agreement(human: dict[str, int], judge: dict[str, int]) -> dict:
    """Agreement between two raters on the same items.

    Quadratic-weighted kappa is the headline: the 1-5 scale is ordinal, so judge=4 vs human=5
    is a much smaller disagreement than judge=1 vs human=5, and plain kappa treats them alike.
    """
    ids = sorted(human)
    h = [human[i] for i in ids]
    j = [judge[i] for i in ids]
    return {
        "n": len(ids),
        "exact": float(np.mean([a == b for a, b in zip(h, j, strict=True)])),
        "within1": float(np.mean([abs(a - b) <= 1 for a, b in zip(h, j, strict=True)])),
        "kappa": float(cohen_kappa_score(h, j, labels=SCORES)),
        "kappa_linear": float(cohen_kappa_score(h, j, labels=SCORES, weights="linear")),
        "kappa_quadratic": float(cohen_kappa_score(h, j, labels=SCORES, weights="quadratic")),
        "bias": float(np.mean(np.array(j) - np.array(h))),  # >0: judge is more lenient
        "confusion": confusion_matrix(h, j, labels=SCORES).tolist(),  # rows human, cols judge
    }


def save_run(name: str, judgments: dict[str, Judgment], extra: dict | None = None) -> None:
    payload = {
        "judge": name,
        "total_cost_usd": sum(j.cost_usd for j in judgments.values()),
        "judgments": {
            k: {"score": v.score, "reason": v.reason, "cost_usd": v.cost_usd}
            for k, v in judgments.items()
        },
        **(extra or {}),
    }
    (CAL / f"judge_{name}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def load_run(name: str) -> dict:
    return json.loads((CAL / f"judge_{name}.json").read_text())


def scores_of(run: dict) -> dict[str, int]:
    return {k: v["score"] for k, v in run["judgments"].items()}


Scorer = Callable[[dict], float]
