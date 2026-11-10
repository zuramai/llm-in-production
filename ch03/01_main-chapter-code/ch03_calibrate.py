"""Judge calibration: run a judge over the reference-labelled set and measure agreement."""

from __future__ import annotations

import json
import random
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
    """Return the 40 answers and the reference score per id (calibration/reference_labels.json)."""
    answers = json.loads((CAL / "answers.json").read_text())
    labels = json.loads((CAL / "reference_labels.json").read_text())["labels"]
    reference = {k: v["score"] for k, v in labels.items()}
    assert [a["id"] for a in answers] == sorted(reference), "answers and labels must match"
    return answers, reference


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


def agreement(reference: dict[str, int], judge: dict[str, int]) -> dict:
    """Agreement between two raters on the same items.

    Quadratic-weighted kappa is the headline: the 1-5 scale is ordinal, so judge=4 vs reference=5
    is a much smaller disagreement than judge=1 vs reference=5, and plain kappa treats them alike.
    """
    ids = sorted(reference)
    r = [reference[i] for i in ids]
    j = [judge[i] for i in ids]
    return {
        "n": len(ids),
        "exact": float(np.mean([a == b for a, b in zip(r, j, strict=True)])),
        "within1": float(np.mean([abs(a - b) <= 1 for a, b in zip(r, j, strict=True)])),
        "kappa": float(cohen_kappa_score(r, j, labels=SCORES)),
        "kappa_linear": float(cohen_kappa_score(r, j, labels=SCORES, weights="linear")),
        "kappa_quadratic": float(cohen_kappa_score(r, j, labels=SCORES, weights="quadratic")),
        "bias": float(np.mean(np.array(j) - np.array(r))),  # >0: judge is more lenient
        "confusion": confusion_matrix(r, j, labels=SCORES).tolist(),  # rows reference, cols judge
    }


def blind_sheet(answers: list[dict], seed: int = 0) -> str:
    """A Markdown labelling sheet: answers in shuffled order, without variant names or scores.

    The labeller sees only the question, the answer and the rubric, so nothing hints at which
    answers were broken on purpose or what any judge said.
    """
    order = list(answers)
    random.Random(seed).shuffle(order)
    out = [
        "# Label the 40 calibration answers (blind)",
        "",
        (
            "Score each answer 1-5 using ONLY datasets/shopflow/docs/faq.md as ground truth."
            " Do not open judge_*.json, reference_labels.json or questions.yaml. Write"
            " `id,score,why` per answer, in this order."
        ),
        "",
        "## Rubric",
        "",
        POINTWISE_RUBRIC.strip(),
        "",
        "## Answers",
    ]
    for a in order:
        quoted = "\n".join("> " + line for line in a["answer"].splitlines())
        out += ["", f"### {a['id']}", f"**Customer:** {a['question']}", "", quoted]
    return "\n".join(out) + "\n"


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
