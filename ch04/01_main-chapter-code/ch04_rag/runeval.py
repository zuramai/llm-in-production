"""Run the retrieval and answer evals. Used by the notebook; results are cached as JSON in
eval/results/ so re-executing the notebook doesn't re-spend money."""

from __future__ import annotations

import json
import statistics
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from pathlib import Path

from .answer import generate
from .evalset import Question, recall_at_k, reciprocal_rank
from .judge import JUDGE_MODEL, NOT_IN_DOCS, judge
from .retrieve import MODES, Retriever
from .store import load_docs

RESULTS = Path(__file__).resolve().parents[1] / "eval" / "results"
CHUNKERS = ("fixed", "heading")
CONFIGS = list(product(MODES, CHUNKERS))


def pct(x: float) -> float:
    return round(100 * x, 1)


def retrieval_eval(retriever: Retriever, questions: list[Question], k: int = 10) -> dict:
    """{config: {per-question recall@5, rr@10, ms}} for answerable questions."""
    out: dict = {}
    for mode, chunker in CONFIGS:
        rows = []
        for q in questions:
            if not q.answerable:
                continue
            t0 = time.perf_counter()
            hits = retriever.retrieve(q.question, mode, chunker, k)
            ms = (time.perf_counter() - t0) * 1000
            rows.append({
                "id": q.id, "category": q.category,
                "recall5": recall_at_k(q, hits, 5), "rr": reciprocal_rank(q, hits, k),
                "ms": ms, "top5": [h.citation for h in hits[:5]],
            })
        out[f"{mode}|{chunker}"] = rows
    return out


def summarize_retrieval(res: dict) -> list[dict]:
    table = []
    for cfg, rows in res.items():
        mode, chunker = cfg.split("|")
        cats = defaultdict(list)
        for r in rows:
            cats[r["category"]].append(r["recall5"])
        table.append({
            "mode": mode, "chunker": chunker,
            "recall5": pct(statistics.mean(r["recall5"] for r in rows)),
            "mrr": round(statistics.mean(r["rr"] for r in rows), 3),
            **{f"recall5_{c}": pct(statistics.mean(v)) for c, v in sorted(cats.items())},
        })
    return table


def full_corpus_context() -> str:
    return "\n\n---\n\n".join(f"[doc:{p}#0]\n{t}" for p, t in load_docs().items())


def _answer_one(q: Question, hits, model: str | None, context: str | None) -> dict:
    ans = generate(q.question, hits, model=model, context=context)
    v = judge(q.question, q.reference, ans.context if context is None else "(whole corpus)",
              ans.text)
    return {
        "id": q.id, "category": q.category, "answer": ans.text, "refused": ans.refused,
        "cites": [f"{p}#{i}" for p, i in ans.citations],
        "invalid_cites": [f"{p}#{i}" for p, i in ans.invalid_citations],
        "correct": v.correct, "faithful": None if context is not None else v.faithful,
        "reason": v.reason, "latency_ms": ans.latency_ms,
        "prompt_tokens": ans.prompt_tokens, "completion_tokens": ans.completion_tokens,
        "cost_usd": ans.cost_usd + v.cost_usd, "gen_cost_usd": ans.cost_usd,
        "retrieved": [h.citation for h in hits],
    }


def answer_eval(
    retriever: Retriever, questions: list[Question], cache: Path, refresh: bool = False,
    workers: int = 6, model: str | None = None,
) -> dict:
    """Answer all 40 questions under each retrieval config plus a whole-corpus baseline."""
    if cache.exists() and not refresh:
        return json.loads(cache.read_text())
    out: dict = {"judge": JUDGE_MODEL, "configs": {}}
    jobs = []
    for mode, chunker in CONFIGS:
        for q in questions:
            hits = retriever.retrieve(q.question, mode, chunker, 5)
            jobs.append((f"{mode}|{chunker}", q, hits, None))
    ctx = full_corpus_context()
    jobs += [("whole_corpus", q, [], ctx) for q in questions]
    with ThreadPoolExecutor(workers) as ex:
        futs = [(name, ex.submit(_answer_one, q, hits, model, c)) for name, q, hits, c in jobs]
        for name, f in futs:
            out["configs"].setdefault(name, []).append(f.result())
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out, indent=1))
    return out


def summarize_answers(res: dict, questions: list[Question]) -> list[dict]:
    answerable = {q.id for q in questions if q.answerable}
    table = []
    for cfg, rows in res["configs"].items():
        ans = [r for r in rows if r["id"] in answerable]
        unans = [r for r in rows if r["id"] not in answerable]
        faith = [r["faithful"] for r in rows if r["faithful"] is not None]
        cited = [r for r in ans if not r["refused"]]
        table.append({
            "config": cfg,
            "correct_answerable": pct(sum(r["correct"] for r in ans) / len(ans)),
            "refused_unanswerable": pct(sum(r["correct"] for r in unans) / len(unans)),
            "false_refusal": pct(sum(r["refused"] for r in ans) / len(ans)),
            "faithful": pct(sum(faith) / len(faith)) if faith else None,
            "bad_citations": sum(bool(r["invalid_cites"]) for r in rows),
            "uncited": sum(not r["cites"] for r in cited),
            "latency_p50_ms": int(statistics.median(r["latency_ms"] for r in rows)),
            "prompt_tokens": int(statistics.mean(r["prompt_tokens"] for r in rows)),
            "usd_per_1k": round(1000 * statistics.mean(r["gen_cost_usd"] for r in rows), 3),
        })
    return table
