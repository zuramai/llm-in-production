"""The Chapter 18 test set (210 tickets), against any OpenAI-compatible endpoint.

The request is exactly what a model fine-tuned on ``formats/openai/train.jsonl`` sees: the
one-line system prompt and the delimited ticket, plus a strict ``json_schema`` response format.
Run it on a base model to get the "before" number, and on the tuned model for the "after".

Results are cached in ``results/eval__<tag>.json``; delete the file (or pass ``force``) to re-run.
"""

from __future__ import annotations

import json
import time
from typing import Any

from ch19_ft import DATA, RESULTS, previous


def as_ticket(r: dict[str, Any]) -> Any:
    return previous().Ticket(r["id"], r["subject"], r["body"], r["channel"], r["category"],
                             r["priority"], r["order_id"], r["hard"])


def segments(test: list[dict[str, Any]], train: list[dict[str, Any]]) -> dict[str, set[int]]:
    """Test-ticket ids by segment, as Chapter 18's ``results/test_segments.json`` defines them.

    ``identical``: templated tickets whose exact text (strip + lower) is also in the training
    split; a model can answer them from memory, so they are reported separately.
    ``unseen``: the other templated tickets. ``hard``: the 60 hand-written tickets.
    ``headline`` = unseen + hard, the number every table leads with.
    """
    from ch19_ft.convert import leakage

    identical = set(leakage(train, test))
    templated = {r["id"] for r in test if not r["hard"]}
    hard = {r["id"] for r in test if r["hard"]}
    unseen = templated - identical
    return {"headline": unseen | hard, "unseen": unseen, "hard": hard,
            "identical": templated & identical, "all": {r["id"] for r in test}}


def segment_scores(results: list[Any], test: list[dict[str, Any]],
                   train: list[dict[str, Any]]) -> dict[str, Any]:
    p = previous()
    tickets = {r["id"]: as_ticket(r) for r in test}
    out = {}
    for name, ids in segments(test, train).items():
        part = [r for r in results if r.ticket_id in ids]
        if part:
            out[name] = p.score(part, tickets)
    return out


def request_options(model: str, price: tuple[float, float] | None = None,
                    extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Keyword arguments for ``run_batch``: the fine-tuning prompt, strict JSON, 300 tokens."""
    p = previous()
    opts: dict[str, Any] = {"rules": p.FT_RULES + p.FORMAT_HINT, "max_tokens": 300,
                            "price": price if price is not None else p.PRICES.get(model)}
    if extra:
        opts["extra"] = extra
    return opts


async def evaluate(client: Any, model: str, tag: str, *, limit: int | None = None,
                   concurrency: int = 8, price: tuple[float, float] | None = None,
                   extra: dict[str, Any] | None = None, force: bool = False) -> dict[str, Any]:
    p = previous()
    path = RESULTS / f"eval__{tag}.json"
    if path.exists() and not force:
        return json.loads(path.read_text())
    test = p.read_jsonl(DATA / "test.jsonl")[:limit]
    train = p.read_jsonl(DATA / "train.jsonl")
    t0 = time.perf_counter()
    results = await p.run_batch(client, [as_ticket(r) for r in test], "json_schema",
                                model=model, concurrency=concurrency,
                                **request_options(model, price, extra))
    wall = time.perf_counter() - t0
    out = {"tag": tag, "model": model, "prompt": "ft", "n": len(test),
           "concurrency": concurrency, "wall_seconds": round(wall, 2),
           "scores": segment_scores(results, test, train),
           "results": [r.to_json() for r in results]}
    RESULTS.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, indent=0, ensure_ascii=False))
    return out
