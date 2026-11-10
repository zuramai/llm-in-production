"""Chapter 18: evaluate prompted and fine-tuned models on the same 210 test tickets.

Runs in the main venv and talks to any OpenAI-compatible endpoint:

    cd ch18/01_main-chapter-code
    # vLLM on :8218 serving Qwen/Qwen3-1.7B plus LoRA adapters (serve_vllm.sh)
    uv run python ch18_eval.py --endpoint vllm --model Qwen/Qwen3-1.7B --prompt ch02
    uv run python ch18_eval.py --endpoint vllm --model q17-lora --prompt ft
    uv run python ch18_eval.py --endpoint openrouter --model openai/gpt-6-luna --prompt ch02
    uv run python ch18_eval.py --endpoint ollama --model gpt-oss:20b --prompt ch02

Each run is cached in ``results/eval__<tag>.json`` with every reply, the score on all tickets,
the templated tickets and the hard tickets, and the wall-clock time of the batch.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from typing import Any

from ch18_data import (
    DATA,
    FORMAT_HINT,
    FT_RULES,
    HERE,
    as_ticket,
    format_ticket,
    previous,
    read_jsonl,
    text_key,
)

RESULTS = HERE / "results"
ENDPOINTS = {
    "vllm": {"base_url": "http://localhost:8218/v1", "api_key": "EMPTY"},
    "ollama": {"base_url": "http://localhost:11434/v1", "api_key": "ollama"},
    "openrouter": {},  # llmprod settings: LLM_BASE_URL / LLM_API_KEY from .env
}
PROMPTS = ("ch02", "fewshot", "ft")
# Qwen3 is a hybrid reasoning model; the chat template switch turns thinking off on vLLM.
# Greedy decoding (temperature 0) makes the small models' runs reproducible.
VLLM_EXTRA = {"temperature": 0,
              "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}


def fewshot_rules(train: list[dict[str, Any]],
                  held_out: list[dict[str, Any]] | None = None) -> str:
    """Chapter 2's rulebook plus one labelled training ticket per template present in train.

    This is the strongest prompt-only competitor to fine-tuning: the labelling policy is shown,
    not described. Only routing labels are shown; the examples come from the training split, and
    a row whose text also appears in ``held_out`` (val/test) is never used as an example.
    """
    banned = {text_key(r) for r in held_out or []}
    first: dict[int, dict[str, Any]] = {}
    for r in sorted(train, key=lambda r: r["id"]):
        if r["template_id"] is not None and text_key(r) not in banned:
            first.setdefault(r["template_id"], r)
    shots = "\n\n".join(
        format_ticket(r["subject"], r["body"], r["channel"]) + "\n-> " + json.dumps(
            {"category": r["category"], "priority": r["priority"], "order_id": r["order_id"]})
        for _, r in sorted(first.items())
    )
    return (previous().RULES + "\n\nExamples of tickets and ShopFlow's labels for them:\n\n"
            + shots)


def request_options(endpoint: str, model: str, prompt: str) -> dict[str, Any]:
    """Keyword arguments for Chapter 2's ``run_batch`` / ``triage_one``.

    ``prompt="ch02"`` is Chapter 2's full rulebook; ``"fewshot"`` adds one labelled training
    example per template (22 after dedupe);
    ``"ft"`` is the one-line instruction the adapters were trained with. All use strict
    ``json_schema`` (guided decoding on vLLM).
    """
    if prompt not in PROMPTS:
        raise ValueError(f"prompt must be one of {PROMPTS}, got {prompt!r}")
    opts: dict[str, Any] = {"max_tokens": 300}
    if prompt == "ft":
        opts["rules"] = FT_RULES + FORMAT_HINT
    elif prompt == "fewshot":
        held_out = read_jsonl(DATA / "val.jsonl") + read_jsonl(DATA / "test.jsonl")
        opts["rules"] = fewshot_rules(read_jsonl(DATA / "train.jsonl"), held_out)
    if endpoint == "vllm":
        opts["extra"] = VLLM_EXTRA
    elif endpoint == "ollama":
        if model.startswith("gpt-oss"):
            # gpt-oss always reasons ("low" is its minimum) and reasoning counts toward max_tokens
            opts.update(extra={"reasoning_effort": "low"}, max_tokens=1500)
        else:
            opts["extra"] = {"reasoning_effort": "none"}
    elif endpoint == "openrouter":
        opts["price"] = previous().PRICES.get(model)
    else:
        raise ValueError(f"unknown endpoint {endpoint!r}")
    return opts


def segments(records: list[dict[str, Any]], train: list[dict[str, Any]]) -> dict[str, set[int]]:
    """Test-ticket ids by segment, defined exactly as in Chapter 19's ``evaluate.segments``.

    ``seen``: templated tickets whose exact text (case-insensitive) is also in train: a model
    can memorise those. ``unseen``: templated tickets with new text. ``hard``: the 60
    hand-written tickets. ``headline`` = unseen + hard, the only tickets no model trained on.
    """
    in_train = {text_key(r) for r in train}
    templated = {r["id"] for r in records if not r["hard"]}
    seen = {r["id"] for r in records if not r["hard"] and text_key(r) in in_train}
    hard = {r["id"] for r in records if r["hard"]}
    return {"all": {r["id"] for r in records}, "seen": seen, "unseen": templated - seen,
            "hard": hard, "headline": (templated - seen) | hard}


def segment_scores(results: list[Any], records: list[dict[str, Any]],
                   train: list[dict[str, Any]]) -> dict[str, Any]:
    """Chapter 2's ``score`` on every segment of ``segments``."""
    p = previous()
    tickets = {r["id"]: as_ticket(r) for r in records}
    out = {}
    for name, ids in segments(records, train).items():
        part = [r for r in results if r.ticket_id in ids]
        if part:
            out[name] = p.score(part, tickets)
    return out


def local_cost_per_1k(tickets_per_second: float, gpu_usd_per_hour: float) -> float:
    """Self-hosted $ per 1,000 tickets: what the GPU costs while it triages them at this rate."""
    if tickets_per_second <= 0:
        raise ValueError("throughput must be positive")
    return gpu_usd_per_hour / 3600 / tickets_per_second * 1000


def tag_for(endpoint: str, model: str, prompt: str, concurrency: int) -> str:
    slug = model.replace("/", "_").replace(":", "_")
    return f"{endpoint}__{slug}__{prompt}__c{concurrency}"


async def run(endpoint: str, model: str, prompt: str, concurrency: int, split_file: str,
              limit: int | None = None, force: bool = False) -> dict[str, Any]:
    p = previous()
    records = read_jsonl(DATA / split_file)[:limit]
    tag = tag_for(endpoint, model, prompt, concurrency) + (f"__n{limit}" if limit else "")
    path = RESULTS / f"eval__{tag}.json"
    train = read_jsonl(DATA / "train.jsonl")
    if path.exists() and not force:
        # Cached replies are re-scored, so segment definitions can change without new calls.
        out = json.loads(path.read_text())
        out["scores"] = segment_scores([p.Result(**d) for d in out["results"]], records, train)
        path.write_text(json.dumps(out, indent=0, ensure_ascii=False))
        return out
    client = p.async_chat_client(timeout=300, max_retries=2, **ENDPOINTS[endpoint])
    t0 = time.perf_counter()
    results = await p.run_batch(client, [as_ticket(r) for r in records], "json_schema",
                                model=model, concurrency=concurrency,
                                **request_options(endpoint, model, prompt))
    wall = time.perf_counter() - t0
    out = {
        "tag": tag, "endpoint": endpoint, "model": model, "prompt": prompt,
        "concurrency": concurrency, "n": len(records), "wall_seconds": round(wall, 2),
        "tickets_per_second": round(len(records) / wall, 3),
        "scores": segment_scores(results, records, train),
        "results": [r.to_json() for r in results],
    }
    RESULTS.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, indent=0, ensure_ascii=False))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate one model on the ch18 test set")
    ap.add_argument("--endpoint", choices=list(ENDPOINTS), required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", choices=PROMPTS, required=True)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--split", default="test.jsonl")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--force", action="store_true", help="ignore the cached result")
    a = ap.parse_args()
    out = asyncio.run(run(a.endpoint, a.model, a.prompt, a.concurrency, a.split, a.limit,
                          a.force))
    s = out["scores"]
    print(f"{out['tag']}: {out['n']} tickets in {out['wall_seconds']} s "
          f"({out['tickets_per_second']}/s)")
    for seg in ("all", "headline", "unseen", "hard", "seen"):
        if seg in s:
            x = s[seg]
            print(f"  {seg:9} n={x['n']:3} valid={x['final_valid_%']}% "
                  f"cat={x['category_acc_%']}% pri={x['priority_acc_%']}% "
                  f"oid={x['order_id_acc_%']}% p50={x['latency_p50_ms']} ms "
                  f"p95={x['latency_p95_ms']} ms $/1k={x['cost_per_1k_usd']} "
                  f"errors={x['errors']}")


if __name__ == "__main__":
    main()
