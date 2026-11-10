"""Ask the same metric questions N times and count how many different numbers come back.

    cd ch07/01_main-chapter-code
    uv run python eval/consistency.py --model openai/gpt-6-luna --repeats 5

A governed metric should give one number per question, every time. Free-form SQL can pick a
different definition on each run. Each run is scored against the semantic layer's own value
(`compile_query`, run against the database), using the number the bot stated in its reply; see
`ch07_semantic/consistency.py` for the matching rules. Results go to
eval/results/consistency-<model>-<timestamp>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import openai

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from ch07_semantic import agent as semantic_agent
from ch07_semantic.consistency import QUESTIONS, consistency_table, row_value, stated_value
from previous_chapters import sql_answer

from llmprod import chat_client


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True)
    p.add_argument("--provider", choices=["default", "ollama"], default="default")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--reasoning-effort")
    args = p.parse_args()
    client = (openai.OpenAI(base_url="http://localhost:11434/v1", api_key="ollama", timeout=600)
              if args.provider == "ollama" else chat_client(timeout=120))
    opts = {"model": args.model, "client": client, "reasoning_effort": args.reasoning_effort}

    def one(job: tuple[str, str]) -> dict:
        variant, question = job
        fn = semantic_agent.answer if variant == "semantic" else sql_answer
        try:
            res = fn(question, **opts)
        except Exception as exc:  # noqa: BLE001 — record, don't stop the experiment
            return {"variant": variant, "question": question, "stated": None, "value": None,
                    "error": f"{type(exc).__name__}: {exc}", "cost_usd": 0.0}
        q = QUESTIONS[question]
        return {"variant": variant, "question": question,
                "stated": stated_value(res.answer, q), "value": row_value(res.rows, q),
                "row": (res.rows or [None])[0], "sql": res.sql, "answer": res.answer,
                "cost_usd": res.usage.get("cost_usd", 0.0)}

    # The governed reference is computed for this moment, so stamp the start of the run.
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    jobs = [(v, q) for v in ("sql", "semantic") for q in QUESTIONS for _ in range(args.repeats)]
    workers = 1 if args.provider == "ollama" else 6
    with ThreadPoolExecutor(workers) as pool:
        runs = list(pool.map(one, jobs))

    result = {"model": args.model, "repeats": args.repeats, "run_at": stamp, "runs": runs,
              "cost_usd": sum(r["cost_usd"] for r in runs)}
    result["summary"] = consistency_table([result])
    for s in result["summary"]:
        print(f"{s['variant']:9} {s['question']:45} distinct={s['distinct']} "
              f"matches={s['matches governed']}/{s['runs']} governed={s['governed']} "
              f"{s['answers']}")
    print(f"cost ${result['cost_usd']:.4f}")
    slug = args.model.replace("/", "_").replace(":", "-")
    # One file per run, so repeated experiments accumulate instead of overwriting each other.
    (HERE / "results" / f"consistency-{slug}-{stamp}.json").write_text(
        json.dumps(result, indent=2, default=str, ensure_ascii=False))


if __name__ == "__main__":
    main()
