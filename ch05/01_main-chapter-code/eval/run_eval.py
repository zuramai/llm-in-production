"""Run the 25 hard questions through one or more variants on one model, judge, and save.

    cd ch05/01_main-chapter-code
    uv run python -m ch05_agentic.evalset                     # questions for today's data
    uv run python eval/run_eval.py --model openai/gpt-6-luna --variants ch04_rag tool_loop graph
    uv run python eval/run_eval.py --model openai/gpt-6-luna --holdout     # the held-out set
    uv run python eval/run_eval.py --model qwen3:8b --variants graph --limit 5

Results go to eval/results/<variant>-<model>.json. Questions run one at a time, so latency is
what one user would see. The judge is anthropic/claude-haiku-5.5 (see ch05_agentic/judge.py).
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from ch05_agentic import evalset
from ch05_agentic.judge import JUDGE_MODEL, judge
from ch05_agentic.llm import LLM
from ch05_agentic.tools import DocSearch, now_text
from ch05_agentic.variants import make_runner

RESULTS = HERE / "results"


def pctl(values: list[float], p: float) -> float:
    """Nearest-rank percentile (no interpolation: with 25 samples, p95 is the 24th value)."""
    s = sorted(values)
    return s[max(1, math.ceil(p / 100 * len(s))) - 1]


def summarize(rows: list[dict], variant: str, model: str) -> dict:
    lat = [r["result"]["latency_ms"] for r in rows]
    cost = [r["result"]["usage"]["cost_usd"] for r in rows]
    calls = [r["result"]["usage"]["calls"] for r in rows]
    n = len(rows)
    return {
        "variant": variant, "model": model, "n": n,
        "accuracy": round(100 * sum(r["judge"]["pass"] for r in rows) / n, 1),
        "facts_ok": round(100 * sum(sum(r["judge"]["facts_ok"]) for r in rows)
                          / sum(len(r["judge"]["facts_ok"]) for r in rows), 1),
        "gold_doc_retrieved": round(100 * sum(r["gold_doc_retrieved"] for r in rows) / n, 1),
        "cites_doc_and_sql": round(100 * sum(r["result"]["citations"].get("cites_doc", False)
                                             and r["result"]["citations"].get("cites_sql", False)
                                             for r in rows) / n, 1),
        "invalid_citations": sum(len(r["result"]["citations"].get("invalid", [])) for r in rows),
        "sql_errors": sum(r["result"]["sql_errors"] for r in rows),
        "errors": sum(r["result"]["error"] is not None for r in rows),
        "p50_ms": round(statistics.median(lat)), "p95_ms": round(pctl(lat, 95)),
        "mean_llm_calls": round(statistics.mean(calls), 2),
        "usd_per_1k": round(1000 * statistics.mean(cost), 4),
        "total_cost_usd": round(sum(cost), 6),
        "judge_cost_usd": round(sum(r["judge"]["judge_cost_usd"] for r in rows), 6),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--variants", nargs="+", default=["ch04_rag", "tool_loop", "graph"])
    ap.add_argument("--judge-model", default=JUDGE_MODEL)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--holdout", action="store_true",
                    help="use eval/questions_holdout.json (results get a -holdout suffix)")
    ap.add_argument("--tag", default="", help="suffix for the results file name")
    args = ap.parse_args()

    if args.holdout:
        built_at, questions = evalset.load(HERE / "questions_holdout.json")
        args.tag = "-holdout" + args.tag
    else:
        built_at, questions = evalset.load()
    if args.ids:
        questions = [q for q in questions if q.id in args.ids]
    questions = questions[: args.limit] if args.limit else questions
    today = now_text()
    if built_at[:10] not in today:
        print(f"WARNING: questions were built on {built_at[:10]}, today is {today}. "
              "Rebuild them: python -m ch05_agentic.evalset", file=sys.stderr)
    llm, judge_llm, docs = LLM(args.model), LLM(args.judge_model, max_tokens=400), DocSearch()

    for variant in args.variants:
        runner = make_runner(variant, llm, docs)
        rows = []
        for q in questions:
            res = runner(q.question, today=today).as_dict()
            verdict = judge(q.question, q.gold, res["answer"], today, judge_llm)
            gold_doc = any(c.startswith(f"[doc:{d}#") for c in res["evidence"] for d in q.docs)
            rows.append({"id": q.id, "template": q.template, "gold": q.gold, "result": res,
                         "judge": verdict, "gold_doc_retrieved": gold_doc})
            mark = "PASS" if verdict["pass"] else "fail"
            print(f"{variant:9} {q.id} {mark} {res['latency_ms']:>6} ms "
                  f"${res['usage']['cost_usd']:.5f} calls={res['usage']['calls']} "
                  f"{res['error'] or verdict['reason'][:90]}", flush=True)
        summary = summarize(rows, variant, args.model)
        print(json.dumps(summary))
        name = f"{variant}-{args.model.replace('/', '_').replace(':', '-')}{args.tag}.json"
        (RESULTS / name).write_text(json.dumps(
            {"summary": summary, "run_at": today, "questions_built_at": built_at,
             "judge_model": args.judge_model, "rows": rows}, indent=1, default=str) + "\n")


if __name__ == "__main__":
    main()
