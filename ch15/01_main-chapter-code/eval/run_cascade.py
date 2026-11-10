"""Cascade eval: every strategy on the 40 questions, graded by `smart` against references + FAQ.

    cd ch15/01_main-chapter-code
    uv run python eval/run_cascade.py            # hosted strategies
    uv run python eval/run_cascade.py --local    # also local only / local -> smart (needs a GPU)

Writes eval/results/cascade.json: one row per (strategy, question) and the list of distinct calls.
Each distinct model call runs once and is replayed for every strategy that makes the same call,
so the eval pays for each call once; each strategy's cost and latency still count all its calls.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from ch15_gateway import gateway
from ch15_gateway.calls import Call, chat, plain_text_messages
from ch15_gateway.cascade import cascade
from ch15_gateway.judge import grade
from previous_chapters import FAQ_PATH, SYSTEM_PROMPT

RESULTS = HERE / "eval" / "results" / "cascade.json"
HOSTED = {
    "fast only": (["fast"], "fast"),
    "smart only": (["smart"], "smart"),
    "fast → smart (fast verifies)": (["fast", "smart"], "fast"),
    "fast → smart (fast-alt verifies)": (["fast", "smart"], "fast-alt"),
}
LOCAL = {
    "local only": (["local"], "local"),
    "local → smart (local verifies)": (["local", "smart"], "local"),
}


def run(local: bool = False, out: Path = RESULTS) -> dict:
    gw = gateway.client(timeout=900)
    questions = yaml.safe_load((HERE / "data" / "cascade_questions.yaml").read_text())
    faq = FAQ_PATH.read_text()
    memo: dict[tuple, Call] = {}

    def gw_chat(model, messages, max_tokens, response_format=None):
        key = (model, json.dumps(messages), max_tokens, json.dumps(response_format))
        if key not in memo:
            kw = {"response_format": response_format} if response_format else {}
            msgs = plain_text_messages(messages) if model.startswith("local") else messages
            memo[key] = chat(gw, model, msgs, max_tokens, extra_body=gateway.NO_CACHE, **kw)
        return memo[key]

    strategies = {**HOSTED, **(LOCAL if local else {})}
    rows, t0 = [], time.perf_counter()
    for q in questions:
        for name, (tiers, verifier) in strategies.items():
            r = cascade(gw_chat, q["q"], SYSTEM_PROMPT, faq, tiers, verifier)
            rows.append({"strategy": name, "id": q["id"], "kind": q["kind"], "answer": r.answer,
                         "answered_by": r.answered_by, "escalated": r.escalated,
                         "verdict": r.verdict.reason if r.verdict else "",
                         "cost_usd": r.cost_usd, "latency_s": r.latency_s, "calls": len(r.calls)})
    answer_calls = len(memo)
    seconds = time.perf_counter() - t0

    ref = {q["id"]: q for q in questions}
    grades: dict[tuple, tuple[bool, str]] = {}
    judge_cost = 0.0
    for row in rows:
        k = (row["id"], row["answer"])
        if k not in grades:
            q = ref[row["id"]]
            passed, reason, c = grade(gw_chat, "smart", faq, q["q"], q["ref"], row["answer"])
            grades[k] = (passed, reason)
            judge_cost += c.cost_usd
        row["pass"], row["judge_reason"] = grades[k]

    result = {
        "date": time.strftime("%Y-%m-%d"), "strategies": list(strategies),
        "answer_calls": answer_calls, "answer_seconds": seconds,
        "graded_answers": len(grades), "judge_cost_usd": judge_cost,
        "hosted_cost_usd": sum(c.cost_usd for c in memo.values()),
        "rows": rows,
        "calls": [{"model": k[0], **asdict(c)} for k, c in memo.items()],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--local", action="store_true", help="add the local strategies")
    args = p.parse_args()
    res = run(local=args.local)
    print(f"{len(res['rows'])} rows, {res['answer_calls']} answer/verify calls in "
          f"{res['answer_seconds']:.0f} s, {res['graded_answers']} answers graded, "
          f"total ${res['hosted_cost_usd']:.4f} (judge ${res['judge_cost_usd']:.4f}) -> {RESULTS}")
