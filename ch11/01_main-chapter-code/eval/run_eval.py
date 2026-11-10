"""Evaluate the incident agent: 3 injected faults x N models x R repeats.

    uv run python eval/run_eval.py --models openai/gpt-6-luna --repeats 3 --out eval/results/x.json
    uv run python eval/run_eval.py --models gpt-oss:20b --repeats 3 --workers 1 --out ...
    uv run python eval/run_eval.py --rescore eval/results/x.json   # re-apply scoring.py

For each scenario: recreate Prometheus+Loki (clean history), run healthy traffic, inject the
fault, let the signals build, run every (model, repeat) against the same live incident,
score whether the injected root cause ranks first.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from ch11_agent.agent import run_agent
from ch11_agent.report import ranked, render_markdown
from ch11_agent.scoring import first_match
from stack import fresh_telemetry

ALERT = ("Checkout Overview alert: elevated errors and/or latency on service `checkout` "
         "(thresholds: 5xx ratio > 5% or p95 latency > 2s for 5 minutes). Find out why.")
SCENARIOS = ["db-pool-exhaustion", "bad-deploy", "payment-provider-timeout"]


def inject(scenario: str) -> dict:
    out = subprocess.run([sys.executable, str(HERE / "inject_fault.py"), "--scenario", scenario],
                         check=True, capture_output=True, text=True).stdout
    return json.loads(out)


def score(row: dict) -> int | None:
    if not row.get("hypotheses"):
        return None
    return first_match([h["hypothesis"] for h in row["hypotheses"]], row["scenario"],
                       bad_version=row.get("bad_version"))


def one_run(model: str, scenario: str, repeat: int, summarize: bool, bad_version: str) -> dict:
    res = run_agent(ALERT, model, summarize=summarize)
    row = {"scenario": scenario, "model": model, "repeat": repeat, "summarize": summarize,
           "bad_version": bad_version if scenario == "bad-deploy" else None,
           "steps": res.steps, "cost_usd": round(res.cost_usd, 5), "latency_s": round(res.latency_s, 1),
           "prompt_tokens": res.prompt_tokens, "completion_tokens": res.completion_tokens,
           "max_prompt_tokens": res.max_prompt_tokens, "tool_chars": res.tool_chars,
           "error": res.error, "rank": None, "top_hypothesis": None,
           "tools": [{"tool": t["tool"], "args": t["args"], "chars": t["chars"]} for t in res.trace]}
    if res.report:
        hyps = ranked(res.report)
        row["hypotheses"] = [{"hypothesis": h.hypothesis, "confidence": h.confidence} for h in hyps]
        row["rank"] = score(row)
        row["top_hypothesis"] = hyps[0].hypothesis if hyps else None
        slug = model.split("/")[-1].replace(":", "-")
        variant = "" if summarize else "-raw"
        path = HERE / "reports" / f"{scenario}-{slug}{variant}-r{repeat}.md"
        path.parent.mkdir(exist_ok=True)
        path.write_text(render_markdown(res.report, alert=ALERT, model=model, steps=res.steps,
                                        cost_usd=res.cost_usd))
        row["report"] = str(path.relative_to(HERE))
    return row


def rescore(path: Path) -> None:
    rows = json.loads(path.read_text())
    for row in rows:
        old, row["rank"] = row["rank"], score(row)
        if old != row["rank"]:
            print(f"  {row['scenario']} {row['model']} r{row['repeat']}: rank {old} -> {row['rank']}")
    path.write_text(json.dumps(rows, indent=1))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+")
    p.add_argument("--scenarios", nargs="+", default=SCENARIOS, choices=SCENARIOS)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--warmup", type=int, default=75, help="seconds of healthy traffic first")
    p.add_argument("--settle", type=int, default=100, help="seconds of fault before the agent runs")
    p.add_argument("--raw-tools", action="store_true", help="no summarizing/truncating (naive)")
    p.add_argument("--workers", type=int, default=4, help="concurrent runs; use 1 for Ollama")
    p.add_argument("--out", default="eval/results/agent-eval.json")
    p.add_argument("--rescore", type=Path, help="re-apply scoring to a results file and exit")
    a = p.parse_args()
    if a.rescore:
        rescore(a.rescore)
        return
    if not a.models:
        p.error("--models is required")

    load = subprocess.Popen([sys.executable, str(HERE / "loadgen.py"), "--rps", "6"],
                            stdout=subprocess.DEVNULL)
    rows = []
    try:
        for scenario in a.scenarios:
            print(f"== {scenario}: fresh telemetry, {a.warmup}s healthy, {a.settle}s fault", flush=True)
            inject("none")
            fresh_telemetry()
            time.sleep(a.warmup)
            state = inject(scenario)
            injected_at = datetime.now(UTC)
            time.sleep(a.settle)
            with ThreadPoolExecutor(max_workers=a.workers) as ex:
                futs = [ex.submit(one_run, m, scenario, r, not a.raw_tools, state["deploy_version"])
                        for m in a.models for r in range(1, a.repeats + 1)]
                for f in futs:
                    row = f.result()
                    row["injected_at"] = injected_at.strftime("%H:%M:%SZ")
                    rows.append(row)
                    print(f"  {row['model']:32s} r{row['repeat']} rank={row['rank']} "
                          f"steps={row['steps']} ${row['cost_usd']:.4f} err={row['error']}", flush=True)
    finally:
        inject("none")
        load.terminate()
    out = HERE / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
