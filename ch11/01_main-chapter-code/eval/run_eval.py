"""Evaluate the incident agent: 3 injected faults x N models x R repeats.

    uv run python eval/run_eval.py --models openai/gpt-6-luna anthropic/claude-haiku-5.5 --repeats 2

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

from ch11_agent.agent import run_agent  # noqa: E402
from ch11_agent.report import ranked, render_markdown  # noqa: E402
from ch11_agent.scoring import rank_of_root_cause  # noqa: E402
from stack import fresh_telemetry  # noqa: E402

ALERT = ("Checkout Overview alert: elevated errors and/or latency on service `checkout` "
         "(thresholds: 5xx ratio > 5% or p95 latency > 2s for 5 minutes). Find out why.")
SCENARIOS = ["db-pool-exhaustion", "bad-deploy", "payment-provider-timeout"]


def inject(scenario: str) -> None:
    subprocess.run([sys.executable, str(HERE / "inject_fault.py"), "--scenario", scenario],
                   check=True, capture_output=True)


def one_run(model: str, scenario: str, repeat: int, summarize: bool) -> dict:
    res = run_agent(ALERT, model, summarize=summarize)
    row = {"scenario": scenario, "model": model, "repeat": repeat, "summarize": summarize,
           "steps": res.steps, "cost_usd": round(res.cost_usd, 5), "latency_s": round(res.latency_s, 1),
           "prompt_tokens": res.prompt_tokens, "max_prompt_tokens": res.max_prompt_tokens,
           "tool_chars": res.tool_chars, "error": res.error, "rank": None, "top_hypothesis": None,
           "tools": [t["tool"] for t in res.trace]}
    if res.report:
        row["rank"] = rank_of_root_cause(res.report, scenario)
        row["top_hypothesis"] = ranked(res.report)[0].hypothesis
        slug = model.split("/")[-1]
        path = HERE / "reports" / f"{scenario}-{slug}-r{repeat}.md"
        path.write_text(render_markdown(res.report, alert=ALERT, model=model, steps=res.steps,
                                        cost_usd=res.cost_usd))
        row["report"] = str(path.relative_to(HERE))
    return row


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--scenarios", nargs="+", default=SCENARIOS, choices=SCENARIOS)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--warmup", type=int, default=75, help="seconds of healthy traffic first")
    p.add_argument("--settle", type=int, default=100, help="seconds of fault before the agent runs")
    p.add_argument("--raw-tools", action="store_true", help="no summarizing/truncating (naive)")
    p.add_argument("--out", default="eval/results/agent-eval.json")
    a = p.parse_args()

    load = subprocess.Popen([sys.executable, str(HERE / "loadgen.py"), "--rps", "6"],
                            stdout=subprocess.DEVNULL)
    rows = []
    try:
        for scenario in a.scenarios:
            print(f"== {scenario}: fresh telemetry, {a.warmup}s healthy, {a.settle}s fault", flush=True)
            inject("none")
            fresh_telemetry()
            time.sleep(a.warmup)
            inject(scenario)
            injected_at = datetime.now(UTC)
            time.sleep(a.settle)
            with ThreadPoolExecutor(max_workers=4) as ex:
                futs = [ex.submit(one_run, m, scenario, r, not a.raw_tools)
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
