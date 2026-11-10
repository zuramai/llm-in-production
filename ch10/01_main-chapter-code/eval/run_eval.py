"""Run the 20-scenario trajectory eval for one model and variant.

    cd ch10/01_main-chapter-code
    uv run python eval/run_eval.py --model qwen3:8b --variant v1 --repeats 3
    uv run python eval/run_eval.py --model openai/gpt-6-luna --variant v0
    uv run python eval/run_eval.py --model openai/gpt-6-luna --compaction raw --only S05,S18,S19

Variants: v0 = no approval gate (issue_refund runs immediately), v1 = interrupt() gate.
Results go to eval/results/<model>-<variant>-<compaction>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import ch10_setup
from ch10_compaction import Compaction
from ch10_graph import AgentConfig, build_graph
from ch10_scenarios import SCENARIOS, pick_context, run_scenario, summarize


def evaluate(
    model: str,
    variant: str = "v1",
    compaction: str = "stable",
    repeats: int = 1,
    only: list[str] | None = None,
    verbose: bool = True,
) -> dict:
    ch10_setup.setup()
    ctx = pick_context()
    scenarios = [s for s in SCENARIOS if not only or s.id in only]
    cfg = AgentConfig(model=model, gate=variant != "v0", compaction=Compaction(mode=compaction))
    rows = []
    t0 = time.time()
    with ch10_setup.checkpointer() as saver:

        def make_graph():
            return build_graph(cfg, saver)

        for rep in range(repeats):
            for sc in scenarios:
                r = run_scenario(sc, ctx, make_graph)
                r["repeat"] = rep
                rows.append(r)
                if verbose:
                    mark = "PASS" if r["passed"] else "FAIL"
                    print(f"[{rep}] {sc.id} {mark} {r['latency_ms']/1000:5.1f}s "
                          f"{r['tools']} {r['detail']}", flush=True)  # fmt: skip
    return {
        "model": model,
        "variant": variant,
        "compaction": compaction,
        "repeats": repeats,
        "wall_s": round(time.time() - t0, 1),
        "summary": summarize(rows),
        "rows": rows,
    }


def out_path(model: str, variant: str, compaction: str, suffix: str = "") -> Path:
    name = f"{model.replace('/', '_').replace(':', '-')}-{variant}-{compaction}{suffix}.json"
    return HERE / "results" / name


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True)
    p.add_argument("--variant", choices=["v0", "v1"], default="v1")
    p.add_argument("--compaction", choices=["raw", "clip", "compact", "stable"], default="stable")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--only", default="", help="comma-separated scenario ids")
    p.add_argument("--suffix", default="", help="added to the result file name")
    args = p.parse_args()
    only = [s for s in args.only.split(",") if s] or None
    res = evaluate(args.model, args.variant, args.compaction, args.repeats, only)
    path = out_path(args.model, args.variant, args.compaction, args.suffix)
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res["summary"], indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
