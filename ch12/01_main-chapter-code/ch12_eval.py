"""Run the reviewer on the buggy PR and the clean PR and save everything it said.

    uv run python ch12_eval.py --model openai/gpt-6-luna --runs 3
    uv run python ch12_eval.py --model qwen3:8b --provider ollama --runs 1

The file in results/ keeps every comment *before* the confidence filter, so the notebook can
sweep the threshold without calling a model again.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from ch12_review.clients import make_client, model_options
from ch12_review.evaluate import score_run_file
from ch12_review.match import load_ground_truth
from ch12_review.reviewer import git_diff, review_diff

REPO = HERE / "sample_repo"
GT = HERE / "ground_truth.json"


def slug(s: str) -> str:
    return s.replace("/", "_").replace(":", "-")


def ensure_repo() -> None:
    if not (REPO / ".git").exists():
        subprocess.run(["bash", str(HERE / "make_sample_pr.sh")], check=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--provider", choices=["hosted", "ollama"], default="hosted")
    ap.add_argument("--granularity", choices=["raw", "hunk", "file", "diff"], default="hunk",
                    help="raw = naive v0: unnumbered git diff, naive prompt, one call")
    ap.add_argument("--no-guidelines", action="store_true", help="omit the repo conventions")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--min-confidence", type=float, default=0.6, help="for the printed summary")
    ap.add_argument("--out-dir", default=str(HERE / "results"))
    ap.add_argument("--timeout", type=float, default=120,
                    help="per-call timeout in seconds (raise it for CPU-only local models)")
    ap.add_argument("--reasoning-effort", default=None,
                    help="override reasoning_effort (e.g. none, low) for reasoning models")
    args = ap.parse_args()

    ensure_repo()
    gt = load_ground_truth(GT)
    diffs = {"buggy": git_diff(str(REPO), gt["base"], gt["buggy_head"]),
             "clean": git_diff(str(REPO), gt["base"], gt["clean_head"])}
    client = make_client(args.provider, timeout=args.timeout)
    options = model_options(args.model, args.provider)
    if args.reasoning_effort:
        options = {**options, "reasoning_effort": args.reasoning_effort}
    workers = 1 if args.provider == "ollama" else 4

    runs = []
    for i in range(args.runs):
        entry = {}
        for name, text in diffs.items():
            r = review_diff(text, client=client, model=args.model, granularity=args.granularity,
                            guidelines=not args.no_guidelines, options=options, workers=workers)
            entry[name] = {"comments": [c.model_dump() for c in r.comments],
                           "raw_comments": [c.model_dump() for c in r.raw],
                           "raw_count": r.raw_count, "n_units": r.n_units,
                           "latency_ms": r.latency_ms, "usage": r.usage, "errors": r.errors}
            print(f"run {i + 1} {name}: {r.raw_count} raw -> {len(r.comments)} deduped, "
                  f"{r.latency_ms} ms, ${r.usage['cost_usd']:.5f}, errors={len(r.errors)}")
        runs.append(entry)

    tag = (f"{slug(args.model)}-{args.granularity}" + ("-noguide" if args.no_guidelines else "")
           + (f"-effort-{args.reasoning_effort}" if args.reasoning_effort else ""))
    out = Path(args.out_dir) / f"{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": args.model, "provider": args.provider, "granularity": args.granularity,
        "guidelines": not args.no_guidelines, "options": options, "runs": runs,
    }, indent=1))
    s = score_run_file(runs, gt, min_confidence=args.min_confidence)
    total_cost = sum(e[k]["usage"]["cost_usd"] for e in runs for k in e)
    print(f"\n{tag} @ confidence>={args.min_confidence}: recall {s['recall']:.1%}, "
          f"precision {s['precision']:.1%}, FP/buggy PR {s['fp_per_buggy_pr']:.2f}, "
          f"FP/clean PR {s['fp_per_clean_pr']:.2f}, total ${total_cost:.4f} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
