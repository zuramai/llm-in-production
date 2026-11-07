"""Review a git diff and write the comments to a JSON file.

    uv run python review.py --repo sample_repo --base main --head feature/order-fixes \
        --model openai/gpt-6-luna --min-confidence 0.6 --out out/comments.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ch12_review.clients import default_model, make_client, model_options
from ch12_review.postprocess import filter_comments
from ch12_review.reviewer import git_diff, review_diff


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--repo", default=".")
    ap.add_argument("--base", default="origin/main")
    ap.add_argument("--head", default="HEAD")
    ap.add_argument("--model", default=None, help="default: LLM_MODEL from .env")
    ap.add_argument("--provider", choices=["hosted", "ollama"], default="hosted")
    ap.add_argument("--granularity", choices=["hunk", "file", "diff"], default="hunk")
    ap.add_argument("--min-confidence", type=float, default=0.6)
    ap.add_argument("--min-severity", default="low")
    ap.add_argument("--max-comments", type=int, default=10)
    ap.add_argument("--out", default="out/comments.json")
    args = ap.parse_args()

    model = args.model or default_model()
    run = review_diff(git_diff(args.repo, args.base, args.head), client=make_client(args.provider),
                      model=model, granularity=args.granularity,
                      options=model_options(model, args.provider),
                      workers=1 if args.provider == "ollama" else 4)
    kept = filter_comments(run.comments, args.min_confidence, args.min_severity,
                           args.max_comments)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([c.model_dump() for c in kept], indent=2))
    print(f"{model}: {run.raw_count} raw -> {len(run.comments)} deduped -> {len(kept)} posted "
          f"({run.n_units} calls, {run.latency_ms} ms, ${run.usage.get('cost_usd', 0):.5f})")
    for e in run.errors:
        print("  error:", e, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
