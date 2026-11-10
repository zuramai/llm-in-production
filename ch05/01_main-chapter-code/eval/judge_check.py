"""How far can we trust the cheap judge? Re-grade saved answers with a stronger model and
count disagreements.

    cd ch05/01_main-chapter-code
    uv run python eval/judge_check.py eval/results/graph-openai_gpt-6-luna.json --limit 12

Writes eval/results/judge_check.json. The disagreements are printed so a human can decide who
was right (see the notebook).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from ch05_agentic.judge import judge
from ch05_agentic.llm import LLM

STRONG = "anthropic/claude-sonnet-5.5"


def main(paths: list[str], limit: int | None = None) -> None:
    # Sonnet 5.5 on OpenRouter rejects reasoning_effort="none" (400: "Reasoning is mandatory for
    # this endpoint"), so it gets "low" and room for its reasoning tokens.
    strong = LLM(STRONG, max_tokens=2000, reasoning_effort="low")
    rows, cost = [], 0.0
    for path in paths:
        data = json.loads(Path(path).read_text())
        for r in data["rows"][:limit]:
            v = judge(r["result"]["question"], r["gold"], r["result"]["answer"], data["run_at"],
                      strong)
            cost += v["judge_cost_usd"]
            rows.append({"file": Path(path).name, "id": r["id"], "cheap": r["judge"]["pass"],
                         "strong": v["pass"], "cheap_reason": r["judge"]["reason"],
                         "strong_reason": v["reason"], "answer": r["result"]["answer"],
                         "gold": r["gold"]})
    agree = sum(r["cheap"] == r["strong"] for r in rows)
    out = {"strong_model": STRONG, "n": len(rows), "agree": agree,
           "agreement": round(100 * agree / len(rows), 1), "strong_cost_usd": round(cost, 6),
           "rows": rows}
    (HERE / "results" / "judge_check.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"agreement {agree}/{len(rows)} = {out['agreement']}%, cost ${cost:.4f}")
    for r in rows:
        if r["cheap"] != r["strong"]:
            print(f"\n{r['file']} {r['id']}: haiku={r['cheap']} sonnet={r['strong']}")
            print("  haiku: ", r["cheap_reason"][:200])
            print("  sonnet:", r["strong_reason"][:200])


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--limit", type=int, help="first N answers per file (Sonnet isn't cheap)")
    args = ap.parse_args()
    main(args.paths, args.limit)
