"""Run the Chapter 13 attack and benign suites and save results to eval/results/.

    cd ch13/01_main-chapter-code
    uv run python eval/run_eval.py suite --defenses none --reps 3
    uv run python eval/run_eval.py suite --defenses all --reps 3
    uv run python eval/run_eval.py suite --defenses url_sanitizer --cases attacks
    uv run python eval/run_eval.py classifiers --guard llama-guard3:1b
    # Local target model (Ollama tag, no '/'); the refusal judge stays on the hosted LLM_MODEL:
    uv run python eval/run_eval.py suite --model qwen3:8b --defenses none --reps 3
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ch13_guards as g
import ch13_harness as H
import ch13_setup

LAYERS = ("privilege_separation", "tool_allowlist", "pii_redaction", "url_sanitizer")


def defenses_from(name: str, guard: str) -> H.Defenses:
    if name == "none":
        return H.NONE
    if name == "all":
        return replace(H.ALL, classifier=guard)
    if name == "structural":  # every layer except the input classifier
        return replace(H.ALL, classifier=None)
    if name == "classifier":
        return H.Defenses(classifier=guard)
    if name in LAYERS:
        return H.Defenses(**{name: True})
    raise SystemExit(f"unknown --defenses {name}")


def slug(text: str) -> str:
    return text.replace("/", "_").replace(":", "-")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("suite")
    s.add_argument("--defenses", default="none")
    s.add_argument("--guard", default="llama-guard3:1b")
    s.add_argument("--cases", choices=["attacks", "benign", "both"], default="both")
    s.add_argument("--reps", type=int, default=1)
    s.add_argument("--model", default=None)
    s.add_argument("--workers", type=int, default=8)
    c = sub.add_parser("classifiers")
    c.add_argument("--guard", required=True)
    c.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    ctx = ch13_setup.setup()
    attacks = H.load_cases(H.HERE / "attacks.yaml", ctx)
    benign = H.load_cases(H.HERE / "benign.yaml", ctx)

    if args.cmd == "classifiers":
        # Every user message, plus the planted texts an indirect attack would smuggle in.
        items = [(c["id"], "attack", c["prompt"]) for c in attacks]
        items += [(c["id"], "benign", c["prompt"]) for c in benign]
        items += [(f"T{t['id']}", "planted", t["body"]) for t in ch13_setup.planted_tickets(ctx)]
        items += [(f"D:{k}", "planted", v) for k, v in ch13_setup.planted_docs(ctx).items()]
        g.classify_input("warm-up", args.guard)  # load the model before timing anything

        def one(item):
            v = g.classify_input(item[2], args.guard)
            return {"id": item[0], "kind": item[1], "flagged": v.flagged,
                    "category": v.category, "latency_ms": v.latency_ms, "cost_usd": v.cost_usd}

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            rows = list(pool.map(one, items))
        path = H.save(rows, f"classifier-{slug(args.guard)}")
        for kind in ("attack", "benign", "planted"):
            sel = [r for r in rows if r["kind"] == kind]
            print(f"{kind:8s} flagged {sum(r['flagged'] for r in sel)}/{len(sel)}")
        print("saved", path)
        return

    d = defenses_from(args.defenses, args.guard)
    cases = {"attacks": attacks, "benign": benign, "both": attacks + benign}[args.cases]
    started = time.time()
    rows = H.run_suite(cases, d, ctx, model=args.model, reps=args.reps, workers=args.workers,
                       judge=True)
    model = slug(args.model or H.get_settings().llm_model)
    name = args.defenses
    if d.classifier and args.guard != "llama-guard3:1b":
        name += f"-{slug(args.guard)}"
    path = H.save(rows, f"suite-{name}-{args.cases}-{model}")
    atk = [r for r in rows if r["success"] is not None]
    ben = [r for r in rows if r["success"] is None]
    if atk:
        print(f"attack success: {sum(r['success'] for r in atk)}/{len(atk)}")
    if ben:
        print(f"benign refused: {sum(r['refused'] for r in ben)}/{len(ben)}")
    print(f"errors: {sum(bool(r['error']) for r in rows)} | cost ${sum(r['cost_usd'] for r in rows):.4f}"
          f" | {time.time() - started:.0f} s | saved {path}")


if __name__ == "__main__":
    main()
