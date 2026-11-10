"""Pairwise-judge experiments: position bias and verbosity bias. Writes pairwise/*.json.

Three pair sets over 12 FAQ questions, each judged in both orders by each judge model:
  gap      good bot vs the "no_faq" bot (a real quality gap on most questions)
  same     good bot vs a second independent sample of the SAME bot (no gap: any "win" is noise)
  padded   good answer vs the same answer padded 3x with filler (no new facts: pure verbosity)

    uv run python ch03/01_main-chapter-code/run_pairwise.py
"""

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ch03_evalkit import Dataset, judge_pair, position_bias
from ch03_faqbot import FAQ_TEXT, VARIANTS, make_bot, ollama_client

from llmprod import chat_client

HERE = Path(__file__).resolve().parent
OUT = HERE / "pairwise"
IDS = [
    "return-window", "refund-timing", "standard-shipping-cost", "free-shipping", "vip-threshold",
    "payment-methods", "failed-payment", "pending-expiry", "damaged", "support-hours", "carriers",
    "delete-account",
]  # fmt: skip
# name -> (model, local on Ollama?, extra request options)
JUDGES = {
    "luna": ("openai/gpt-6-luna", False, {}),
    "haiku": ("anthropic/claude-haiku-5.5", False, {}),
    # Local judges on the GPU (Ollama, free, 4k default context).
    "gptoss20b": ("gpt-oss:20b", True, {"max_tokens": 2000, "reasoning_effort": "low"}),
    # A small local model as a deliberately weak judge.
    "qwen3_1.7b": ("qwen3:1.7b", True, {"reasoning_effort": "none"}),
}
PAD_PROMPT = (
    "Rewrite the support answer below so it is about 3x longer. Add polite filler, restate the "
    "same facts in different words and add a friendly closing. Do NOT add any new fact, number, "
    "promise or link. Output only the rewritten answer.\n\n"
)


def build_pairs() -> list[dict]:
    qs = {c.id: c.input for c in Dataset.load(HERE / "faq_eval.yaml").cases if c.id in IDS}
    good, no_faq = make_bot(VARIANTS["good"]), make_bot(VARIANTS["no_faq"])
    padder = make_bot("You rewrite text exactly as instructed.")
    pairs = []
    with ThreadPoolExecutor(8) as pool:
        for qid in IDS:
            q = qs[qid]
            a1, a2, bad = pool.map(lambda f, q=q: f(q), [good, good, no_faq])
            padded = padder(PAD_PROMPT + a1)
            pairs += [
                {"set": "gap", "id": qid, "question": q, "x": a1, "y": bad},
                {"set": "same", "id": qid, "question": q, "x": a1, "y": a2},
                {"set": "padded", "id": qid, "question": q, "x": a1, "y": padded},
            ]
    return pairs


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    pairs_path = OUT / "pairs.json"
    if not pairs_path.exists():
        pairs_path.write_text(json.dumps(build_pairs(), indent=2, ensure_ascii=False))
    pairs = json.loads(pairs_path.read_text())
    results_path = OUT / "results.json"
    summary = json.loads(results_path.read_text()) if results_path.exists() else {}
    for jname in sys.argv[1:] or JUDGES:
        model, local, opts = JUDGES[jname]
        client = ollama_client() if local else chat_client(timeout=90)
        with ThreadPoolExecutor(2 if local else 8) as pool:
            outs = list(
                pool.map(
                    lambda p, model=model, opts=opts, client=client: judge_pair(
                        client, model, p["question"], p["x"], p["y"], FAQ_TEXT, **opts
                    ),
                    pairs,
                )
            )
        for p, o in zip(pairs, outs, strict=True):
            o.update(set=p["set"], id=p["id"])
        summary[jname] = outs
        print(jname, f"cost=${sum(o['cost_usd'] for o in outs):.4f}")
        for s in ("gap", "same", "padded"):
            sub = [o for o in outs if o["set"] == s]
            print(" ", s, position_bias(sub))
        results_path.write_text(json.dumps(summary, indent=2))
