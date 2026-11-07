"""Decode speed of a llama-server, with or without a speculative-decoding draft model.

Run once against a plain server and once against one started with a draft model, then compare:

    uv run python ch17_spec_bench.py --base-url http://localhost:8117 --label baseline
    uv run python ch17_spec_bench.py --base-url http://localhost:8117 --label draft-0.6b

Greedy decoding (temperature 0) means speculative decoding must not change the text. We save the
outputs so the notebook can check that claim instead of assuming it.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import httpx

TRIAGE = (
    'Classify the ticket and reply with JSON {{"category": ..., "priority": ..., '
    '"summary": ...}}. Categories: shipping, refund, product_defect, account, payment, other.\n'
    "Ticket: {t}"
)
PROMPTS = {
    "json_triage_1": TRIAGE.format(
        t="My Max Cutting Board (order #370) arrived cracked. Photos attached."
    ),
    "json_triage_2": TRIAGE.format(
        t="I was charged twice for order #1182 and nobody answers my emails."
    ),
    "json_triage_3": TRIAGE.format(
        t="Package says delivered but I never got it. Order #5521, urgent."
    ),
    "freeform_1": "Explain in about 120 words why the sky is blue.",
    "freeform_2": "Write a short, friendly apology email to a customer whose parcel arrived late.",
    "freeform_3": "Give me 5 tips for packing fragile kitchenware for shipping, one sentence each.",
}


def run(base_url: str, max_tokens: int) -> list[dict]:
    rows = []
    with httpx.Client(timeout=600) as client:
        # warm-up (graph/shader compilation, page-in), not measured
        client.post(
            f"{base_url}/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 8,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        ).raise_for_status()
        for name, prompt in PROMPTS.items():
            r = client.post(
                f"{base_url}/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": 0,
                    "cache_prompt": False,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            r.raise_for_status()
            d = r.json()
            t = d["timings"]
            rows.append(
                {
                    "prompt": name,
                    "text": d["choices"][0]["message"]["content"],
                    "completion_tokens": d["usage"]["completion_tokens"],
                    "decode_tok_s": round(t["predicted_per_second"], 2),
                    "draft_n": t.get("draft_n"),
                    "draft_accepted": t.get("draft_n_accepted"),
                }
            )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8117")
    ap.add_argument("--label", required=True)
    ap.add_argument("--max-tokens", type=int, default=160)
    ap.add_argument("--out", default="results/spec.json")
    args = ap.parse_args()
    rows = run(args.base_url, args.max_tokens)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    store = json.loads(out.read_text()) if out.exists() else {}
    store[args.label] = rows
    out.write_text(json.dumps(store, indent=2))
    speeds = [r["decode_tok_s"] for r in rows]
    print(f"{args.label}: median {statistics.median(speeds):.1f} tok/s over {len(rows)} prompts")
    for r in rows:
        acc = ""
        if r["draft_n"]:
            acc = f"  accepted {r['draft_accepted']}/{r['draft_n']} drafted"
        print(f"  {r['prompt']:14s} {r['decode_tok_s']:6.1f} tok/s{acc}")


if __name__ == "__main__":
    main()
