"""Quantization experiment: the same model at q4 / q8 / fp16 on ShopFlow ticket triage.

For each Ollama tag we measure, on one fixed stratified sample of ``support_tickets``:
  accuracy      exact match on ``label_category``
  decode tok/s  from a long generation (Ollama's own ``eval_duration``)
  prefill tok/s from the classification prompts (``prompt_eval_duration``)
  memory        ``size`` of the loaded model from ``/api/ps``

    uv run python ch17_quant_bench.py --per-class 20 --out results/quant.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from pathlib import Path

import httpx

from llmprod import connect

OLLAMA = "http://localhost:11434"
TAGS = {
    "q4_K_M": "qwen3:1.7b",
    "q8_0": "qwen3:1.7b-q8_0",
    "fp16": "qwen3:1.7b-fp16",
}
LABELS = ["shipping", "refund", "product_defect", "account", "payment", "other"]
PROMPT = (
    "Classify this ShopFlow support ticket into exactly one category.\n"
    "Categories: shipping, refund, product_defect, account, payment, other.\n"
    "Reply with the category name only.\n\n"
    "Subject: {subject}\nBody: {body}"
)
LONG_GENERATION = "Count from 1 to 150, separated by commas. Output only the numbers."


def sample_tickets(per_class: int, seed: int = 17) -> list[dict]:
    """Stratified, seeded sample: ``per_class`` tickets of each label."""
    conn = connect(readonly=True)
    rows = conn.execute(
        "SELECT id, subject, body, label_category FROM support_tickets ORDER BY id"
    ).fetchall()
    rng = random.Random(seed)
    by_label: dict[str, list[dict]] = {}
    for tid, subject, body, label in rows:
        by_label.setdefault(label, []).append(
            {"id": tid, "subject": subject, "body": body, "label": label}
        )
    out: list[dict] = []
    for label in sorted(by_label):
        out += rng.sample(by_label[label], per_class)
    rng.shuffle(out)
    return out


def parse_label(text: str) -> str | None:
    """First known label mentioned in the reply, or None."""
    t = text.strip().lower().replace(" ", "_")
    hits = [(t.find(label), label) for label in LABELS if label in t]
    return min(hits)[1] if hits else None


def chat(client: httpx.Client, model: str, prompt: str, num_predict: int) -> dict:
    r = client.post(
        f"{OLLAMA}/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": False,
            "options": {"temperature": 0, "seed": 17, "num_predict": num_predict},
            "keep_alive": "10m",
        },
    )
    r.raise_for_status()
    return r.json()


def wilson(correct: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = correct / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return centre - half, centre + half


def bench_tag(name: str, model: str, tickets: list[dict]) -> dict:
    with httpx.Client(timeout=600) as client:
        chat(client, model, "hi", 4)  # load the model; not measured
        # decode speed: 3 long generations, median
        decode = []
        for _ in range(3):
            d = chat(client, model, LONG_GENERATION, 400)
            decode.append(d["eval_count"] / (d["eval_duration"] / 1e9))
        ps = client.get(f"{OLLAMA}/api/ps").json()["models"]
        loaded = next(m for m in ps if m["name"] == model)
        t0 = time.perf_counter()
        correct, unparsed, prefill_tok, prefill_ns = 0, 0, 0, 0
        wrong: list[dict] = []
        for t in tickets:
            d = chat(client, model, PROMPT.format(subject=t["subject"], body=t["body"]), 12)
            pred = parse_label(d["message"]["content"])
            unparsed += pred is None
            if pred == t["label"]:
                correct += 1
            elif len(wrong) < 5:
                wrong.append({"id": t["id"], "gold": t["label"], "pred": pred})
            prefill_tok += d["prompt_eval_count"]
            prefill_ns += d["prompt_eval_duration"]
        wall = time.perf_counter() - t0
        client.post(f"{OLLAMA}/api/generate", json={"model": model, "keep_alive": 0})
    lo, hi = wilson(correct, len(tickets))
    return {
        "quant": name,
        "model": model,
        "n": len(tickets),
        "correct": correct,
        "accuracy": round(100 * correct / len(tickets), 1),
        "ci95": [round(100 * lo, 1), round(100 * hi, 1)],
        "unparsed": unparsed,
        "decode_tok_s": round(statistics.median(decode), 1),
        "prefill_tok_s": round(prefill_tok / (prefill_ns / 1e9), 1),
        "loaded_size_gb": round(loaded["size"] / 1e9, 2),
        "wall_s": round(wall, 1),
        "example_errors": wrong,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=20)
    ap.add_argument("--out", default="results/quant.json")
    ap.add_argument("--only", nargs="*", choices=list(TAGS), help="run just these quants")
    args = ap.parse_args()
    tickets = sample_tickets(args.per_class)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    store = {r["quant"]: r for r in json.loads(out.read_text())} if out.exists() else {}
    results = []
    for name, model in TAGS.items():
        if args.only and name not in args.only:
            continue
        store[name] = bench_tag(name, model, tickets)
        results.append(store[name])
        out.write_text(json.dumps(list(store.values()), indent=2))
    for r in results:
        print(
            f"{r['quant']:8s} acc {r['accuracy']:5.1f}% (n={r['n']}, 95% CI {r['ci95']})  "
            f"decode {r['decode_tok_s']:6.1f} tok/s  prefill {r['prefill_tok_s']:7.1f} tok/s  "
            f"mem {r['loaded_size_gb']:.2f} GB"
        )


if __name__ == "__main__":
    main()
