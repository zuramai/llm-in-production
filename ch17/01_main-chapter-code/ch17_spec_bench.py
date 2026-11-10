"""Decode speed with and without speculative decoding, on llama-server or vLLM.

Run the same script against a plain server and against one started with a draft model (or
n-gram / EAGLE-3 speculation), then compare:

    uv run python ch17_spec_bench.py --base-url http://localhost:8117 --label llama/baseline
    uv run python ch17_spec_bench.py --base-url http://localhost:8117 --label llama/draft-0.6b
    uv run python ch17_spec_bench.py --base-url http://localhost:8217 --model Qwen/Qwen3-8B \
        --label vllm/eagle3 --concurrency 1 8

Greedy decoding (temperature 0) means speculation must not change the text, up to floating-point
noise in batched kernels. We save the outputs so the notebook can check that claim instead of
assuming it. Acceptance comes from the server: llama-server's ``timings.draft_n`` /
``draft_n_accepted``, and vLLM's ``vllm:spec_decode_*`` Prometheus counters.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
from ch17_stream import counter_delta, prometheus_counters, stream_chat

HERE = Path(__file__).parent

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
    "code_edit": (
        "Here is a Python function. Return the same function with type hints added and nothing "
        "else changed.\n\n"
        "def order_total(items, shipping, coupon=None):\n"
        "    subtotal = sum(i['price'] * i['qty'] for i in items)\n"
        "    if coupon and coupon['kind'] == 'percent':\n"
        "        subtotal = subtotal * (100 - coupon['value']) / 100\n"
        "    elif coupon and coupon['kind'] == 'fixed':\n"
        "        subtotal = max(0, subtotal - coupon['value'])\n"
        "    free = subtotal >= 75\n"
        "    return round(subtotal + (0 if free else shipping), 2)\n"
    ),
    "freeform_1": "Explain in about 120 words why the sky is blue.",
    "freeform_2": "Write a short, friendly apology email to a customer whose parcel arrived late.",
    "freeform_3": "Give me 5 tips for packing fragile kitchenware for shipping, one sentence each.",
}


def body(model: str | None, prompt: str, max_tokens: int) -> dict:
    b = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "cache_prompt": False,  # llama-server: no prompt reuse between runs (vLLM ignores it)
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if model:
        b["model"] = model
    return b


def metrics(client: httpx.Client, base_url: str) -> dict[str, float]:
    try:
        r = client.get(f"{base_url}/metrics")
        return prometheus_counters(r.text) if r.status_code == 200 else {}
    except httpx.HTTPError:
        return {}


def run_one(client: httpx.Client, base_url: str, model: str | None, name: str, prompt: str,
            max_tokens: int) -> dict:
    r = stream_chat(client, base_url, body(model, prompt, max_tokens))
    t = r.timings
    return {
        "prompt": name,
        "text": r.text,
        "completion_tokens": r.completion_tokens,
        "ttft_ms": r.ttft_ms,
        "client_decode_tok_s": round(r.decode_tok_s or 0, 2),
        "server_decode_tok_s": round(t["predicted_per_second"], 2) if t else None,
        "draft_n": t.get("draft_n"),
        "draft_accepted": t.get("draft_n_accepted"),
    }


def run(base_url: str, model: str | None, max_tokens: int, concurrency: int) -> dict:
    with httpx.Client(timeout=600, limits=httpx.Limits(max_connections=concurrency + 4)) as c:
        run_one(c, base_url, model, "warmup", "hi", 8)  # CUDA graphs, page-in: not measured
        before = metrics(c, base_url)
        t0 = time.perf_counter()
        if concurrency == 1:
            rows = [run_one(c, base_url, model, n, p, max_tokens) for n, p in PROMPTS.items()]
        else:
            # every prompt `concurrency` times, all in flight together
            jobs = [(n, p) for n, p in PROMPTS.items() for _ in range(concurrency)]
            with ThreadPoolExecutor(concurrency) as pool:
                rows = list(pool.map(lambda j: run_one(c, base_url, model, *j, max_tokens), jobs))
        wall = time.perf_counter() - t0
        spec = {k: v for k, v in counter_delta(before, metrics(c, base_url)).items()
                if "spec_decode" in k}
    tokens = sum(r["completion_tokens"] or 0 for r in rows)
    return {
        "concurrency": concurrency,
        "median_stream_tok_s": round(statistics.median(r["client_decode_tok_s"] for r in rows), 1),
        "aggregate_tok_s": round(tokens / wall, 1),
        "wall_s": round(wall, 2),
        "vllm_spec_counters": spec,
        "rows": rows,
    }


def acceptance(result: dict) -> float | None:
    """Fraction of drafted tokens the target model accepted, from whichever server reported it."""
    drafted = sum(r["draft_n"] or 0 for r in result["rows"])
    accepted = sum(r["draft_accepted"] or 0 for r in result["rows"])
    if not drafted:
        c = result.get("vllm_spec_counters", {})
        drafted = c.get("vllm:spec_decode_num_draft_tokens_total", 0)
        accepted = c.get("vllm:spec_decode_num_accepted_tokens_total", 0)
    return round(accepted / drafted, 3) if drafted else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8117")
    ap.add_argument("--model")
    ap.add_argument("--label", required=True)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--concurrency", nargs="+", type=int, default=[1])
    ap.add_argument("--out", default=str(HERE / "results" / "spec.json"))
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    store = json.loads(out.read_text()) if out.exists() else {}
    store[args.label] = {}
    for conc in args.concurrency:
        res = run(args.base_url, args.model, args.max_tokens, conc)
        res["acceptance"] = acceptance(res)
        store[args.label][str(conc)] = res
        out.write_text(json.dumps(store, indent=2))
        print(f"{args.label:22s} c={conc:<3d} median stream {res['median_stream_tok_s']:6.1f} "
              f"tok/s  aggregate {res['aggregate_tok_s']:7.1f} tok/s  "
              f"acceptance {res['acceptance']}")


if __name__ == "__main__":
    main()
