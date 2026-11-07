"""Prefix-cache benchmark for llama-server: TTFT with and without a shared long prefix.

Start the server first (see the chapter), then:

    uv run python ch17_prefix_bench.py --base-url http://localhost:8117 --out results/prefix.json

Three scenarios, same questions, same long ShopFlow knowledge-base prefix. Each scenario gets
its own session nonce at the very top of the prompt, so its first request is a guaranteed
cold start and nothing leaks in from the previous scenario:
  cache off     every request pays full prefill (``cache_prompt: false``)
  cache on      stable prefix first, question last: requests 2..N reuse the prefix KV
  cache on, bust a per-request timestamp at the very start of the prompt: nothing is shared
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[2]
DOCS = REPO / "datasets" / "shopflow" / "docs"

QUESTIONS = [
    "What is your refund window?",
    "Is shipping free for orders over a certain amount?",
    "How long do I have to return an item?",
    "Do VIP customers pay for shipping?",
    "When will my refund arrive?",
    "What are your support hours?",
    "Can I change my shipping address after ordering?",
    "How do I clean a cast-iron pan I bought from you?",
]


def build_prefix(max_chars: int = 12_000) -> str:
    """Concatenate ShopFlow docs (faq first) into one stable, long system prompt."""
    files = [DOCS / "faq.md"] + sorted(p for p in DOCS.rglob("*.md") if p.name != "faq.md")
    parts: list[str] = []
    used = 0
    for path in files:
        text = f"## {path.relative_to(DOCS)}\n{path.read_text().strip()}\n"
        if used + len(text) > max_chars:
            break
        parts.append(text)
        used += len(text)
    return "You are ShopFlow support. Answer only from these documents.\n\n" + "\n".join(parts)


def ttft_request(
    client: httpx.Client, base_url: str, system: str, question: str, cache_prompt: bool
) -> dict:
    """Stream one chat completion; return TTFT (ms) and llama-server's own prompt accounting."""
    body = {
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": question}],
        "max_tokens": 16,
        "temperature": 0,
        "stream": True,
        "cache_prompt": cache_prompt,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.perf_counter()
    ttft = None
    timings: dict = {}
    with client.stream("POST", f"{base_url}/v1/chat/completions", json=body) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line.startswith("data: ") or line.endswith("[DONE]"):
                continue
            chunk = json.loads(line[6:])
            delta = (chunk.get("choices") or [{}])[0].get("delta", {})
            if ttft is None and (delta.get("content") or delta.get("reasoning_content")):
                ttft = (time.perf_counter() - t0) * 1000
            if "timings" in chunk:
                timings = chunk["timings"]
    return {
        "ttft_ms": round(ttft or (time.perf_counter() - t0) * 1000, 1),
        "prompt_tokens_processed": timings.get("prompt_n"),
        "prompt_tokens_from_cache": timings.get("cache_n"),
        "prefill_ms": timings.get("prompt_ms"),
    }


def environment(base_url: str) -> dict:
    """What else was using this (shared) machine while we measured."""
    import os

    try:
        ps = httpx.get("http://localhost:11434/api/ps", timeout=5).json()["models"]
        ollama = [m["name"] for m in ps]
    except httpx.HTTPError:
        ollama = []
    return {"load_avg_1m": round(os.getloadavg()[0], 1), "other_ollama_models_loaded": ollama}


def run_scenario(
    base_url: str, name: str, prefix: str, cache_prompt: bool, bust: bool, n_questions: int
) -> dict:
    rows = []
    nonce = f"[session {time.time_ns()}]\n"
    with httpx.Client(timeout=300) as client:
        # Tiny request so the model is warm; it shares no prefix with the real prompts.
        ttft_request(client, base_url, "warm up", "hi", cache_prompt=False)
        for i, q in enumerate(QUESTIONS[:n_questions]):
            # bust: a fresh timestamp on EVERY request, at the start of the prompt
            head = f"[request-id {time.time_ns()}]\n" if bust else nonce
            row = ttft_request(client, base_url, head + prefix, q, cache_prompt)
            row["request"] = i + 1
            rows.append(row)
    later = [r["ttft_ms"] for r in rows[1:]]
    return {
        "scenario": name,
        "environment": environment(base_url),
        "first_ttft_ms": rows[0]["ttft_ms"],
        "later_median_ttft_ms": round(statistics.median(later), 1),
        "later_min_ttft_ms": round(min(later), 1),
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8117")
    ap.add_argument("--out", default="results/prefix.json")
    args = ap.parse_args()
    prefix = build_prefix()
    results = [
        run_scenario(args.base_url, "cache off", prefix, False, False, 4),
        run_scenario(args.base_url, "cache on, stable prefix", prefix, True, False, 8),
        run_scenario(args.base_url, "cache on, timestamp at start", prefix, True, True, 4),
    ]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    for r in results:
        print(
            f"{r['scenario']:32s} first {r['first_ttft_ms']:8.1f} ms   "
            f"later median {r['later_median_ttft_ms']:8.1f} ms   "
            f"later min {r['later_min_ttft_ms']:8.1f} ms"
        )


if __name__ == "__main__":
    main()
