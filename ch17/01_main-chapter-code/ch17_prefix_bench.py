"""Prefix-cache benchmark: time to first token with and without a reusable prompt prefix.

The prompt is a long, stable ShopFlow knowledge base (system message) plus a short question.
Start a server first (see the chapter or ``run_gpu_benchmarks.sh``), then:

    # llama-server: cache_prompt is a per-request switch, so one server covers every scenario
    uv run python ch17_prefix_bench.py --engine llama --base-url http://localhost:8117
    # vLLM: automatic prefix caching is a server flag, so run once per server
    uv run python ch17_prefix_bench.py --engine vllm --base-url http://localhost:8217 --apc on
    uv run python ch17_prefix_bench.py --engine vllm --base-url http://localhost:8217 --apc off

Scenarios, each with its own session nonce at the top of the prompt so its first request is a
guaranteed cold start and nothing leaks in from the previous scenario:
  cache off                every request pays the full prefill
  cache on, stable prefix  requests 2..N reuse the prefix KV
  cache on, timestamp      a per-request timestamp at the very START of the prompt: nothing matches
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import httpx
from ch17_stream import stream_chat

HERE = Path(__file__).parent
REPO = HERE.parents[1]
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


def request_body(model: str | None, system: str, question: str, cache_prompt: bool | None) -> dict:
    body = {
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": question}],
        "max_tokens": 16,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if model:
        body["model"] = model
    if cache_prompt is not None:  # llama-server only; vLLM decides at server start
        body["cache_prompt"] = cache_prompt
    return body


def run_scenario(
    base_url: str, model: str | None, name: str, prefix: str, cache_prompt: bool | None,
    bust: bool, n_questions: int,
) -> dict:
    rows = []
    nonce = f"[session {time.time_ns()}]\n"
    warm_cache = None if cache_prompt is None else False
    with httpx.Client(timeout=300) as client:
        # Tiny request so the model is warm; it shares no prefix with the real prompts.
        stream_chat(client, base_url, request_body(model, "warm up", "hi", warm_cache))
        for i, q in enumerate(QUESTIONS[:n_questions]):
            head = f"[request-id {time.time_ns()}]\n" if bust else nonce
            r = stream_chat(client, base_url, request_body(model, head + prefix, q, cache_prompt))
            rows.append({
                "request": i + 1,
                "ttft_ms": r.ttft_ms,
                "prompt_tokens": r.prompt_tokens,
                # llama-server: prompt_n = tokens it actually prefilled, cache_n = reused ones
                "server_prefilled": r.timings.get("prompt_n"),
                "server_cached": r.timings.get("cache_n", r.cached_tokens),
            })
    later = [r["ttft_ms"] for r in rows[1:]]
    return {
        "scenario": name,
        "prompt_tokens": rows[0]["prompt_tokens"],
        "first_ttft_ms": rows[0]["ttft_ms"],
        "later_median_ttft_ms": round(statistics.median(later), 1),
        "later_min_ttft_ms": round(min(later), 1),
        "rows": rows,
    }


def scenarios(engine: str, apc: str) -> list[tuple[str, bool | None, bool, int]]:
    """(name, cache_prompt field, bust, n_questions) for one server."""
    if engine == "llama":
        return [("cache off", False, False, 4), ("cache on, stable prefix", True, False, 8),
                ("cache on, timestamp at start", True, True, 4)]
    if apc == "off":
        return [("cache off", None, False, 4)]
    return [("cache on, stable prefix", None, False, 8),
            ("cache on, timestamp at start", None, True, 4)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["llama", "vllm"], required=True)
    ap.add_argument("--base-url", default="http://localhost:8117")
    ap.add_argument("--model", help="model name to send (vLLM needs it)")
    ap.add_argument("--apc", choices=["on", "off"], default="on",
                    help="vLLM only: was the server started with prefix caching?")
    ap.add_argument("--prefix-chars", nargs="+", type=int, default=[12_000, 48_000])
    ap.add_argument("--out", default=str(HERE / "results" / "prefix.json"))
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    store = json.loads(out.read_text()) if out.exists() else {}
    for chars in args.prefix_chars:
        prefix = build_prefix(chars)
        for name, cache_prompt, bust, n in scenarios(args.engine, args.apc):
            r = run_scenario(args.base_url, args.model, name, prefix, cache_prompt, bust, n)
            key = f"{args.engine} | {chars} chars | {name}"
            store[key] = {"engine": args.engine, "prefix_chars": chars, **r}
            out.write_text(json.dumps(store, indent=2))
            print(f"{key:52s} prompt {r['prompt_tokens']:>6} tok  first {r['first_ttft_ms']:8.1f}"
                  f" ms  later median {r['later_median_ttft_ms']:8.1f} ms")


if __name__ == "__main__":
    main()
