"""Quantization experiment: Qwen3-8B at different precisions on the Chapter 2 triage eval.

Same model, same prompt, same 300 tickets (the Chapter 2 eval subset: all 60 hand-written hard
tickets plus 240 seeded templated ones), different weight formats and engines:

  Ollama (llama.cpp GGUF)   qwen3:8b (q4_K_M)   qwen3:8b-q8_0   qwen3:8b-fp16
  vLLM (safetensors)        Qwen/Qwen3-8B (bf16)  Qwen/Qwen3-8B-AWQ (int4)
                            RedHatAI/Qwen3-8B-FP8-dynamic (fp8 weights; Marlin W8A16 on Ampere)

For each one we record:
  accuracy      category and priority exact match, strict JSON schema (Chapter 2's v1 prompt)
  latency       per-ticket wall time at concurrency 1 (p50 / p95)
  decode tok/s  single stream, 256-token free-form generation, median of 3
  memory        what the engine reports (Ollama ``/api/ps``; vLLM's startup log, if given)

One configuration per call, because each vLLM model needs its own server:

    uv run python ch17_quant_bench.py --config ollama-q4_K_M
    uv run python ch17_quant_bench.py --config vllm-awq-int4 --server-log results/logs/vllm-awq.log
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import httpx
from ch17_stream import stream_chat

HERE = Path(__file__).parent


def _previous():
    """Chapter 2 code, loaded under a unique name (every chapter ships a previous_chapters.py)."""
    name = "ch17_previous_chapters"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, HERE / "previous_chapters.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


@dataclass(frozen=True)
class Config:
    engine: str  # "ollama" or "vllm"
    base_url: str
    model: str
    weights: str  # human label for the weight format


CONFIGS: dict[str, Config] = {
    "ollama-q4_K_M": Config("ollama", "http://localhost:11434", "qwen3:8b", "GGUF q4_K_M"),
    "ollama-q8_0": Config("ollama", "http://localhost:11434", "qwen3:8b-q8_0", "GGUF q8_0"),
    "ollama-fp16": Config("ollama", "http://localhost:11434", "qwen3:8b-fp16", "GGUF f16"),
    "vllm-bf16": Config("vllm", "http://localhost:8217", "Qwen/Qwen3-8B", "bf16"),
    "vllm-awq-int4": Config("vllm", "http://localhost:8217", "Qwen/Qwen3-8B-AWQ", "AWQ int4"),
    "vllm-fp8-w8a16": Config(
        "vllm", "http://localhost:8217", "RedHatAI/Qwen3-8B-FP8-dynamic", "FP8 (W8A16)"
    ),
}

DECODE_PROMPT = (
    "Write a detailed, friendly 400-word guide for a new ShopFlow support agent on handling a "
    "customer whose parcel is late."
)


def no_thinking(engine: str) -> dict:
    """Qwen3 thinks by default. Each engine has its own switch for turning that off."""
    if engine == "ollama":
        return {"reasoning_effort": "none"}  # honored by Ollama's OpenAI-compatible endpoint
    return {"chat_template_kwargs": {"enable_thinking": False}}


def triage_body(cfg: Config, ticket) -> dict:
    prev = _previous()
    return {
        "model": cfg.model,
        "max_tokens": 300,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": prev.RULES},
            {"role": "user", "content": prev.format_ticket(ticket.subject, ticket.body,
                                                           ticket.channel)},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "TicketTriage", "strict": True,
                            "schema": prev.strict_schema()},
        },
        **no_thinking(cfg.engine),
    }


def score_one(raw: str, ticket) -> dict:
    """Parse one reply and compare it with the gold labels. A parse failure counts as wrong."""
    prev = _previous()
    try:
        t = prev.parse_triage(raw)
    except prev.TriageError as e:
        return {"id": ticket.id, "hard": ticket.hard, "parsed": False, "error": e.kind,
                "category_ok": False, "priority_ok": False}
    return {"id": ticket.id, "hard": ticket.hard, "parsed": True, "category": t.category,
            "priority": t.priority, "category_ok": t.category == ticket.category,
            "priority_ok": t.priority == ticket.priority}


def wilson(correct: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion; honest error bars for n = 300."""
    if n <= 0 or not 0 <= correct <= n:
        raise ValueError("need 0 <= correct <= n and n > 0")
    p = correct / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return centre - half, centre + half


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile (q in 0..100)."""
    if not values:
        raise ValueError("no values")
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, math.ceil(q / 100 * len(ordered)) - 1))
    return ordered[k]


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    cat = sum(r["category_ok"] for r in rows)
    pri = sum(r["priority_ok"] for r in rows)
    hard = [r for r in rows if r["hard"]]
    lo, hi = wilson(cat, n)
    return {
        "n": n,
        "category_acc": round(100 * cat / n, 1),
        "category_ci95": [round(100 * lo, 1), round(100 * hi, 1)],
        "priority_acc": round(100 * pri / n, 1),
        "hard_category_acc": round(100 * sum(r["category_ok"] for r in hard) / len(hard), 1)
        if hard else None,
        "parse_failures": sum(not r["parsed"] for r in rows),
    }


_VLLM_LOG = {
    "weights_gib": re.compile(r"Model loading took ([0-9.]+) GiB"),
    "kv_cache_gib": re.compile(r"Available KV cache memory: ([0-9.]+) GiB"),
    "kv_cache_tokens": re.compile(r"GPU KV cache size: ([0-9,]+) tokens"),
}


def parse_vllm_log(text: str) -> dict:
    """The memory facts vLLM prints at startup: weights, KV budget, KV capacity in tokens."""
    out: dict = {}
    for key, rx in _VLLM_LOG.items():
        found = rx.findall(text)
        if found:
            value = found[-1].replace(",", "")
            out[key] = int(value) if key == "kv_cache_tokens" else float(value)
    return out


def ollama_memory(cfg: Config) -> dict:
    ps = httpx.get(f"{cfg.base_url}/api/ps", timeout=10).json()["models"]
    m = next((m for m in ps if m["name"] == cfg.model), None)
    if m is None:
        return {}
    return {"loaded_gb": round(m["size"] / 1e9, 2), "vram_gb": round(m["size_vram"] / 1e9, 2),
            "context_length": m.get("context_length")}


def decode_speed(client: httpx.Client, cfg: Config, runs: int = 3) -> list[float]:
    speeds = []
    for _ in range(runs):
        r = stream_chat(client, cfg.base_url, {
            "model": cfg.model, "max_tokens": 256, "temperature": 0,
            "messages": [{"role": "user", "content": DECODE_PROMPT}], **no_thinking(cfg.engine),
        })
        if r.decode_tok_s:
            speeds.append(round(r.decode_tok_s, 1))
    return speeds


def run_config(name: str, n: int, server_log: str | None = None, limit: int | None = None) -> dict:
    cfg = CONFIGS[name]
    prev = _previous()
    tickets = prev.sample_tickets(prev.load_tickets(), n, seed=7)[:limit]
    rows, latencies = [], []
    with httpx.Client(timeout=300) as client:
        stream_chat(client, cfg.base_url, {  # load / warm up, not measured
            "model": cfg.model, "max_tokens": 4,
            "messages": [{"role": "user", "content": "hi"}], **no_thinking(cfg.engine),
        })
        speeds = decode_speed(client, cfg)
        t0 = time.perf_counter()
        for t in tickets:
            r = stream_chat(client, cfg.base_url, triage_body(cfg, t))
            latencies.append(r.total_ms)
            rows.append({**score_one(r.text, t), "ms": r.total_ms,
                         "completion_tokens": r.completion_tokens})
        wall = time.perf_counter() - t0
    memory = ollama_memory(cfg) if cfg.engine == "ollama" else {}
    if server_log and Path(server_log).exists():
        memory |= parse_vllm_log(Path(server_log).read_text(errors="replace"))
    return {
        "config": name, "engine": cfg.engine, "model": cfg.model, "weights": cfg.weights,
        **summarize(rows),
        "latency_p50_ms": round(statistics.median(latencies)),
        "latency_p95_ms": round(percentile(latencies, 95)),
        "wall_s": round(wall, 1),
        "decode_tok_s": statistics.median(speeds) if speeds else None,
        "decode_runs": speeds,
        "memory": memory,
        "rows": rows,
    }


def throughput(cfg: Config, concurrency: int, max_tokens: int = 256) -> dict:
    """Aggregate decode throughput with ``concurrency`` identical-length requests in flight."""
    body = {"model": cfg.model, "max_tokens": max_tokens, "temperature": 0, "ignore_eos": True,
            "messages": [{"role": "user", "content": DECODE_PROMPT}], **no_thinking(cfg.engine)}
    with httpx.Client(timeout=600, limits=httpx.Limits(max_connections=concurrency + 4)) as c:
        t0 = time.perf_counter()
        with ThreadPoolExecutor(concurrency) as pool:
            results = list(pool.map(lambda _: stream_chat(c, cfg.base_url, body),
                                    range(concurrency)))
        wall = time.perf_counter() - t0
    tokens = sum(r.completion_tokens or 0 for r in results)
    return {"concurrency": concurrency, "output_tokens": tokens, "wall_s": round(wall, 2),
            "output_tok_s": round(tokens / wall, 1),
            "per_stream_tok_s": round(statistics.median(r.decode_tok_s or 0 for r in results), 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=list(CONFIGS))
    ap.add_argument("--n", type=int, default=300, help="Chapter 2 eval subset size")
    ap.add_argument("--limit", type=int, help="smoke test: only the first N tickets")
    ap.add_argument("--server-log", help="vLLM log to read memory facts from")
    ap.add_argument("--throughput", nargs="*", type=int, default=[],
                    help="also measure aggregate decode tok/s at these concurrencies (vLLM)")
    ap.add_argument("--out", default=str(HERE / "results" / "quant.json"))
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    store = json.loads(out.read_text()) if out.exists() else {}
    res = run_config(args.config, args.n, args.server_log, args.limit)
    res["throughput"] = [throughput(CONFIGS[args.config], c) for c in args.throughput]
    store[args.config] = res
    out.write_text(json.dumps(store, indent=2))
    print(f"{args.config:16s} category {res['category_acc']:5.1f}% {res['category_ci95']}  "
          f"priority {res['priority_acc']:5.1f}%  hard {res['hard_category_acc']}%  "
          f"parse fails {res['parse_failures']}  p50 {res['latency_p50_ms']} ms  "
          f"decode {res['decode_tok_s']} tok/s  memory {res['memory']}")
    for t in res["throughput"]:
        print(f"  concurrency {t['concurrency']:3d}: {t['output_tok_s']:7.1f} tok/s total, "
              f"{t['per_stream_tok_s']:5.1f} tok/s per stream")


if __name__ == "__main__":
    main()
