"""Load-test any OpenAI-compatible server and measure TTFT, TPOT and throughput from streams.

    uv run python ch16/01_main-chapter-code/ch16_bench.py \
        --base-url http://localhost:8216/v1 --model Qwen/Qwen3-8B --label vllm-bf16 \
        --concurrency 1,2,4,8,16,32 --requests 32 --prompt-tokens 512 --max-tokens 256 \
        --extra-body '{"chat_template_kwargs": {"enable_thinking": false}, "ignore_eos": true}' \
        --out ch16/01_main-chapter-code/results/vllm-bf16.json

Every request gets a unique prompt (a nonce plus a different slice of the ShopFlow docs), so
prefix caching can't make later requests look faster than the first one.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import statistics
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import openai
import tiktoken

DOCS_DIR = Path(__file__).resolve().parents[2] / "datasets" / "shopflow" / "docs"
INSTRUCTION = (
    "You are a ShopFlow support analyst. Using only the documents above, write a detailed "
    "briefing for a new support agent: summarize every rule, number and exception, with "
    "examples. Write at least 400 words."
)


@dataclass
class RequestRecord:
    """Timings of one streamed request (seconds, relative to its own start)."""

    ok: bool
    ttft: float | None = None  # first content or reasoning token
    e2e: float | None = None  # last byte of the stream
    decode: float | None = None  # first token -> last token
    prompt_tokens: int | None = None
    completion_tokens: int = 0
    error: str | None = None

    @property
    def tpot(self) -> float | None:
        """Time per output token after the first one (a.k.a. mean inter-token latency)."""
        if not self.ok or self.decode is None or self.completion_tokens < 2:
            return None
        return self.decode / (self.completion_tokens - 1)


@dataclass
class LevelResult:
    concurrency: int
    requests: int
    ok: int
    errors: list[str]
    wall_s: float
    output_tok_per_s: float
    requests_per_s: float
    ttft_p50: float | None
    ttft_p95: float | None
    tpot_p50: float | None
    tpot_p95: float | None
    e2e_p50: float | None
    e2e_p95: float | None
    mean_prompt_tokens: float | None
    mean_completion_tokens: float | None
    raw: list[dict] = field(default_factory=list)


def parse_concurrency(text: str) -> list[int]:
    """'1,2,4' -> [1, 2, 4]. Rejects empty, non-positive, duplicate or unsorted levels."""
    try:
        levels = [int(x) for x in text.split(",") if x.strip()]
    except ValueError as e:
        raise ValueError(f"concurrency must be comma-separated integers, got {text!r}") from e
    if not levels:
        raise ValueError("at least one concurrency level is required")
    if any(c < 1 for c in levels):
        raise ValueError(f"concurrency levels must be >= 1, got {levels}")
    if levels != sorted(set(levels)):
        raise ValueError(f"concurrency levels must be strictly increasing, got {levels}")
    return levels


def percentile(values: list[float], q: float) -> float | None:
    """Linear-interpolated percentile (numpy's default method). None for an empty list."""
    if not 0 <= q <= 100:
        raise ValueError(f"q must be in [0, 100], got {q}")
    if not values:
        return None
    xs = sorted(values)
    pos = (len(xs) - 1) * q / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def summarize(concurrency: int, records: list[RequestRecord], wall_s: float) -> LevelResult:
    """Aggregate one concurrency level. Failed requests count against the level, not the stats."""
    if wall_s <= 0:
        raise ValueError("wall_s must be positive")
    good = [r for r in records if r.ok]
    ttfts = [r.ttft for r in good if r.ttft is not None]
    tpots = [t for r in good if (t := r.tpot) is not None]
    e2es = [r.e2e for r in good if r.e2e is not None]
    prompts = [r.prompt_tokens for r in good if r.prompt_tokens is not None]
    out_tokens = sum(r.completion_tokens for r in good)
    return LevelResult(
        concurrency=concurrency,
        requests=len(records),
        ok=len(good),
        errors=[r.error or "unknown" for r in records if not r.ok],
        wall_s=wall_s,
        output_tok_per_s=out_tokens / wall_s,
        requests_per_s=len(good) / wall_s,
        ttft_p50=percentile(ttfts, 50),
        ttft_p95=percentile(ttfts, 95),
        tpot_p50=percentile(tpots, 50),
        tpot_p95=percentile(tpots, 95),
        e2e_p50=percentile(e2es, 50),
        e2e_p95=percentile(e2es, 95),
        mean_prompt_tokens=statistics.fmean(prompts) if prompts else None,
        mean_completion_tokens=out_tokens / len(good) if good else None,
        raw=[asdict(r) | {"tpot": r.tpot} for r in records],
    )


def load_corpus(docs_dir: Path = DOCS_DIR) -> str:
    """All ShopFlow docs, concatenated in a stable order."""
    paths = sorted(docs_dir.rglob("*.md"))
    if not paths:
        raise FileNotFoundError(f"no .md files under {docs_dir}")
    return "\n\n".join(p.read_text() for p in paths)


def build_prompts(
    n: int,
    prompt_tokens: int,
    corpus: str,
    encoding: tiktoken.Encoding,
    seed: int = 16,
) -> list[str]:
    """n distinct prompts of about `prompt_tokens` tokens (counted with `encoding`).

    Each prompt starts with a unique nonce and takes a different window of the corpus, so no two
    prompts share a prefix longer than a few tokens.
    """
    if n < 1 or prompt_tokens < 64:
        raise ValueError("need n >= 1 and prompt_tokens >= 64")
    rng = random.Random(seed)
    corpus_ids = encoding.encode(corpus)
    tail = "\n\n" + INSTRUCTION
    prompts = []
    for i in range(n):
        head = f"Ticket {i:05d}-{rng.getrandbits(32):08x}. Documents:\n\n"
        budget = prompt_tokens - len(encoding.encode(head + tail))
        if budget <= 0:
            raise ValueError(f"prompt_tokens={prompt_tokens} too small for the instruction")
        start = rng.randrange(0, max(1, len(corpus_ids) - budget))
        body = encoding.decode(corpus_ids[start : start + budget])
        prompts.append(head + body + tail)
    return prompts


async def one_request(
    client: openai.AsyncOpenAI,
    model: str,
    prompt: str,
    max_tokens: int,
    extra: dict,
) -> RequestRecord:
    start = time.perf_counter()
    first = last = None
    chunks = 0
    usage = None
    try:
        stream = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0,
            stream=True,
            stream_options={"include_usage": True},
            **extra,
        )
        async for chunk in stream:
            now = time.perf_counter()
            if chunk.usage is not None:
                usage = chunk.usage
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            text = (delta.content or "") + (getattr(delta, "reasoning_content", None) or "")
            text += getattr(delta, "reasoning", None) or ""
            if text:
                chunks += 1
                first = first if first is not None else now
                last = now
    except (openai.APIError, ValueError) as e:
        return RequestRecord(ok=False, error=f"{type(e).__name__}: {str(e)[:200]}")
    end = time.perf_counter()
    if first is None:
        return RequestRecord(ok=False, error="empty stream", e2e=end - start)
    return RequestRecord(
        ok=True,
        ttft=first - start,
        e2e=end - start,
        decode=last - first,
        prompt_tokens=usage.prompt_tokens if usage else None,
        # Servers may pack several tokens into one chunk; trust usage when it is there.
        completion_tokens=usage.completion_tokens if usage else chunks,
    )


async def run_level(
    client: openai.AsyncOpenAI,
    model: str,
    prompts: list[str],
    concurrency: int,
    max_tokens: int,
    extra: dict,
) -> LevelResult:
    """Send all prompts with at most `concurrency` in flight (a closed-loop load generator)."""
    sem = asyncio.Semaphore(concurrency)

    async def guarded(p: str) -> RequestRecord:
        async with sem:
            return await one_request(client, model, p, max_tokens, extra)

    t0 = time.perf_counter()
    records = await asyncio.gather(*(guarded(p) for p in prompts))
    return summarize(concurrency, list(records), time.perf_counter() - t0)


async def run(args: argparse.Namespace) -> dict:
    levels = parse_concurrency(args.concurrency)
    extra = json.loads(args.extra_body) if args.extra_body else {}
    if args.reasoning_effort:
        extra["reasoning_effort"] = args.reasoning_effort
    body = {k: v for k, v in extra.items() if k != "reasoning_effort"}
    kwargs = {"extra_body": body} if body else {}
    if "reasoning_effort" in extra:
        kwargs["reasoning_effort"] = extra["reasoning_effort"]
    client = openai.AsyncOpenAI(
        base_url=args.base_url,
        api_key=args.api_key or os.environ.get("BENCH_API_KEY", "not-needed"),
        timeout=args.timeout,
        max_retries=0,  # a retry would hide the very queueing we want to see
    )
    enc = tiktoken.get_encoding("o200k_base")
    corpus = load_corpus()
    n_total = 1 + args.requests * len(levels)
    prompts = build_prompts(n_total, args.prompt_tokens, corpus, enc, seed=args.seed)
    warm = await one_request(client, args.model, prompts[0], 16, kwargs)  # load / compile
    print(f"[{args.label}] warm-up ok={warm.ok} ttft={warm.ttft} {warm.error or ''}")
    results = []
    for i, c in enumerate(levels):
        batch = prompts[1 + i * args.requests : 1 + (i + 1) * args.requests]
        r = await run_level(client, args.model, batch, c, args.max_tokens, kwargs)
        print(
            f"[{args.label}] c={c:>3} ok={r.ok}/{r.requests} "
            f"TTFT p50={_ms(r.ttft_p50)} p95={_ms(r.ttft_p95)} "
            f"TPOT p50={_ms(r.tpot_p50)} p95={_ms(r.tpot_p95)} "
            f"out={r.output_tok_per_s:,.0f} tok/s wall={r.wall_s:.1f}s",
            flush=True,
        )
        results.append(asdict(r))
    await client.close()
    return {
        "label": args.label,
        "base_url": args.base_url,
        "model": args.model,
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "config": {
            "concurrency": levels,
            "requests": args.requests,
            "prompt_tokens": args.prompt_tokens,
            "max_tokens": args.max_tokens,
            "request_options": extra,
        },
        "levels": results,
    }


def _ms(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 1000:,.0f}ms"


def plot(results: list[dict], out: Path) -> None:
    """Three panels (TTFT p50, TPOT p50, output tokens/s) against concurrency, one line per run."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    panels = [
        ("ttft_p50", 1000, "TTFT p50 (ms, log)", True),
        ("tpot_p50", 1000, "TPOT p50 (ms per token)", False),
        ("output_tok_per_s", 1, "output tokens / s (all requests)", False),
    ]
    for res in results:
        xs = [lv["concurrency"] for lv in res["levels"]]
        for ax, (key, scale, title, logy) in zip(axes, panels, strict=True):
            ys = [lv[key] * scale if lv[key] is not None else float("nan") for lv in res["levels"]]
            ax.plot(xs, ys, marker="o", label=res["label"])
            ax.set_title(title)
            ax.set_xscale("log", base=2)
            ax.set_xticks(xs, [str(x) for x in xs])
            ax.set_xlabel("concurrent requests")
            if logy:
                ax.set_yscale("log")
            ax.grid(alpha=0.3)
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--label", help="name of this run in tables and plots (default: model)")
    p.add_argument("--api-key", help="default: $BENCH_API_KEY or a dummy value")
    p.add_argument("--concurrency", default="1,2,4,8")
    p.add_argument("--requests", type=int, default=32, help="requests per concurrency level")
    p.add_argument("--prompt-tokens", type=int, default=512)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--extra-body", help="JSON merged into every request body")
    p.add_argument("--reasoning-effort", help="e.g. none (Ollama: disables qwen3 thinking)")
    p.add_argument("--timeout", type=float, default=600)
    p.add_argument("--seed", type=int, default=16)
    p.add_argument("--out", type=Path, help="write results JSON here")
    p.add_argument("--plot", type=Path, help="write a PNG with the three panels here")
    args = p.parse_args()
    args.label = args.label or args.model
    result = asyncio.run(run(args))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))
        print("wrote", args.out)
    if args.plot:
        plot([result], args.plot)
        print("wrote", args.plot)


if __name__ == "__main__":
    main()
