"""F1: 429 storms. The SDK obeys Retry-After to the millisecond, so everyone retries together."""

from __future__ import annotations

import asyncio
import threading
import time
from collections import Counter
from http.server import BaseHTTPRequestHandler

import openai
from ch22_common import PORT, Result, completion_json, run_coro, serve, timed

ID, TITLE, LAYER, KIND = "F1", "429 storms: synchronized retries", "Infra & serving", "local"

LIMIT_PER_S, BURST, N_REQUESTS = 20, 20, 100


class Bucket:
    """Token bucket: `rate` tokens per second, at most `burst` stored."""

    def __init__(self, rate: float, burst: float) -> None:
        self.rate, self.burst, self.tokens, self.ts = rate, burst, burst, time.monotonic()
        self.lock = threading.Lock()

    def take(self) -> float:
        """Take a token; return 0 if granted, else seconds until one is available."""
        with self.lock:
            now = time.monotonic()
            self.tokens = min(self.burst, self.tokens + (now - self.ts) * self.rate)
            self.ts = now
            if self.tokens >= 1:
                self.tokens -= 1
                return 0.0
            return (1 - self.tokens) / self.rate


class Provider:
    """State of the fake provider: its own limiter plus a log of every arrival."""

    def __init__(self) -> None:
        self.bucket = Bucket(LIMIT_PER_S, BURST)
        self.arrivals: list[tuple[float, int]] = []
        self.t0 = time.monotonic()


def handler_for(p: Provider) -> type:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            ok = p.bucket.take() == 0
            p.arrivals.append((time.monotonic() - p.t0, 200 if ok else 429))
            body = completion_json("OK") if ok else b'{"error": {"message": "rate limited"}}'
            self.send_response(200 if ok else 429)
            if not ok:
                self.send_header("retry-after", "1")
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    return Handler


class AsyncBucket:
    """Client-side limiter: wait for our own token before calling, at 90% of the provider's rate."""

    def __init__(self, rate: float, burst: float) -> None:
        self.bucket = Bucket(rate, burst)

    async def acquire(self) -> None:
        while (wait := self.bucket.take()) > 0:
            await asyncio.sleep(wait)


async def burst(strategy: str) -> dict:
    p = Provider()
    with serve(handler_for(p)):
        client = openai.AsyncOpenAI(base_url=f"http://127.0.0.1:{PORT}/v1", api_key="stub",
                                    max_retries=10 if strategy == "sdk" else 0)
        limiter = AsyncBucket(LIMIT_PER_S * 0.9, BURST * 0.9)

        async def one() -> None:
            if strategy == "client bucket":
                await limiter.acquire()
            await client.chat.completions.create(
                model="stub", messages=[{"role": "user", "content": "hi"}], max_tokens=5)

        t0 = time.monotonic()
        await asyncio.gather(*(one() for _ in range(N_REQUESTS)))
        elapsed = time.monotonic() - t0
        await client.close()
    per_100ms = Counter(int(t * 10) for t, _ in p.arrivals)
    return {"elapsed_s": round(elapsed, 2), "upstream_calls": len(p.arrivals),
            "rejected_429": sum(code == 429 for _, code in p.arrivals),
            "peak_per_100ms": max(per_100ms.values()),
            "arrivals": [round(t, 3) for t, _ in p.arrivals]}


@timed
def run() -> Result:
    out = {s: run_coro(burst(s)) for s in ("sdk", "client bucket")}
    sdk, ours = out["sdk"], out["client bucket"]
    return Result(
        ID, TITLE, LAYER, KIND,
        reproduced=sdk["rejected_429"] > N_REQUESTS,
        fixed=ours["rejected_429"] <= N_REQUESTS * 0.05,
        evidence=f"{N_REQUESTS} requests at once, provider allows {LIMIT_PER_S}/s, Retry-After: 1. "
                 + "; ".join(f"{k}: {v['upstream_calls']} upstream calls, {v['rejected_429']} "
                             f"429s, peak {v['peak_per_100ms']}/100 ms, done in {v['elapsed_s']} s"
                             for k, v in out.items()),
        data=out,
    )
