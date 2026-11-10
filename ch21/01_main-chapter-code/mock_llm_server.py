"""An OpenAI-compatible fake LLM for load tests: configurable TTFT, tokens/s and failure rates.

    cd ch21/01_main-chapter-code
    MOCK_TTFT_MS=800 MOCK_TOKENS_PER_S=120 MOCK_RATE_429=0.05 \
        uv run uvicorn mock_llm_server:app --port 8121 --no-access-log

It implements just enough of `POST /v1/chat/completions` (streaming and non-streaming, with
`stream_options.include_usage`) for the OpenAI SDK to be happy. Behavior can be changed at runtime
with `POST /admin/config` (to simulate an outage in the middle of a load test), and `GET
/admin/stats` counts what the server saw, so experiments can measure retries and wasted calls.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass, fields

from ch21_ratelimit import TokenBucket
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

WORDS = ["You", "can", "return", "most", "items", "within", "30", "days", "of", "delivery", "as", "long", "as", "they", "are", "unused", "and", "in", "their", "original", "packaging", ".", "Refunds", "go", "back", "to", "the", "original", "payment", "method", "within", "5", "to", "7", "business", "days", "after", "we", "inspect", "the", "return", "."]


@dataclass
class MockConfig:
    ttft_ms: float = 800.0  # time to first token
    tokens_per_s: float = 120.0  # decode speed after the first token
    output_tokens: int = 60  # answer length, capped by the request's max_tokens
    rate_429: float = 0.0  # probability of answering 429 instead
    rate_500: float = 0.0  # probability of answering 500 instead
    retry_after_s: float | None = 1.0  # Retry-After header on 429s (None: no header)
    rps_limit: float | None = None  # provider-side token bucket (requests/s); None: unlimited
    burst: float | None = None  # bucket capacity; defaults to rps_limit
    seed: int | None = None

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None) -> MockConfig:
        env = os.environ if environ is None else environ
        values: dict = {}
        for f in fields(cls):
            raw = env.get(f"MOCK_{f.name.upper()}")
            if raw is None:
                continue
            if raw.strip().lower() in ("", "none"):
                values[f.name] = None
            elif f.name in ("output_tokens", "seed"):
                values[f.name] = int(raw)
            else:
                values[f.name] = float(raw)
        return cls(**values)


class ConfigPatch(BaseModel):
    ttft_ms: float | None = None
    tokens_per_s: float | None = None
    output_tokens: int | None = None
    rate_429: float | None = None
    rate_500: float | None = None
    retry_after_s: float | None = None
    rps_limit: float | None = None
    burst: float | None = None


def _error(status: int, message: str, kind: str, headers: dict[str, str] | None = None):
    body = {"error": {"message": message, "type": kind, "code": kind}}
    return JSONResponse(body, status_code=status, headers=headers)


def create_app(config: MockConfig | None = None) -> FastAPI:
    cfg = config or MockConfig()
    rng = random.Random(cfg.seed)
    state: dict = {"bucket": None, "stats": {}}

    def reset_bucket() -> None:
        state["bucket"] = (
            TokenBucket(cfg.rps_limit, cfg.burst or cfg.rps_limit) if cfg.rps_limit else None
        )

    def reset_stats() -> None:
        state["stats"] = {"requests": 0, "ok": 0, "status_429": 0, "status_500": 0,
                          "in_flight": 0, "max_in_flight": 0, "arrivals": []}

    reset_bucket()
    reset_stats()
    app = FastAPI(title="mock LLM")

    @app.get("/v1/models")
    async def models() -> dict:
        return {"object": "list", "data": [{"id": "mock", "object": "model", "owned_by": "ch21"}]}

    @app.get("/admin/config")
    async def get_config() -> dict:
        return asdict(cfg)

    @app.post("/admin/config")
    async def patch_config(patch: ConfigPatch) -> dict:
        for key, value in patch.model_dump(exclude_unset=True).items():
            setattr(cfg, key, value)
        reset_bucket()
        return asdict(cfg)

    @app.get("/admin/stats")
    async def get_stats(arrivals: bool = False) -> dict:
        stats = dict(state["stats"])
        if not arrivals:
            stats.pop("arrivals")
        return stats

    @app.post("/admin/reset")
    async def reset() -> dict:
        reset_stats()
        reset_bucket()
        return {"ok": True}

    @app.post("/v1/chat/completions")
    async def completions(request: Request):
        body = await request.json()
        stats = state["stats"]
        stats["requests"] += 1
        stats["arrivals"].append(time.time())
        bucket: TokenBucket | None = state["bucket"]
        if bucket is not None:
            allowed, wait = bucket.acquire()
            if not allowed:
                stats["status_429"] += 1
                headers = {"retry-after": f"{cfg.retry_after_s:g}"} if cfg.retry_after_s else {}
                headers["x-mock-bucket-retry-after"] = f"{wait:.3f}"
                return _error(429, "Rate limit reached (mock bucket)", "rate_limit_exceeded",
                              headers)
        roll = rng.random()
        if roll < cfg.rate_429:
            stats["status_429"] += 1
            headers = {"retry-after": f"{cfg.retry_after_s:g}"} if cfg.retry_after_s else None
            return _error(429, "Rate limit reached (mock)", "rate_limit_exceeded", headers)
        if roll < cfg.rate_429 + cfg.rate_500:
            stats["status_500"] += 1
            return _error(500, "The server had an error (mock)", "server_error")

        n_out = max(1, min(int(body.get("max_tokens") or cfg.output_tokens), cfg.output_tokens))
        prompt_chars = sum(len(str(m.get("content", ""))) for m in body.get("messages", []))
        usage = {"prompt_tokens": max(1, prompt_chars // 4), "completion_tokens": n_out,
                 "total_tokens": max(1, prompt_chars // 4) + n_out}
        model = body.get("model", "mock")
        cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        tokens = [(" " if i else "") + WORDS[i % len(WORDS)] for i in range(n_out)]
        ttft, tpot = cfg.ttft_ms / 1000, 1 / cfg.tokens_per_s

        def chunk(delta: dict, finish: str | None = None, with_usage: bool = False) -> str:
            payload = {"id": cid, "object": "chat.completion.chunk", "created": int(time.time()),
                       "model": model,
                       "choices": [] if with_usage else
                       [{"index": 0, "delta": delta, "finish_reason": finish}]}
            if with_usage:
                payload["usage"] = usage
            return f"data: {json.dumps(payload)}\n\n"

        async def tracked(gen: AsyncIterator[str]) -> AsyncIterator[str]:
            stats["in_flight"] += 1
            stats["max_in_flight"] = max(stats["max_in_flight"], stats["in_flight"])
            try:
                async for part in gen:
                    yield part
                stats["ok"] += 1
            finally:
                stats["in_flight"] -= 1

        if body.get("stream"):
            include_usage = bool((body.get("stream_options") or {}).get("include_usage"))

            async def events() -> AsyncIterator[str]:
                await asyncio.sleep(ttft)
                yield chunk({"role": "assistant", "content": ""})
                for i, tok in enumerate(tokens):
                    if i:
                        await asyncio.sleep(tpot)
                    yield chunk({"content": tok})
                yield chunk({}, finish="stop")
                if include_usage:
                    yield chunk({}, with_usage=True)
                yield "data: [DONE]\n\n"

            return StreamingResponse(tracked(events()), media_type="text/event-stream")

        async def whole() -> AsyncIterator[str]:
            await asyncio.sleep(ttft + tpot * (n_out - 1))
            yield json.dumps({
                "id": cid, "object": "chat.completion", "created": int(time.time()),
                "model": model, "usage": usage,
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": "".join(tokens)}}],
            })

        return StreamingResponse(tracked(whole()), media_type="application/json")

    return app


app = create_app(MockConfig.from_env())
