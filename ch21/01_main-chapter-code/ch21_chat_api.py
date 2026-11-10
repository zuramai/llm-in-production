"""Chapter 1's chat API, with the LLM call made swappable so we can load-test it.

    cd ch21/01_main-chapter-code
    CH21_MODE=async uv run uvicorn ch21_chat_api:app --port 8021 --no-access-log

CH21_MODE picks how `/chat` calls the LLM:
  thread    Chapter 1 as shipped: a `def` endpoint returning a *sync* generator. sse-starlette
            iterates it in AnyIO's worker-thread pool (40 threads unless CH21_THREAD_LIMIT).
  blocking  The classic mistake: an `async def` endpoint iterating the sync client's stream
            directly, on the event loop.
  async     `AsyncOpenAI` on the event loop, with optional failover and per-user rate limit.

Other settings (environment):
  CH21_LLM_BASE_URLS   comma-separated OpenAI-compatible base URLs; more than one = failover
                       (async mode). Default: the mock server on :8121.
  CH21_LLM_MODEL       model name sent upstream (default "mock")
  CH21_LLM_API_KEY     API key sent upstream (default "mock")
  CH21_MAX_RETRIES     SDK retries per provider (default 2, the SDK default)
  CH21_MAX_CONNECTIONS httpx connection-pool size for the async client (default: SDK's 1000)
  CH21_THREAD_LIMIT    AnyIO worker threads (default 40)
  CH21_USER_RPS        per-session token-bucket refill rate in requests/s (0 = off; async mode)
  CH21_USER_BURST      per-session bucket capacity (default 5)
  CH21_REDIS_URL       Redis for the rate limiter (default redis://localhost:6386/0)
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager

import anyio.to_thread
import httpx2
import openai
import redis.asyncio as aredis
from ch21_ratelimit import AsyncRedisTokenBucket
from ch21_resilience import AllProvidersFailed, CircuitBreaker, Provider, stream_with_failover
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from previous_chapters import SessionStore, trim_history
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

MODE = os.environ.get("CH21_MODE", "thread")
BASE_URLS = os.environ.get("CH21_LLM_BASE_URLS", "http://localhost:8121/v1").split(",")
MODEL = os.environ.get("CH21_LLM_MODEL", "mock")
API_KEY = os.environ.get("CH21_LLM_API_KEY", "mock")
MAX_RETRIES = int(os.environ.get("CH21_MAX_RETRIES", "2"))
MAX_CONNECTIONS = int(os.environ.get("CH21_MAX_CONNECTIONS", "0"))
THREAD_LIMIT = int(os.environ.get("CH21_THREAD_LIMIT", "40"))
USER_RPS = float(os.environ.get("CH21_USER_RPS", "0"))
USER_BURST = float(os.environ.get("CH21_USER_BURST", "5"))
REDIS_URL = os.environ.get("CH21_REDIS_URL", "redis://localhost:6386/0")
MAX_CONTEXT_TOKENS = 8_000
TIMEOUT = 60.0

if MODE not in ("thread", "blocking", "async"):
    raise RuntimeError(f"CH21_MODE must be thread, blocking or async, got {MODE!r}")

sessions = SessionStore()
sync_client = openai.OpenAI(base_url=BASE_URLS[0], api_key=API_KEY, timeout=TIMEOUT,
                            max_retries=MAX_RETRIES)


def _async_client(base_url: str) -> openai.AsyncOpenAI:
    http_client = None
    if MAX_CONNECTIONS:  # the SDK default is 1,000 connections; we shrink it to show the effect
        limits = httpx2.Limits(max_connections=MAX_CONNECTIONS,
                               max_keepalive_connections=MAX_CONNECTIONS)
        http_client = openai.DefaultAsyncHttpxClient(limits=limits)
    return openai.AsyncOpenAI(base_url=base_url, api_key=API_KEY, timeout=TIMEOUT,
                              max_retries=MAX_RETRIES, http_client=http_client)


providers = [
    Provider(name=f"p{i}:{url}", client=_async_client(url), model=MODEL,
             breaker=CircuitBreaker(failure_threshold=3, reset_after_s=10))
    for i, url in enumerate(BASE_URLS)
]
# redis-py's asyncio pool *raises* MaxConnectionsError past 100 concurrent commands by default;
# a BlockingConnectionPool makes callers wait for a free connection instead.
_redis = aredis.Redis(connection_pool=aredis.BlockingConnectionPool.from_url(
    REDIS_URL, max_connections=64, timeout=5))
limiter = AsyncRedisTokenBucket(_redis, USER_RPS, USER_BURST) if USER_RPS > 0 else None


@asynccontextmanager
async def lifespan(_: FastAPI):
    anyio.to_thread.current_default_thread_limiter().total_tokens = THREAD_LIMIT
    yield


app = FastAPI(title=f"ShopFlow chat API (ch21, mode={MODE})", lifespan=lifespan)


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=100)
    message: str = Field(min_length=1, max_length=4000)


@app.get("/healthz")
def healthz() -> dict:  # a `def` endpoint, as in Chapter 1: it also needs a worker thread
    return {"status": "ok", "mode": MODE}


def _sync_events(req: ChatRequest) -> Iterator[dict]:
    """Chapter 1's generator: blocking network I/O, meant to run in a worker thread."""
    history = sessions.get(req.session_id) + [{"role": "user", "content": req.message}]
    parts: list[str] = []
    usage: dict = {}
    try:
        history = trim_history(history, MAX_CONTEXT_TOKENS)
        stream = sync_client.chat.completions.create(
            model=MODEL, messages=history, max_tokens=800, stream=True,
            stream_options={"include_usage": True},
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                parts.append(chunk.choices[0].delta.content)
                yield {"data": json.dumps({"delta": chunk.choices[0].delta.content})}
            if chunk.usage is not None:
                usage = chunk.usage.model_dump(exclude_none=True)
    except (openai.OpenAIError, ValueError) as exc:
        yield {"data": json.dumps({"error": str(exc)})}
        return
    sessions.put(req.session_id, history + [{"role": "assistant", "content": "".join(parts)}])
    yield {"data": json.dumps({"done": True, "usage": usage})}


if MODE == "thread":

    @app.post("/chat")
    def chat(req: ChatRequest) -> EventSourceResponse:
        return EventSourceResponse(_sync_events(req))

elif MODE == "blocking":

    @app.post("/chat")
    async def chat(req: ChatRequest) -> EventSourceResponse:
        async def events() -> AsyncIterator[dict]:
            for event in _sync_events(req):  # DON'T: blocking I/O on the event loop
                yield event

        return EventSourceResponse(events())

else:

    @app.post("/chat", response_model=None)
    async def chat(req: ChatRequest) -> EventSourceResponse | JSONResponse:
        if limiter is not None:
            allowed, retry_after = await limiter.acquire(f"user:{req.session_id}")
            if not allowed:
                return JSONResponse(
                    {"error": "rate limited", "retry_after_s": round(retry_after, 3)},
                    status_code=429, headers={"retry-after": f"{max(1, round(retry_after))}"},
                )

        async def events() -> AsyncIterator[dict]:
            history = sessions.get(req.session_id) + [{"role": "user", "content": req.message}]
            parts: list[str] = []
            meta: dict = {}
            try:
                history = trim_history(history, MAX_CONTEXT_TOKENS)
                async for delta in stream_with_failover(providers, history, max_tokens=800,
                                                        meta=meta):
                    parts.append(delta)
                    yield {"data": json.dumps({"delta": delta})}
            except (openai.OpenAIError, AllProvidersFailed, ValueError) as exc:
                yield {"data": json.dumps({"error": str(exc)})}
                return
            sessions.put(req.session_id,
                         history + [{"role": "assistant", "content": "".join(parts)}])
            yield {"data": json.dumps({"done": True, "usage": meta.get("usage", {}),
                                       "provider": meta.get("provider")})}

        return EventSourceResponse(events())
