"""ShopFlow support chatbot API, production edition (Chapter 1's app.py, deployed).

    uvicorn --factory ch20_chat.app:create_app --env-file .env --port 8120

Differences from Chapter 1: configuration from the environment only, versioned system prompts
with a percentage canary, a pluggable session store, split liveness/readiness endpoints and one
structured log line per request.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import openai
from fastapi import FastAPI, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from .canary import choose_version
from .core import load_prompts, stream_reply, trim_history
from .sessions import MemorySessionStore, PostgresSessionStore, SessionStore
from .settings import Settings, load_settings

STATIC = Path(__file__).with_name("static")
POD = os.environ.get("HOSTNAME") or socket.gethostname()

log = logging.getLogger("ch20_chat")
if not log.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(_handler)
    log.setLevel(logging.INFO)
    log.propagate = False


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=100)
    message: str = Field(min_length=1, max_length=4000)


def make_store(settings: Settings) -> SessionStore:
    if settings.session_store == "postgres":
        return PostgresSessionStore(settings.sessions_database_url)
    return MemorySessionStore()


def create_app(
    settings: Settings | None = None,
    *,
    client: openai.OpenAI | None = None,
    store: SessionStore | None = None,
) -> FastAPI:
    settings = settings or load_settings()
    prompts = load_prompts()
    for version in filter(None, (settings.prompt_stable, settings.prompt_canary)):
        if version not in prompts:
            raise ValueError(f"unknown prompt version {version!r}; have {sorted(prompts)}")
    client = client or openai.OpenAI(base_url=settings.llm_base_url,
                                     api_key=settings.llm_api_key,
                                     timeout=settings.llm_timeout_s, max_retries=1)
    store = store or make_store(settings)
    app = FastAPI(title="ShopFlow support chatbot")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    # Both probes are `async def`: they run on the event loop, not in the thread pool that the
    # sync /chat streams fill up under load. A sync probe waits behind those streams, times out,
    # and liveness then restarts a healthy pod (ch21 measured /healthz p95 at 4.4 s under load).

    @app.get("/healthz")
    async def healthz() -> dict:
        # Liveness: "is this process able to serve at all?" No dependency checks here: a
        # database outage must not make Kubernetes restart every replica in a loop.
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz(response: Response) -> dict:
        # Readiness: "should the Service send me traffic?" Here dependencies do matter.
        try:
            await store.aping()
        except Exception as exc:  # noqa: BLE001 - any failure means not ready
            response.status_code = 503
            return {"status": "not ready", "error": type(exc).__name__}
        return {"status": "ready", "prompt_stable": settings.prompt_stable,
                "prompt_canary": settings.prompt_canary,
                "canary_percent": settings.canary_percent}

    @app.post("/chat")
    def chat(req: ChatRequest) -> EventSourceResponse:
        version = choose_version(req.session_id, settings.prompt_stable,
                                 settings.prompt_canary, settings.canary_percent)

        def events() -> Iterator[dict]:
            started = time.perf_counter()
            ttft_ms = None
            usage: dict = {}
            parts: list[str] = []
            status = "ok"
            try:
                turns = store.get(req.session_id)
                history = trim_history(
                    [{"role": "system", "content": prompts[version]}, *turns,
                     {"role": "user", "content": req.message}],
                    settings.max_context_tokens)
                for delta in stream_reply(client, history, model=settings.llm_model,
                                          usage=usage, max_tokens=settings.max_output_tokens,
                                          reasoning_effort=settings.reasoning_effort):
                    if ttft_ms is None:
                        ttft_ms = round((time.perf_counter() - started) * 1000)
                    parts.append(delta)
                    yield {"data": json.dumps({"delta": delta})}
                store.put(req.session_id,
                          history[1:] + [{"role": "assistant", "content": "".join(parts)}])
                yield {"data": json.dumps({"done": True, "usage": usage,
                                           "prompt_version": version, "pod": POD})}
            except (openai.OpenAIError, ValueError, OSError) as exc:
                status = type(exc).__name__
                yield {"data": json.dumps({"error": str(exc)})}
            finally:
                log.info(json.dumps({
                    "event": "chat", "pod": POD, "prompt_version": version,
                    "session_id": req.session_id, "status": status, "ttft_ms": ttft_ms,
                    "total_ms": round((time.perf_counter() - started) * 1000),
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "cost_usd": usage.get("cost"),
                }))

        return EventSourceResponse(events(), headers={"X-Prompt-Version": version,
                                                      "X-Served-By": POD})

    return app
