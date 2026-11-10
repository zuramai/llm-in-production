"""ShopFlow data bot as an HTTP API: POST /ask {"question": ...} -> answer, SQL and rows.

    uvicorn --factory ch20_sqlbot.app:create_app --env-file .env --port 8020

Chapter 6 served the bot from Slack and a CLI. Here it becomes a plain JSON API that any
client (Slack handler, dashboard, CI eval gate) can call.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import sys

import openai
import psycopg
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from .agent import SqlAnswer, answer
from .settings import Settings, load_settings

POD = os.environ.get("HOSTNAME") or socket.gethostname()
ROWS_IN_RESPONSE = 50

log = logging.getLogger("ch20_sqlbot")
if not log.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(_handler)
    log.setLevel(logging.INFO)
    log.propagate = False


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


class AskResponse(SqlAnswer):
    row_count: int
    pod: str


def create_app(settings: Settings | None = None, *, client: openai.OpenAI | None = None,
               ask=answer) -> FastAPI:
    settings = settings or load_settings()
    client = client or openai.OpenAI(base_url=settings.llm_base_url,
                                     api_key=settings.llm_api_key,
                                     timeout=settings.llm_timeout_s, max_retries=1)
    app = FastAPI(title="ShopFlow data bot")

    # Probes are `async def` so they never queue behind /ask calls in the thread pool.
    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz(response: Response) -> dict:
        try:
            async with await psycopg.AsyncConnection.connect(
                    settings.database_url_ro, autocommit=True, connect_timeout=3) as conn:
                await conn.execute("SELECT 1")
        except Exception as exc:  # noqa: BLE001 - any failure means not ready
            response.status_code = 503
            return {"status": "not ready", "error": type(exc).__name__}
        return {"status": "ready", "model": settings.llm_model}

    @app.post("/ask")
    def ask_endpoint(req: AskRequest) -> AskResponse:
        # Sync endpoint: FastAPI runs it in the thread pool, so the blocking OpenAI and psycopg
        # calls don't stall the event loop.
        try:
            result = ask(req.question, settings=settings, client=client)
        except openai.OpenAIError as exc:
            log.info(json.dumps({"event": "ask", "pod": POD, "status": type(exc).__name__}))
            raise HTTPException(status_code=502, detail=f"LLM provider error: {exc}") from None
        log.info(json.dumps({
            "event": "ask", "pod": POD, "status": "ok", "attempts": result.attempts,
            "latency_ms": result.latency_ms, "cost_usd": result.usage.get("cost_usd"),
        }))
        return AskResponse(**{**result.model_dump(), "rows": result.rows[:ROWS_IN_RESPONSE]},
                           row_count=len(result.rows), pod=POD)

    return app
