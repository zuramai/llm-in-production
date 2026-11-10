"""ShopFlow support chatbot API.

    cd ch01/01_main-chapter-code && uv run uvicorn app:app --port 8001
"""

from __future__ import annotations

import json
import threading
from collections import OrderedDict
from collections.abc import Iterator
from pathlib import Path

import openai
from chat import SYSTEM_PROMPT, Message, stream_reply, trim_history
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

MAX_SESSIONS = 1_000
MAX_CONTEXT_TOKENS = 8_000  # our own budget, far below the model's window: cost + latency
STATIC = Path(__file__).parent / "static"

app = FastAPI(title="ShopFlow support chatbot")


class SessionStore:
    """In-process LRU of conversations. Lost on restart; one process only (see ch20/ch21)."""

    def __init__(self, max_sessions: int = MAX_SESSIONS):
        self._data: OrderedDict[str, list[Message]] = OrderedDict()
        self._max = max_sessions
        self._lock = threading.Lock()

    def get(self, session_id: str) -> list[Message]:
        with self._lock:
            if session_id in self._data:
                self._data.move_to_end(session_id)
                return list(self._data[session_id])
            return [{"role": "system", "content": SYSTEM_PROMPT}]

    def put(self, session_id: str, messages: list[Message]) -> None:
        with self._lock:
            self._data[session_id] = messages
            self._data.move_to_end(session_id)
            while len(self._data) > self._max:
                self._data.popitem(last=False)


sessions = SessionStore()


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=100)
    message: str = Field(min_length=1, max_length=4000)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/healthz")
async def healthz() -> dict:
    # async: a plain `def` runs in the same thread pool as the chat streams and queues behind them.
    return {"status": "ok"}


@app.post("/chat")
def chat(req: ChatRequest) -> EventSourceResponse:
    def events() -> Iterator[dict]:
        # A sync generator: sse-starlette runs it in a worker thread, so the blocking
        # OpenAI client doesn't stall the event loop (Chapter 21 measures the alternative).
        history = sessions.get(req.session_id) + [{"role": "user", "content": req.message}]
        usage: dict = {}
        parts: list[str] = []
        try:
            history = trim_history(history, MAX_CONTEXT_TOKENS)
            for delta in stream_reply(history, usage=usage):
                parts.append(delta)
                yield {"data": json.dumps({"delta": delta})}
        except (openai.OpenAIError, ValueError) as exc:
            yield {"data": json.dumps({"error": str(exc)})}
            return
        sessions.put(req.session_id, history + [{"role": "assistant", "content": "".join(parts)}])
        yield {"data": json.dumps({"done": True, "usage": usage})}

    return EventSourceResponse(events())
