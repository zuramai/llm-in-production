"""Shared plumbing for the Chapter 22 repros: result records and a hosted-call cost ledger."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from http.server import ThreadingHTTPServer
from typing import Any

from llmprod import chat_client, get_settings

LUNA = "openai/gpt-6-luna"
HAIKU = "anthropic/claude-haiku-5.5"
SONNET = "anthropic/claude-sonnet-5.5"
QWEN_FLASH = "qwen/qwen3.8-flash"


@dataclass
class Result:
    """What one repro observed. `reproduced`: the failure showed up; `fixed`: the fix held."""

    id: str
    title: str
    layer: str
    kind: str  # hosted | db | local | docker
    reproduced: bool | None
    fixed: bool | None
    evidence: str
    data: dict[str, Any] = field(default_factory=dict)
    cost_usd: float = 0.0
    seconds: float = 0.0

    @property
    def status(self) -> str:
        if self.fixed is None:
            return "PENDING"
        if not self.fixed:
            return "FAIL"
        return "PASS" if self.reproduced else "PASS (not reproduced)"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"status": self.status}


class Ledger:
    """Sums OpenRouter's `usage.cost` over every hosted call a repro makes."""

    def __init__(self) -> None:
        self.cost = 0.0
        self.calls = 0
        self._lock = threading.Lock()

    def add(self, response: Any) -> Any:
        usage = getattr(response, "usage", None)
        with self._lock:
            self.calls += 1
            if usage is not None:
                self.cost += float(getattr(usage, "cost", 0.0) or 0.0)
        return response


LEDGER = Ledger()


def chat(model: str, messages: list[dict], *, max_tokens: int = 200, client=None, **kw) -> Any:
    """One Chat Completions call through the shared client, recorded in the ledger."""
    client = client or chat_client(timeout=60, max_retries=0)
    return LEDGER.add(
        client.chat.completions.create(model=model, messages=messages, max_tokens=max_tokens, **kw)
    )


def timed(fn):
    """Run a repro, fill in its wall time and the hosted spend it caused."""

    def wrapper(*args, **kwargs) -> Result:
        cost0, t0 = LEDGER.cost, time.perf_counter()
        result: Result = fn(*args, **kwargs)
        result.seconds = round(time.perf_counter() - t0, 2)
        result.cost_usd = round(LEDGER.cost - cost0, 6)
        return result

    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    return wrapper


def settings():
    return get_settings()



_CATALOG: dict[str, dict] | None = None


def model_catalog() -> dict[str, dict]:
    """OpenRouter's public model list (free endpoint), keyed by model id, fetched once."""
    global _CATALOG
    if _CATALOG is None:
        import httpx

        resp = httpx.get("https://openrouter.ai/api/v1/models", timeout=30)
        resp.raise_for_status()
        _CATALOG = {m["id"]: m for m in resp.json()["data"]}
    return _CATALOG


PORT = 8022  # the chapter's port (80NN); repros start and stop their own servers in turn


@contextmanager
def serve(handler_cls: type, port: int = PORT) -> Iterator[ThreadingHTTPServer]:
    """Run a stdlib HTTP handler in a background thread for the duration of a repro."""
    server = ThreadingHTTPServer(("127.0.0.1", port), handler_cls)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


def completion_json(content: str, model: str = "stub") -> bytes:
    """A minimal, valid Chat Completions response body."""
    return json.dumps({
        "id": "chatcmpl-stub", "object": "chat.completion", "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
    }).encode()


def run_coro(coro):
    """asyncio.run() that also works inside Jupyter, whose kernel already runs an event loop."""
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with ThreadPoolExecutor(1) as pool:
        return pool.submit(asyncio.run, coro).result()