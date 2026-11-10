"""A small async SSE client and load generator for the chat service.

Every request becomes one record: when it started, how it ended (ok / error event / HTTP status /
dropped stream), time to first token, total time, which pod and prompt version served it, and
the reply text. Rollout, autoscaling and canary experiments are all analyses of these records.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from collections.abc import Iterable

import httpx

FAKE_TURNS = re.compile(r"user_messages=(\d+)")


def parse_sse(lines: Iterable[str]) -> list[dict]:
    """Decode `data:` lines of an SSE stream into JSON events (multi-line data joined)."""
    events, data = [], []
    for line in lines:
        if line.startswith("data:"):
            data.append(line[5:].lstrip())
        elif line == "" and data:
            events.append(json.loads("\n".join(data)))
            data = []
    if data:
        events.append(json.loads("\n".join(data)))
    return events


def remembered_turns(text: str) -> int | None:
    """With the fake LLM, how many user messages the service sent upstream (None otherwise)."""
    m = FAKE_TURNS.search(text)
    return int(m.group(1)) if m else None


async def chat_turn(client: httpx.AsyncClient, base_url: str, session_id: str,
                    message: str, t0: float = 0.0) -> dict:
    started = time.perf_counter()
    rec = {"session_id": session_id, "t_start": started - t0, "status": "incomplete",
           "ttft_s": None, "total_s": None, "text": "", "prompt_version": None, "pod": None,
           "usage": None}
    data: list[str] = []
    try:
        async with client.stream("POST", f"{base_url}/chat",
                                 json={"session_id": session_id, "message": message}) as r:
            rec["prompt_version"] = r.headers.get("x-prompt-version")
            rec["pod"] = r.headers.get("x-served-by")
            if r.status_code != 200:
                rec["status"] = f"http_{r.status_code}"
                return rec
            async for line in r.aiter_lines():
                for ev in parse_sse([line, ""]) if line.startswith("data:") else []:
                    if "delta" in ev:
                        if rec["ttft_s"] is None:
                            rec["ttft_s"] = time.perf_counter() - started
                        data.append(ev["delta"])
                    elif "error" in ev:
                        rec["status"] = "error_event"
                        rec["error"] = ev["error"]
                    elif ev.get("done"):
                        rec["status"] = "ok"
                        rec["usage"] = ev.get("usage")
                        rec["prompt_version"] = ev.get("prompt_version", rec["prompt_version"])
                        rec["pod"] = ev.get("pod", rec["pod"])
    except httpx.HTTPError as exc:
        rec["status"] = "incomplete" if data else "connect_error"
        rec["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        rec["total_s"] = time.perf_counter() - started
        rec["text"] = "".join(data)
    return rec


async def run_conversations(base_url: str, messages: list[str], n_sessions: int,
                            concurrency: int = 10, prefix: str = "conv") -> list[dict]:
    """`n_sessions` conversations, each sending `messages` in order. Records carry `turn`."""
    sem = asyncio.Semaphore(concurrency)
    t0 = time.perf_counter()
    records: list[dict] = []

    async def one(i: int, client: httpx.AsyncClient) -> None:
        sid = f"{prefix}-{i}-{uuid.uuid4().hex[:6]}"
        async with sem:
            for turn, msg in enumerate(messages, start=1):
                rec = await chat_turn(client, base_url, sid, msg, t0)
                rec["turn"] = turn
                records.append(rec)

    # No keep-alive: each request opens a new connection, as separate browsers would, so the
    # Service spreads them over pods. A reused connection would stick to one pod.
    limits = httpx.Limits(max_keepalive_connections=0)
    async with httpx.AsyncClient(timeout=httpx.Timeout(60, connect=5), limits=limits) as client:
        await asyncio.gather(*(one(i, client) for i in range(n_sessions)))
    return records


async def run_steady(base_url: str, concurrency: int, duration_s: float,
                     message: str = "How long do I have to return an item?") -> list[dict]:
    """`concurrency` workers send one-turn sessions back to back for `duration_s` seconds."""
    t0 = time.perf_counter()
    records: list[dict] = []

    async def worker(w: int, client: httpx.AsyncClient) -> None:
        n = 0
        while time.perf_counter() - t0 < duration_s:
            rec = await chat_turn(client, base_url, f"steady-{w}-{n}", message, t0)
            records.append(rec)
            n += 1
            if rec["status"] == "connect_error":
                await asyncio.sleep(0.2)

    limits = httpx.Limits(max_connections=concurrency * 2, max_keepalive_connections=0)
    async with httpx.AsyncClient(timeout=httpx.Timeout(60, connect=5), limits=limits) as client:
        await asyncio.gather(*(worker(w, client) for w in range(concurrency)))
    return records
