"""F3: a sync `def healthz()` waits behind busy sync handlers, so liveness probes time out."""

from __future__ import annotations

import asyncio
import threading
import time

import httpx
import uvicorn
from ch22_common import PORT, Result, run_coro, timed
from fastapi import FastAPI

ID, TITLE, LAYER, KIND = "F3", "Sync health check starves under load", "Infra & serving", "local"

SLOW_S, N_BUSY, PROBE_TIMEOUT_S = 2.0, 60, 1.0  # k8s probe timeoutSeconds defaults to 1

app = FastAPI()


@app.get("/chat")
def chat_sync() -> dict:
    time.sleep(SLOW_S)  # a sync LLM client call holding a threadpool worker
    return {"ok": True}


@app.get("/healthz-sync")
def healthz_sync() -> dict:  # runs in the same 40-thread pool as /chat
    return {"ok": True}


@app.get("/healthz")
async def healthz() -> dict:  # runs on the event loop: answers even when the pool is full
    return {"ok": True}


async def probe(client: httpx.AsyncClient, path: str) -> float | None:
    t0 = time.perf_counter()
    try:
        await client.get(f"http://127.0.0.1:{PORT}{path}", timeout=PROBE_TIMEOUT_S)
        return time.perf_counter() - t0
    except httpx.TimeoutException:
        return None


async def load_and_probe() -> dict:
    async with httpx.AsyncClient(timeout=30,
                                 limits=httpx.Limits(max_connections=200)) as client:
        busy = [asyncio.create_task(client.get(f"http://127.0.0.1:{PORT}/chat"))
                for _ in range(N_BUSY)]
        await asyncio.sleep(0.3)  # let the pool fill up
        sync_s, async_s = await asyncio.gather(probe(client, "/healthz-sync"),
                                               probe(client, "/healthz"))
        await asyncio.gather(*busy)
    return {"sync": sync_s, "async": async_s}


@timed
def run() -> Result:
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=PORT, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.05)
    try:
        out = run_coro(load_and_probe())
    finally:
        server.should_exit = True
        thread.join(10)

    def fmt(x):
        return f"timed out (> {PROBE_TIMEOUT_S:.0f} s)" if x is None else f"{x * 1000:.0f} ms"

    return Result(
        ID, TITLE, LAYER, KIND,
        reproduced=out["sync"] is None or out["sync"] > PROBE_TIMEOUT_S,
        fixed=out["async"] is not None and out["async"] < 0.2,
        evidence=f"{N_BUSY} in-flight sync /chat requests ({SLOW_S:.0f} s each, 40-thread pool): "
                 f"def healthz -> {fmt(out['sync'])}; async def healthz -> {fmt(out['async'])}",
        data={k: v for k, v in out.items()},
    )
