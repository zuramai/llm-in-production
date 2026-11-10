"""F2: a proxy's idle timeout cuts an SSE stream while the model is silent (thinking, tool call)."""

from __future__ import annotations

import asyncio
import json
import time
from http.server import BaseHTTPRequestHandler

import httpx
from ch22_common import PORT, Result, run_coro, serve, timed

ID, TITLE, LAYER, KIND = "F2", "Streaming disconnects behind proxies", "Infra & serving", "local"

PROXY_IDLE_S = 1.0   # stands in for nginx proxy_read_timeout (60 s) or an LB idle timeout
SILENCE_S = 2.5      # the model goes quiet: reasoning, or waiting on a tool
HEARTBEAT_S = 0.4


def handler_for(heartbeat: bool) -> type:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")  # tell nginx not to buffer
            self.end_headers()
            try:
                self._event({"delta": "Checking your order"})
                deadline = time.monotonic() + SILENCE_S
                while time.monotonic() < deadline:
                    time.sleep(HEARTBEAT_S if heartbeat else SILENCE_S)
                    if heartbeat:
                        self.wfile.write(b": ping\n\n")  # SSE comment: clients ignore it
                        self.wfile.flush()
                for word in (" ... it", " shipped", " today."):
                    self._event({"delta": word})
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _event(self, obj: dict) -> None:
            self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
            self.wfile.flush()

        def log_message(self, *args):
            pass

    return Handler


async def idle_timeout_proxy(upstream_port: int) -> asyncio.base_events.Server:
    """A TCP proxy that hangs up when no bytes flow either way for PROXY_IDLE_S, like most LBs."""

    async def handle(client_r, client_w):
        up_r, up_w = await asyncio.open_connection("127.0.0.1", upstream_port)
        last = [time.monotonic()]

        async def pipe(reader, writer):
            while data := await reader.read(65536):
                last[0] = time.monotonic()
                writer.write(data)
                await writer.drain()

        async def watchdog():
            while time.monotonic() - last[0] < PROXY_IDLE_S:
                await asyncio.sleep(0.05)

        tasks = [asyncio.create_task(c) for c in
                 (pipe(client_r, up_w), pipe(up_r, client_w), watchdog())]
        # Done when upstream finishes (pipe 2) or the connection idles out (watchdog).
        await asyncio.wait(tasks[1:], return_when=asyncio.FIRST_COMPLETED)
        for t in tasks:
            t.cancel()
        for w in (up_w, client_w):
            w.close()

    return await asyncio.start_server(handle, "127.0.0.1", 0)


async def consume(url: str) -> dict:
    """Read the stream; a stream that ends without [DONE] is truncated, not finished."""
    text, done, error = "", False, None
    try:
        async with httpx.AsyncClient(timeout=10) as client, client.stream("GET", url) as r:
            async for line in r.aiter_lines():
                if line == "data: [DONE]":
                    done = True
                elif line.startswith("data: "):
                    text += json.loads(line[6:])["delta"]
    except httpx.HTTPError as e:
        error = type(e).__name__
    return {"text": text, "done": done, "error": error}


async def through_proxy(heartbeat: bool) -> dict:
    with serve(handler_for(heartbeat)):
        proxy = await idle_timeout_proxy(PORT)
        port = proxy.sockets[0].getsockname()[1]
        async with proxy:
            return await consume(f"http://127.0.0.1:{port}/stream")


@timed
def run() -> Result:
    naive = run_coro(through_proxy(heartbeat=False))
    fixed = run_coro(through_proxy(heartbeat=True))
    return Result(
        ID, TITLE, LAYER, KIND,
        reproduced=not naive["done"],
        fixed=fixed["done"] and fixed["text"].endswith("today."),
        evidence=f"proxy idle timeout {PROXY_IDLE_S}s, model silent {SILENCE_S}s: no heartbeat -> "
                 f"got {naive['text']!r}, [DONE]={naive['done']}, error={naive['error']}; "
                 f"': ping' every {HEARTBEAT_S}s -> {fixed['text']!r}, [DONE]={fixed['done']}",
        data={"naive": naive, "fixed": fixed},
    )
