"""Locust load test for the chat API: streaming chats plus a health-check probe.

    cd ch21/01_main-chapter-code
    uv run locust -f ch21_locustfile.py --headless -u 50 -r 25 -t 40s --host http://localhost:8021

Each simulated customer keeps one session and asks FAQ questions with 0.5–1.5 s of "think time"
in between. We report two custom metrics per chat, because a streaming response is not done
when its headers arrive: `ttft` (first delta) and `chat` (the final `done` event). A single
`HealthProbe` user hits `/healthz` four times a second; if the server's event loop or thread
pool is stuck, this is where it shows.

If CH21_EVENTS is set, every request is also appended to that CSV file (ts, name, ms, ok,
error, provider) so the notebook can compute exact percentiles and timelines.
"""

from __future__ import annotations

import csv
import json
import os
import random
import time
import uuid

from locust import HttpUser, between, constant, events, task

QUESTIONS = [
    "How long do I have to return an item?",
    "Who pays for return shipping?",
    "My payment failed but I see a charge on my card. What's going on?",
    "Do you ship to Canada?",
    "How much is express shipping?",
    "Can I change my delivery address after ordering?",
]
THINK = (float(os.environ.get("CH21_THINK_MIN", "0.5")), float(os.environ.get("CH21_THINK_MAX",
                                                                             "1.5")))
_rows: list[tuple] = []


def _record(environment, name: str, ms: float, exc: Exception | None, provider: str = "") -> None:
    environment.events.request.fire(
        request_type="SSE", name=name, response_time=ms, response_length=0,
        exception=exc, context={},
    )
    _rows.append((time.time(), name, round(ms, 2), int(exc is None),
                  "" if exc is None else str(exc)[:80], provider))


class ChatUser(HttpUser):
    wait_time = between(*THINK)

    def on_start(self) -> None:
        self.session_id = uuid.uuid4().hex

    @task
    def chat(self) -> None:
        payload = {"session_id": self.session_id, "message": random.choice(QUESTIONS)}
        start = time.perf_counter()
        ttft = None
        err: Exception | None = None
        provider = ""
        try:
            with self.client.post("/chat", json=payload, stream=True, timeout=90,
                                  name="/chat (headers)", catch_response=True) as resp:
                if resp.status_code != 200:
                    err = RuntimeError(f"HTTP {resp.status_code}")
                    resp.failure(err)
                else:
                    done = False
                    for line in resp.iter_lines(decode_unicode=True):
                        if not line or not line.startswith("data:"):
                            continue
                        event = json.loads(line[5:])
                        if "delta" in event and ttft is None:
                            ttft = (time.perf_counter() - start) * 1000
                            _record(self.environment, "ttft", ttft, None)
                        elif "error" in event:
                            err = RuntimeError(f"SSE error: {event['error'][:60]}")
                            break
                        elif event.get("done"):
                            done = True
                            provider = event.get("provider") or ""
                            break
                    if err is None and not done:
                        err = RuntimeError("stream ended without done")
                    resp.success()
        except Exception as exc:  # noqa: BLE001 - reset, read timeout...: a failed chat
            err = exc
        _record(self.environment, "chat", (time.perf_counter() - start) * 1000, err, provider)


class HealthProbe(HttpUser):
    fixed_count = 1
    wait_time = constant(0.25)

    @task
    def health(self) -> None:
        start = time.perf_counter()
        err: Exception | None = None
        try:
            r = self.client.get("/healthz", timeout=30, name="/healthz (locust)")
            r.raise_for_status()
        except Exception as exc:  # noqa: BLE001 - any failure is a failed probe
            err = exc
        _record(self.environment, "healthz", (time.perf_counter() - start) * 1000, err)


@events.quitting.add_listener
def _dump(environment, **_kwargs) -> None:
    path = os.environ.get("CH21_EVENTS")
    if not path:
        return
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ts", "name", "ms", "ok", "error", "provider"])
        w.writerows(_rows)
