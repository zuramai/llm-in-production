"""Start the mock LLM and the chat API, run Locust headless, and summarize the results.

Used by the notebook; also runnable:  uv run python ch21_loadtest.py async 50
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from statistics import quantiles

CODE_DIR = Path(__file__).resolve().parent
RESULTS = CODE_DIR / "results"
LOGS = RESULTS / "logs"
MOCK_PORT, API_PORT = 8121, 8021


def _get(url: str, timeout: float = 2.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def post_json(url: str, body: dict) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read())


@contextmanager
def server(app: str, port: int, env: dict[str, str] | None = None, ready_path: str = "/healthz",
           workers: int = 1, name: str | None = None) -> Iterator[str]:
    """Run `uvicorn <app>` on `port` until the block exits. Yields the base URL."""
    LOGS.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "uvicorn", app, "--port", str(port), "--no-access-log",
           "--log-level", "warning", "--workers", str(workers)]
    base = f"http://127.0.0.1:{port}"
    with open(LOGS / f"{name or app.split(':')[0]}-{port}.log", "w") as log:
        proc = subprocess.Popen(cmd, cwd=CODE_DIR, env={**os.environ, **(env or {})},
                                stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.time() + 30
            while True:
                if proc.poll() is not None:
                    raise RuntimeError(f"{app} exited with {proc.returncode}; see {log.name}")
                try:
                    _get(base + ready_path, timeout=1)
                    break
                except OSError:
                    if time.time() > deadline:
                        raise
                    time.sleep(0.2)
            yield base
        finally:
            proc.terminate()
            try:
                proc.wait(10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


def mock_env(**cfg) -> dict[str, str]:
    return {f"MOCK_{k.upper()}": "none" if v is None else str(v) for k, v in cfg.items()}


def run_locust(host: str, users: int, duration_s: int, tag: str, spawn_rate: float | None = None,
               think: tuple[float, float] = (0.5, 1.5)) -> Path:
    """Run Locust headless (`users` chat users + 1 health probe). Returns the events CSV path."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    events = LOGS / f"{tag}.events.csv"
    cmd = [sys.executable, "-m", "locust", "-f", "ch21_locustfile.py", "--headless",
           "-u", str(users + 1), "-r", str(spawn_rate or max(1, users / 5)),
           "-t", f"{duration_s}s", "--host", host, "--only-summary", "--stop-timeout", "5",
           "--csv", str(LOGS / tag)]
    env = {**os.environ, "CH21_EVENTS": str(events),
           "CH21_THINK_MIN": str(think[0]), "CH21_THINK_MAX": str(think[1])}
    with open(LOGS / f"{tag}.locust.log", "w") as log:
        subprocess.run(cmd, cwd=CODE_DIR, env=env, stdout=log, stderr=subprocess.STDOUT,
                       check=False, timeout=duration_s + 120)
    return events


def load_events(path: Path) -> list[dict]:
    with open(path) as f:
        return [{**r, "ts": float(r["ts"]), "ms": float(r["ms"]), "ok": r["ok"] == "1"}
                for r in csv.DictReader(f)]


def pct(values: list[float], p: int) -> float | None:
    """p-th percentile (inclusive method); None for an empty list."""
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return quantiles(values, n=100, method="inclusive")[p - 1]


def summarize(rows: list[dict], warmup_s: float = 5.0) -> dict:
    """Steady-state stats: drop the first `warmup_s` seconds (ramp-up) of the run."""
    if not rows:
        return {}
    t0 = min(r["ts"] for r in rows) + warmup_s
    t1 = max(r["ts"] for r in rows)
    steady = [r for r in rows if r["ts"] >= t0]
    window = max(1e-9, t1 - t0)

    def ms(name: str, ok: bool = True) -> list[float]:
        return [r["ms"] for r in steady if r["name"] == name and r["ok"] == ok]

    chats_ok, chats_failed = ms("chat"), ms("chat", ok=False)
    health = ms("healthz")
    out = {
        "window_s": round(window, 1),
        "chats_ok": len(chats_ok),
        "chats_failed": len(chats_failed),
        "error_rate": round(len(chats_failed) / max(1, len(chats_ok) + len(chats_failed)), 4),
        "throughput_rps": round(len(chats_ok) / window, 2),
    }
    for name, vals in (("ttft", ms("ttft")), ("chat", chats_ok), ("healthz", health)):
        for p in (50, 95):
            v = pct(vals, p)
            out[f"{name}_p{p}_ms"] = None if v is None else round(v)
    all_health = health + ms("healthz", ok=False)
    out["healthz_max_ms"] = round(max(all_health)) if all_health else None
    out["healthz_failed"] = len(all_health) - len(health)
    errors: dict[str, int] = {}
    for r in steady:
        if not r["ok"]:
            errors[r["error"]] = errors.get(r["error"], 0) + 1
    out["errors"] = errors
    return out


def run_experiment(mode: str, users: int, duration_s: int = 40, app_env: dict | None = None,
                   mock_cfg: dict | None = None, tag: str | None = None,
                   think: tuple[float, float] = (0.5, 1.5), workers: int = 1) -> dict:
    """Mock LLM + chat API in `mode` (`workers` uvicorn processes) + Locust with `users` users.

    Saves the summary to results/<tag>.json and returns it.
    """
    tag = tag or f"{mode}-u{users}"
    cfg = {"ttft_ms": 800, "tokens_per_s": 120, "output_tokens": 60, **(mock_cfg or {})}
    env = {"CH21_MODE": mode, "CH21_LLM_BASE_URLS": f"http://127.0.0.1:{MOCK_PORT}/v1",
           **(app_env or {})}
    with server("mock_llm_server:app", MOCK_PORT, mock_env(**cfg), ready_path="/v1/models"):
        with server("ch21_chat_api:app", API_PORT, env, name=f"api-{tag}",
                    workers=workers) as api:
            events = run_locust(api, users, duration_s, tag, think=think)
        mock_stats = _get(f"http://127.0.0.1:{MOCK_PORT}/admin/stats")
    summary = {"tag": tag, "mode": mode, "users": users, "workers": workers,
               "duration_s": duration_s, "app_env": app_env or {}, "mock": cfg,
               **summarize(load_events(events)),
               "mock_requests": mock_stats["requests"],
               "mock_max_in_flight": mock_stats["max_in_flight"]}
    (RESULTS / f"{tag}.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    print(json.dumps(run_experiment(sys.argv[1], int(sys.argv[2])), indent=2))
