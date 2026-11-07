"""Small helpers to drive the Docker stack from scripts and the notebook."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
CHECKOUT, PROM, LOKI, GRAFANA = (
    "http://localhost:8011", "http://localhost:9091", "http://localhost:3101", "http://localhost:3011")


def compose(*args: str) -> None:
    subprocess.run(["docker", "compose", *args], cwd=HERE, check=True,
                   capture_output=True, text=True)


def wait_ready(timeout: float = 90) -> None:
    """Block until checkout, Prometheus and Loki all answer."""
    checks = [f"{CHECKOUT}/healthz", f"{PROM}/-/ready", f"{LOKI}/ready"]
    end = time.time() + timeout
    pending = list(checks)
    while pending and time.time() < end:
        for url in list(pending):
            try:
                if httpx.get(url, timeout=2).status_code == 200:
                    pending.remove(url)
            except httpx.HTTPError:
                pass
        time.sleep(1)
    if pending:
        sys.exit(f"not ready after {timeout}s: {pending}")


def fresh_telemetry() -> None:
    """Recreate Prometheus and Loki (no volumes) so an old incident can't leak into the next."""
    compose("rm", "-sfv", "prometheus", "loki")
    compose("up", "-d", "prometheus", "loki")
    wait_ready()
