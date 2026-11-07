"""Generate steady traffic against the checkout service.

    uv run python loadgen.py --rps 6 --duration 120
"""

from __future__ import annotations

import argparse
import asyncio
import random
import time

import httpx

ROUTES = [  # (method, path, weight)
    ("GET", "/products", 0.35),
    ("GET", "/orders/{id}", 0.25),
    ("POST", "/checkout", 0.40),
]


async def one(client: httpx.AsyncClient, stats: dict) -> None:
    method, path, _ = random.choices(ROUTES, weights=[r[2] for r in ROUTES])[0]
    url = path.replace("{id}", str(random.randint(1, 99999)))
    try:
        r = await client.request(method, url)
        stats[r.status_code] = stats.get(r.status_code, 0) + 1
    except httpx.HTTPError:
        stats["client_error"] = stats.get("client_error", 0) + 1


async def run(base_url: str, rps: float, duration: float) -> dict:
    stats: dict = {}
    limits = httpx.Limits(max_connections=200)
    async with httpx.AsyncClient(base_url=base_url, timeout=15, limits=limits) as client:
        tasks, end = set(), time.monotonic() + duration if duration else float("inf")
        while time.monotonic() < end:
            t = asyncio.create_task(one(client, stats))
            tasks.add(t)
            t.add_done_callback(tasks.discard)
            await asyncio.sleep(random.expovariate(rps))  # Poisson arrivals
        await asyncio.gather(*tasks)
    return stats


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://localhost:8011")
    p.add_argument("--rps", type=float, default=6.0)
    p.add_argument("--duration", type=float, default=0, help="seconds; 0 = until Ctrl-C")
    a = p.parse_args()
    try:
        print(asyncio.run(run(a.base_url, a.rps, a.duration)))
    except KeyboardInterrupt:
        pass
