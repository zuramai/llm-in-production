"""Simulated infrastructure for the checkout service: a DB pool, payment providers, faults.

No real database or payments are involved. The point is to produce realistic telemetry.
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field

SCENARIOS = ("none", "db-pool-exhaustion", "bad-deploy", "payment-provider-timeout")


class PoolExhausted(Exception):
    pass


class ProviderTimeout(Exception):
    pass


@dataclass
class FaultState:
    scenario: str = "none"
    deploy_version: str = "v0.0.0-0"
    provider: str = "paypal"  # which provider is degraded in payment-provider-timeout
    since: float = field(default_factory=time.time)

    def set(self, scenario: str, deploy_version: str | None = None, provider: str | None = None):
        if scenario not in SCENARIOS:
            raise ValueError(f"unknown scenario {scenario!r}; choose from {SCENARIOS}")
        self.scenario = scenario
        self.since = time.time()
        if deploy_version:
            self.deploy_version = deploy_version
        if provider:
            self.provider = provider


class SimPool:
    """Fixed-size connection pool. Waiters give up after wait_timeout_s."""

    def __init__(self, size: int = 10, wait_timeout_s: float = 2.0):
        self.size, self.wait_timeout_s = size, wait_timeout_s
        self._sem = asyncio.Semaphore(size)
        self.in_use = 0

    async def acquire(self) -> float:
        """Return seconds spent waiting. Raise PoolExhausted if the wait times out."""
        t0 = time.perf_counter()
        try:
            await asyncio.wait_for(self._sem.acquire(), self.wait_timeout_s)
        except TimeoutError:
            raise PoolExhausted("db pool exhausted: timed out waiting for connection") from None
        self.in_use += 1
        return time.perf_counter() - t0

    def release(self) -> None:
        self.in_use -= 1
        self._sem.release()


def query_seconds(fault: FaultState, route: str) -> float:
    """How long a query holds a connection."""
    base = random.uniform(0.02, 0.06)
    if fault.scenario == "db-pool-exhaustion" and route == "/checkout":
        return random.uniform(3.0, 4.5)  # a slow inventory-lock query holds the connection
    return base


async def call_provider(fault: FaultState, provider: str, timeout_s: float = 5.0) -> float:
    """Simulate a charge. Returns the seconds taken, or raises ProviderTimeout."""
    latency = random.uniform(0.08, 0.30)
    if fault.scenario == "payment-provider-timeout" and provider == fault.provider:
        latency = random.uniform(6.0, 9.0)  # far beyond the client timeout
    if latency > timeout_s:
        await asyncio.sleep(timeout_s)
        raise ProviderTimeout(provider)
    await asyncio.sleep(latency)
    return latency
