"""J2: a labeled Prometheus counter has no series until its first increment: 'No data' panels."""

from __future__ import annotations

from ch22_common import Result, timed
from prometheus_client import CollectorRegistry, Counter, generate_latest

ID, TITLE, LAYER, KIND = "J2", "Labeled counter missing until first increment", "Ops", "local"

STATUSES = ("2xx", "4xx", "5xx")


def make_counter(registry: CollectorRegistry, preinit: bool) -> Counter:
    c = Counter("llm_requests", "LLM API requests by status class", ["status"],
                registry=registry)
    if preinit:
        for s in STATUSES:
            c.labels(status=s)  # creates the series at 0 without incrementing
    return c


def series(registry: CollectorRegistry) -> list[str]:
    return [line for line in generate_latest(registry).decode().splitlines()
            if line.startswith("llm_requests_total")]


@timed
def run() -> Result:
    out = {}
    for preinit in (False, True):
        reg = CollectorRegistry()
        c = make_counter(reg, preinit)
        for _ in range(20):
            c.labels(status="2xx").inc()  # a healthy day: no errors yet
        out["pre-initialized" if preinit else "naive"] = series(reg)
    naive_has_5xx = any('status="5xx"' in s for s in out["naive"])
    fixed_has_5xx = any('status="5xx"' in s for s in out["pre-initialized"])
    return Result(
        ID, TITLE, LAYER, KIND,
        reproduced=not naive_has_5xx,
        fixed=fixed_has_5xx,
        evidence=f"after 20 successful requests: naive exports {out['naive']} (so "
                 f"rate(5xx)/rate(all) is empty, not 0); pre-initialized exports "
                 f"{out['pre-initialized']}",
        data=out,
    )
