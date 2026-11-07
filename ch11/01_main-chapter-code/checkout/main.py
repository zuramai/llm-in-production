"""ShopFlow `checkout` service (simulated): FastAPI + Prometheus metrics + JSON logs to Loki.

Metric and log names follow the observability contract in the ShopFlow runbooks
(datasets/shopflow/docs/runbooks/). Faults are switched at runtime through /admin/state.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import sys
import time

from ch11_loki_handler import LokiHandler
from ch11_sim import (
    SCENARIOS,
    FaultState,
    PoolExhausted,
    ProviderTimeout,
    SimPool,
    call_provider,
    query_seconds,
)
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Gauge, Histogram, make_asgi_app
from pydantic import BaseModel

POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "10"))
fault = FaultState(deploy_version=os.getenv("DEPLOY_VERSION", "v0.0.0-0"))
pool = SimPool(size=POOL_SIZE, wait_timeout_s=2.0)

HTTP_REQS = Counter("http_requests_total", "HTTP requests", ["route", "method", "status"])
HTTP_DUR = Histogram("http_request_duration_seconds", "HTTP latency", ["route"],
                     buckets=(.01, .025, .05, .1, .25, .5, 1, 2, 3, 5, 8, 12))
Gauge("db_pool_size", "Configured DB pool size").set(POOL_SIZE)
Gauge("db_pool_connections_in_use", "Connections checked out").set_function(lambda: pool.in_use)
POOL_WAIT = Histogram("db_pool_wait_seconds", "Time waiting for a pooled connection",
                      buckets=(.001, .005, .025, .1, .5, 1, 2, 3))
PROV_DUR = Histogram("payment_provider_request_duration_seconds", "Provider call latency",
                     ["provider"], buckets=(.05, .1, .25, .5, 1, 2, 3, 5, 8, 12))
PROV_ERR = Counter("payment_provider_errors_total", "Provider errors", ["provider", "reason"])
for _p in ("stripe", "paypal"):  # export zeros so rate() works before the first error
    for _r in ("timeout", "declined", "error"):
        PROV_ERR.labels(_p, _r)

log = logging.getLogger("checkout")
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler(sys.stdout))
if os.getenv("LOKI_URL"):
    log.addHandler(LokiHandler(os.environ["LOKI_URL"], service="checkout"))


def emit(level: int, msg: str, **fields) -> None:
    log.log(level, msg, extra={"fields": {"deploy_version": fault.deploy_version, **fields}})


app = FastAPI(title="ShopFlow checkout (simulated)")
app.mount("/metrics", make_asgi_app())


@app.middleware("http")
async def observe(request: Request, call_next):
    t0 = time.perf_counter()
    err = None
    try:
        response = await call_next(request)
        status = response.status_code
    except Exception as exc:  # noqa: BLE001 the bad deploy lands here
        status, err = 500, f"{type(exc).__name__}: {exc}"
        response = JSONResponse({"detail": "internal error"}, status_code=500)
    if request.url.path.startswith("/metrics"):
        return response
    route = request.scope.get("route")
    route_path = route.path if route else "unmatched"
    if route_path.startswith("/admin") or route_path == "/healthz":
        return response
    dur = time.perf_counter() - t0
    HTTP_REQS.labels(route_path, request.method, str(status)).inc()
    HTTP_DUR.labels(route_path).observe(dur)
    fields = {"route": route_path, "status": status, "duration_ms": round(dur * 1000)}
    if err:
        emit(logging.ERROR, "unhandled exception", error=err, **fields)
    elif status < 400:
        emit(logging.INFO, "request completed", **fields)
    return response


async def with_connection(route: str):
    """Hold a pooled connection for the duration of the query. Returns the wait time."""
    try:
        waited = await pool.acquire()
    except PoolExhausted:
        POOL_WAIT.observe(pool.wait_timeout_s)
        raise
    POOL_WAIT.observe(waited)
    try:
        await asyncio.sleep(query_seconds(fault, route))
        if fault.scenario == "bad-deploy" and route == "/checkout" and random.random() < 0.7:
            # the new release dereferences a field that older cart payloads do not have
            discount = None
            return waited * discount["pct"]  # TypeError on purpose
    finally:
        pool.release()
    return waited


def pool_error(route: str) -> JSONResponse:
    emit(logging.ERROR, "db pool exhausted: timed out waiting for connection", route=route,
         status=503, error="PoolExhausted")
    return JSONResponse({"detail": "service busy"}, status_code=503)


@app.get("/products")
async def products():
    try:
        await with_connection("/products")
    except PoolExhausted:
        return pool_error("/products")
    return {"items": 20}


@app.get("/orders/{order_id}")
async def order(order_id: int):
    try:
        await with_connection("/orders/{order_id}")
    except PoolExhausted:
        return pool_error("/orders/{order_id}")
    return {"id": order_id, "status": "paid"}


@app.post("/checkout")
async def checkout():
    try:
        await with_connection("/checkout")
    except PoolExhausted:
        return pool_error("/checkout")
    provider = random.choice(["stripe", "paypal"])
    t0 = time.perf_counter()
    try:
        await call_provider(fault, provider)
    except ProviderTimeout:
        PROV_DUR.labels(provider).observe(time.perf_counter() - t0)
        PROV_ERR.labels(provider, "timeout").inc()
        emit(logging.ERROR, "payment provider timeout", route="/checkout", status=504,
             provider=provider, error="ProviderTimeout")
        return JSONResponse({"detail": "payment provider timeout"}, status_code=504)
    PROV_DUR.labels(provider).observe(time.perf_counter() - t0)
    if random.random() < 0.02:
        PROV_ERR.labels(provider, "declined").inc()
        return JSONResponse({"detail": "card declined"}, status_code=402)
    return {"status": "paid", "provider": provider}


class StateIn(BaseModel):
    scenario: str
    deploy_version: str | None = None
    provider: str | None = None


@app.get("/admin/state")
def get_state():
    return {"scenario": fault.scenario, "deploy_version": fault.deploy_version,
            "provider": fault.provider, "pool_size": pool.size, "pool_in_use": pool.in_use}


@app.post("/admin/state")
def set_state(body: StateIn):
    if body.scenario not in SCENARIOS:
        return JSONResponse({"detail": f"scenario must be one of {SCENARIOS}"}, status_code=422)
    fault.set(body.scenario, body.deploy_version, body.provider)
    return get_state()


@app.get("/healthz")
def healthz():
    return {"status": "ok"}
