"""Ask Cube and our compiler the same questions, and compare the numbers.

    docker compose -f ch07/02_bonus_cube/docker-compose.yml up -d
    uv run python ch07/02_bonus_cube/cube_query.py
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "01_main-chapter-code"))

from ch07_semantic import MetricQuery, TimeRange, compile_query, resolve
from ch07_semantic.db import run_query

CUBE_URL = "http://localhost:8007/cubejs-api/v1"
SECRET = "ch07-local-demo-secret-change-me"     # same as docker-compose.yml
TZ = "America/New_York"


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def token(secret: str = SECRET, ttl_s: int = 3600) -> str:
    """HS256 JWT, as Cube expects in the Authorization header (stdlib only)."""
    header = _b64(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    payload = _b64(json.dumps({"exp": int(time.time()) + ttl_s}).encode())
    signature = hmac.new(secret.encode(), f"{header}.{payload}".encode(), hashlib.sha256)
    return f"{header}.{payload}.{_b64(signature.digest())}"


def cube_load(query: dict, timeout_s: float = 60) -> list[dict]:
    """GET /load, retrying while Cube answers {"error": "Continue wait"}."""
    url = f"{CUBE_URL}/load?" + urllib.parse.urlencode({"query": json.dumps(query)})
    deadline = time.monotonic() + timeout_s
    while True:
        req = urllib.request.Request(url, headers={"Authorization": token()})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            body = json.loads(e.read() or b"{}") or {"error": str(e)}
        if body.get("error") != "Continue wait":
            if "error" in body:
                raise RuntimeError(body["error"])
            return body["data"]
        if time.monotonic() > deadline:
            raise TimeoutError("Cube kept answering 'Continue wait'")
        time.sleep(0.5)


def cube_sql(query: dict) -> str:
    url = f"{CUBE_URL}/sql?" + urllib.parse.urlencode({"query": json.dumps(query)})
    req = urllib.request.Request(url, headers={"Authorization": token()})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())["sql"]["sql"][0]


def ours(**kwargs) -> list[dict]:
    return run_query(compile_query(MetricQuery(**kwargs)))


def td(dimension: str, date_range: str) -> list[dict]:
    return [{"dimension": dimension, "dateRange": date_range}]


CASES = [
    ("transactions today (business TZ)",
     {"measures": ["orders.transactions"], "timeDimensions": td("orders.placed_at", "today"),
      "timezone": TZ},
     {"metrics": ["transactions"], "time_range": TimeRange(preset="today")}),
    ("transactions today (Cube default TZ = UTC)",
     {"measures": ["orders.transactions"], "timeDimensions": td("orders.placed_at", "today")},
     {"metrics": ["transactions"], "time_range": TimeRange(preset="today")}),
    ("gross revenue by segment, last month",
     {"measures": ["orders.gross_revenue"], "dimensions": ["customers.segment"],
      "timeDimensions": td("orders.placed_at", "last month"), "timezone": TZ},
     {"metrics": ["gross_revenue"], "dimensions": ["customer.segment"],
      "time_range": TimeRange(preset="last_month")}),
    ("AOV and refund rate, this month",
     {"measures": ["orders.aov", "orders.refund_rate"],
      "timeDimensions": td("orders.placed_at", "this month"), "timezone": TZ},
     {"metrics": ["aov", "refund_rate"], "time_range": TimeRange(preset="this_month")}),
    ("refunded amount by refund date, last month",
     {"measures": ["refunds.refunded_amount"],
      "timeDimensions": td("refunds.created_at", "last month"), "timezone": TZ},
     {"metrics": ["refunded_amount"], "time_range": TimeRange(preset="last_month")}),
    ("active customers, Cube 'last 90 days'",
     {"measures": ["orders.active_customers"],
      "timeDimensions": td("orders.placed_at", "last 90 days"), "timezone": TZ},
     {"metrics": ["active_customers"]}),
    ("active customers, explicit 90 days incl. today",
     {"measures": ["orders.active_customers"],
      "timeDimensions": td("orders.placed_at", "LAST90"), "timezone": TZ},
     {"metrics": ["active_customers"]}),
    ("weekly revenue, Cube 'from 4 weeks ago to 1 week ago'",
     {"measures": ["orders.gross_revenue"],
      "timeDimensions": [{"dimension": "orders.placed_at", "granularity": "week",
                          "dateRange": "from 4 weeks ago to 1 week ago"}],
      "timezone": TZ},
     {"metrics": ["gross_revenue"], "grain": "week",
      "time_range": TimeRange(preset="previous_n", unit="week", n=4)}),
    ("weekly revenue, explicit 4 complete weeks",
     {"measures": ["orders.gross_revenue"],
      "timeDimensions": [{"dimension": "orders.placed_at", "granularity": "week",
                          "dateRange": "PREV4W"}],
      "timezone": TZ},
     {"metrics": ["gross_revenue"], "grain": "week",
      "time_range": TimeRange(preset="previous_n", unit="week", n=4)}),
]


def explicit_range(tr: TimeRange) -> list[str]:
    """Our period as the inclusive [first day, last day] Cube expects."""
    lo, hi = resolve(tr, datetime.now(UTC), TZ)
    return [lo.date().isoformat(), (hi - timedelta(days=1)).date().isoformat()]


EXPLICIT = {
    "LAST90": TimeRange(preset="last_n", unit="day", n=90),
    "PREV4W": TimeRange(preset="previous_n", unit="week", n=4),
}


def _resolve_placeholders(query: dict) -> dict:
    query = json.loads(json.dumps(query))
    for t in query.get("timeDimensions", []):
        if t.get("dateRange") in EXPLICIT:
            t["dateRange"] = explicit_range(EXPLICIT[t["dateRange"]])
    return query


def _values(rows: list[dict], drop: set[str]) -> list[list]:
    out = []
    for row in rows:
        vals = []
        for k, v in row.items():
            if k in drop:        # Cube repeats a time dimension without its granularity suffix
                continue
            try:
                vals.append(round(float(v), 2))
            except (TypeError, ValueError):
                vals.append(str(v)[:10])
        out.append(vals)
    return out


def main() -> None:
    print(f"{'question':52} {'Cube':30} {'compile_query':30} same?")
    for label, cube_q, our_q in CASES:
        cube_q = _resolve_placeholders(cube_q)
        drop = {t["dimension"] for t in cube_q.get("timeDimensions", []) if "granularity" in t}
        c, o = _values(cube_load(cube_q), drop), _values(ours(**our_q), set())
        print(f"{label:52} {json.dumps(sorted(c))[:30]:30} {json.dumps(sorted(o))[:30]:30} "
              f"{'yes' if sorted(c) == sorted(o) else 'NO'}")
    for i in (5, 7):
        q = _resolve_placeholders(CASES[i][1])
        print(f"\nCube's bounds for {CASES[i][0]!r}: {cube_sql_params(q)}")
    print("\nCube's SQL for 'gross revenue by segment, last month':\n")
    print(cube_sql(CASES[2][1]))


def cube_sql_params(query: dict) -> list:
    url = f"{CUBE_URL}/sql?" + urllib.parse.urlencode({"query": json.dumps(query)})
    req = urllib.request.Request(url, headers={"Authorization": token()})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())["sql"]["sql"][1]


if __name__ == "__main__":
    main()
