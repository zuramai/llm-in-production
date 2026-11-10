"""D2: a timeout after the server committed + a retry = the customer gets refunded twice."""

from __future__ import annotations

import json
import random
import time
import uuid
from http.server import BaseHTTPRequestHandler

import httpx
from ch22_common import PORT, Result, serve, timed

from llmprod import connect

ID, TITLE, LAYER, KIND = "D2", "Retry -> duplicate side effects", "Agents & tools", "db"
SLOW_S, CLIENT_TIMEOUT_S = 1.5, 0.5


def setup(conn) -> None:
    conn.execute("CREATE SCHEMA IF NOT EXISTS ch22")
    conn.execute("CREATE TABLE IF NOT EXISTS ch22.refunds (id serial PRIMARY KEY, order_id int, "
                 "amount_cents int, idempotency_key text UNIQUE, created_at timestamptz "
                 "DEFAULT now())")


def refund(conn, order_id: int, amount_cents: int, key: str | None) -> dict:
    """Insert a refund. With a key, a replay returns the original row instead of a new one."""
    row = conn.execute(
        "INSERT INTO ch22.refunds (order_id, amount_cents, idempotency_key) VALUES (%s, %s, %s) "
        "ON CONFLICT (idempotency_key) DO NOTHING RETURNING id", (order_id, amount_cents, key),
    ).fetchone()
    if row is None:  # replay: same key seen before
        row = conn.execute("SELECT id FROM ch22.refunds WHERE idempotency_key = %s",
                           (key,)).fetchone()
        return {"refund_id": row[0], "replayed": True}
    return {"refund_id": row[0], "replayed": False}


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        with connect() as conn:
            out = refund(conn, body["order_id"], body["amount_cents"],
                         self.headers.get("Idempotency-Key"))
        time.sleep(SLOW_S)  # committed, but the payment provider is slow to answer
        payload = json.dumps(out).encode()
        try:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass  # the client gave up long ago; the refund is committed anyway

    def log_message(self, *args):
        pass


def call_with_retries(body: dict, key: str | None, attempts: int = 3) -> int:
    """What every HTTP client with retries does: time out, try again."""
    headers = {"Idempotency-Key": key} if key else {}
    for i in range(1, attempts + 1):
        try:
            httpx.post(f"http://127.0.0.1:{PORT}/refund", json=body, headers=headers,
                       timeout=CLIENT_TIMEOUT_S)
            return i
        except httpx.TimeoutException:
            continue
    return attempts


@timed
def run() -> Result:
    with serve(Handler), connect() as conn:
        setup(conn)
        order_id = 10_000_000 + random.randrange(10**8)  # a fresh fake order id per run
        body = {"order_id": order_id, "amount_cents": 4599}
        call_with_retries(body, key=None)
        # The key is minted once per *intent* (e.g. the tool_call id), not per HTTP attempt.
        call_with_retries(body | {"order_id": order_id + 1}, key=f"refund-{uuid.uuid4()}")
        time.sleep(SLOW_S + 0.2)
        counts = {o: conn.execute("SELECT count(*), coalesce(sum(amount_cents), 0) FROM "
                                  "ch22.refunds WHERE order_id = %s", (o,)).fetchone()
                  for o in (order_id, order_id + 1)}
    naive, keyed = counts[order_id], counts[order_id + 1]
    return Result(
        ID, TITLE, LAYER, KIND,
        reproduced=naive[0] > 1,
        fixed=keyed[0] == 1,
        evidence=f"server commits then answers after {SLOW_S}s, client timeout "
                 f"{CLIENT_TIMEOUT_S}s, 3 attempts: no key -> {naive[0]} refund rows "
                 f"(${naive[1] / 100:.2f}); Idempotency-Key -> {keyed[0]} row "
                 f"(${keyed[1] / 100:.2f})",
        data={"naive_rows": naive[0], "keyed_rows": keyed[0]},
    )
