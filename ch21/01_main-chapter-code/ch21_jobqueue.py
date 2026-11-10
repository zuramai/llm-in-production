"""A Postgres job queue: `FOR UPDATE SKIP LOCKED` to claim, a lease to survive crashed workers.

Tables live in the `ch21` schema of the ShopFlow database. Claiming is one short autocommit
statement; the worker then calls the LLM *outside* any transaction, holding only a lease
(`locked_until`). If it crashes, the lease expires and another worker reclaims the job.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass

import psycopg
from psycopg.types.json import Jsonb

SCHEMA_SQL = """
CREATE SCHEMA IF NOT EXISTS ch21;
CREATE TABLE IF NOT EXISTS ch21.jobs (
    id           bigserial PRIMARY KEY,
    kind         text        NOT NULL,
    payload      jsonb       NOT NULL,
    status       text        NOT NULL DEFAULT 'queued'
                 CHECK (status IN ('queued', 'running', 'done', 'failed')),
    attempts     int         NOT NULL DEFAULT 0,
    max_attempts int         NOT NULL DEFAULT 5,
    run_after    timestamptz NOT NULL DEFAULT now(),
    locked_until timestamptz,
    locked_by    text,
    last_error   text,
    result       jsonb,
    created_at   timestamptz NOT NULL DEFAULT now(),
    updated_at   timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ch21_jobs_queued ON ch21.jobs (run_after, id) WHERE status = 'queued';
CREATE INDEX IF NOT EXISTS ch21_jobs_running ON ch21.jobs (locked_until) WHERE status = 'running';
-- one row per execution attempt, so we can prove no job ran twice concurrently
CREATE TABLE IF NOT EXISTS ch21.job_runs (
    job_id     bigint      NOT NULL,
    attempt    int         NOT NULL,
    worker     text        NOT NULL,
    started_at timestamptz NOT NULL DEFAULT now()
);
"""

_CLAIM = """
WITH next AS (
    SELECT id FROM ch21.jobs
    WHERE (status = 'queued' AND run_after <= now())
       OR (status = 'running' AND locked_until < now() AND attempts < max_attempts)
    ORDER BY run_after, id
    LIMIT %(n)s
    FOR UPDATE {skip}
)
UPDATE ch21.jobs j
SET status = 'running', attempts = j.attempts + 1, locked_by = %(worker)s,
    locked_until = now() + make_interval(secs => %(lease)s), updated_at = now()
FROM next WHERE j.id = next.id
RETURNING j.id, j.kind, j.payload, j.attempts
"""


@dataclass
class Job:
    id: int
    kind: str
    payload: dict
    attempts: int


def setup(conn: psycopg.Connection, reset: bool = False) -> None:
    """Create the queue tables (idempotent). `reset` empties them first."""
    conn.execute(SCHEMA_SQL)
    if reset:
        conn.execute("TRUNCATE ch21.jobs, ch21.job_runs RESTART IDENTITY")


def enqueue(conn: psycopg.Connection, kind: str, payloads: list[dict],
            max_attempts: int = 5) -> list[int]:
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO ch21.jobs (kind, payload, max_attempts) VALUES (%s, %s, %s) RETURNING id",
            [(kind, Jsonb(p), max_attempts) for p in payloads], returning=True,
        )
        ids = []
        while True:
            ids.append(cur.fetchone()[0])
            if not cur.nextset():
                break
    return ids


def claim(conn: psycopg.Connection, worker: str, n: int = 1, lease_s: float = 60,
          skip_locked: bool = True) -> list[Job]:
    """Atomically claim up to `n` runnable jobs (queued, or running with an expired lease)."""
    sql = _CLAIM.format(skip="SKIP LOCKED" if skip_locked else "")
    with conn.transaction():
        rows = conn.execute(sql, {"n": n, "worker": worker, "lease": lease_s}).fetchall()
        for job_id, _, _, attempt in rows:
            conn.execute("INSERT INTO ch21.job_runs (job_id, attempt, worker) VALUES (%s, %s, %s)",
                         (job_id, attempt, worker))
    return [Job(*r) for r in rows]


def complete(conn: psycopg.Connection, job: Job, worker: str, result: dict) -> bool:
    """Mark done, but only if we still own the lease (fencing). False: someone reclaimed it."""
    cur = conn.execute(
        """UPDATE ch21.jobs SET status = 'done', result = %s, locked_until = NULL,
               updated_at = now()
           WHERE id = %s AND status = 'running' AND locked_by = %s AND attempts = %s""",
        (Jsonb(result), job.id, worker, job.attempts),
    )
    return cur.rowcount == 1


def fail(conn: psycopg.Connection, job: Job, worker: str, error: str,
         base_delay_s: float = 1.0, rng: random.Random | None = None) -> str:
    """Re-queue with full-jitter backoff, or mark failed after max_attempts. Returns new status."""
    delay = (rng or random).uniform(0, base_delay_s * 2 ** (job.attempts - 1))
    row = conn.execute(
        """UPDATE ch21.jobs
           SET status = CASE WHEN attempts >= max_attempts THEN 'failed' ELSE 'queued' END,
               run_after = now() + make_interval(secs => %s), last_error = %s,
               locked_until = NULL, updated_at = now()
           WHERE id = %s AND status = 'running' AND locked_by = %s AND attempts = %s
           RETURNING status""",
        (delay, error[:500], job.id, worker, job.attempts),
    ).fetchone()
    return row[0] if row else "lost-lease"


def _drained(conn: psycopg.Connection) -> bool:
    return not conn.execute(
        "SELECT EXISTS (SELECT 1 FROM ch21.jobs WHERE status IN ('queued', 'running'))"
    ).fetchone()[0]


def run_worker(connect: Callable[[], psycopg.Connection], worker: str,
               handler: Callable[[Job], dict], *, lease_s: float = 60, skip_locked: bool = True,
               poll_s: float = 0.02) -> dict:
    """Lease-based worker: claim (short transaction), work outside any transaction, complete.

    Exits once no job is queued or running (a real worker would keep polling). Returns counters.
    """
    stats = {"claimed": 0, "done": 0, "failed": 0, "empty_claims": 0, "claim_ms": 0.0}
    conn = connect()
    try:
        while True:
            t = time.perf_counter()
            jobs = claim(conn, worker, 1, lease_s, skip_locked)
            stats["claim_ms"] += (time.perf_counter() - t) * 1000
            if not jobs:
                stats["empty_claims"] += 1
                if _drained(conn):
                    break
                time.sleep(poll_s)
                continue
            for job in jobs:
                stats["claimed"] += 1
                try:
                    result = handler(job)
                except Exception as exc:  # noqa: BLE001 - a failing job is re-queued, not fatal
                    fail(conn, job, worker, f"{type(exc).__name__}: {exc}")
                    stats["failed"] += 1
                    continue
                if complete(conn, job, worker, result):
                    stats["done"] += 1
    finally:
        conn.close()
    return stats


def run_worker_in_transaction(connect: Callable[[], psycopg.Connection], worker: str,
                              handler: Callable[[Job], dict], *, skip_locked: bool = True,
                              poll_s: float = 0.02) -> dict:
    """The pattern most tutorials show: lock the row and keep the transaction open while the
    (slow) LLM call runs, then mark it done and commit. Shown for comparison only.
    """
    stats = {"claimed": 0, "done": 0, "failed": 0, "empty_claims": 0, "claim_ms": 0.0}
    lock = "FOR UPDATE SKIP LOCKED" if skip_locked else "FOR UPDATE"
    conn = connect()
    try:
        while True:
            with conn.transaction():
                t = time.perf_counter()
                row = conn.execute(
                    "SELECT id, kind, payload, attempts + 1 FROM ch21.jobs WHERE status = 'queued'"
                    f" AND run_after <= now() ORDER BY run_after, id LIMIT 1 {lock}"
                ).fetchone()
                stats["claim_ms"] += (time.perf_counter() - t) * 1000
                if row is not None:
                    job = Job(*row)
                    stats["claimed"] += 1
                    conn.execute("INSERT INTO ch21.job_runs (job_id, attempt, worker) "
                                 "VALUES (%s, %s, %s)", (job.id, job.attempts, worker))
                    result = handler(job)  # an exception rolls back: the job stays queued
                    conn.execute("UPDATE ch21.jobs SET status = 'done', attempts = %s, "
                                 "result = %s, updated_at = now() WHERE id = %s",
                                 (job.attempts, Jsonb(result), job.id))
                    stats["done"] += 1
            if row is None:
                stats["empty_claims"] += 1
                if _drained(conn):
                    break
                time.sleep(poll_s)
    finally:
        conn.close()
    return stats
