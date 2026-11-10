"""C1: pgvector HNSW + WHERE filter: the index returns ef_search candidates, the filter eats them."""

from __future__ import annotations

import random

from ch22_common import Result, timed

from llmprod import connect

ID, TITLE, LAYER, KIND = "C1", "Filtered HNSW query returns fewer rows than LIMIT", "Retrieval", "db"

N_ROWS, DIM, N_TENANTS, K = 20_000, 16, 50, 30  # a tenant filter keeps 2% of rows


def setup(conn) -> None:
    """Idempotent: our own schema, rebuilt only if the row count is off."""
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.execute("CREATE SCHEMA IF NOT EXISTS ch22")
    conn.execute(f"CREATE TABLE IF NOT EXISTS ch22.vec_items (id int PRIMARY KEY, tenant int, "
                 f"embedding vector({DIM}))")
    if conn.execute("SELECT count(*) FROM ch22.vec_items").fetchone()[0] == N_ROWS:
        return
    conn.execute("TRUNCATE ch22.vec_items")
    rng = random.Random(22)
    with conn.cursor().copy("COPY ch22.vec_items (id, tenant, embedding) FROM STDIN") as cp:
        for i in range(N_ROWS):
            vec = "[" + ",".join(f"{rng.gauss(0, 1):.4f}" for _ in range(DIM)) + "]"
            cp.write_row((i, i % N_TENANTS, vec))
    conn.execute("CREATE INDEX IF NOT EXISTS ch22_vec_hnsw ON ch22.vec_items "
                 "USING hnsw (embedding vector_l2_ops)")
    conn.execute("ANALYZE ch22.vec_items")


def search(conn, query_vec: str, tenant: int, iterative: str) -> int:
    conn.execute(f"SET hnsw.iterative_scan = {iterative}")
    rows = conn.execute(
        "SELECT id FROM ch22.vec_items WHERE tenant = %s ORDER BY embedding <-> %s::vector "
        "LIMIT %s", (tenant, query_vec, K)).fetchall()
    return len(rows)


@timed
def run() -> Result:
    with connect() as conn:
        setup(conn)
        conn.execute("SET enable_seqscan = off")  # what the planner does on a big table
        q = "[" + ",".join("0.1" for _ in range(DIM)) + "]"
        plan = "\n".join(r[0] for r in conn.execute(
            "EXPLAIN SELECT id FROM ch22.vec_items WHERE tenant = 7 "
            "ORDER BY embedding <-> %s::vector LIMIT 30", (q,)).fetchall())
        got = {mode: search(conn, q, 7, mode) for mode in ("off", "relaxed_order", "strict_order")}
        ef = conn.execute("SHOW hnsw.ef_search").fetchone()[0]
    return Result(
        ID, TITLE, LAYER, KIND,
        reproduced=got["off"] < K and "ch22_vec_hnsw" in plan,
        fixed=got["relaxed_order"] == K and got["strict_order"] == K,
        evidence=f"{N_ROWS:,} rows, filter keeps {100 / N_TENANTS:.0f}%, ef_search={ef}: asked "
                 f"{K}, got " + ", ".join(f"{v} (iterative_scan={k})" for k, v in got.items()),
        data=got,
    )
