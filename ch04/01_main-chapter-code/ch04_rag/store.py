"""Postgres storage for chunks: pgvector for meaning, tsvector for words, one table for both."""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from llmprod import connect, embed_client, get_settings

from .chunking import CHUNKERS, Chunk

DOCS_ROOT = Path(__file__).resolve().parents[3] / "datasets" / "shopflow" / "docs"
EMBED_DIM = 768
DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "

# One table holds the chunks of every chunker, told apart by `chunker`. `tsv` is a generated
# column, so Postgres keeps the full-text index in sync for us.
DDL = f"""
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS ch04_doc_chunks (
    id           bigserial PRIMARY KEY,
    chunker      text    NOT NULL CHECK (chunker IN ('fixed', 'heading')),
    doc_path     text    NOT NULL,
    chunk_idx    int     NOT NULL,
    heading_path text    NOT NULL DEFAULT '',
    content      text    NOT NULL,
    token_count  int     NOT NULL,
    embedding    vector({EMBED_DIM}) NOT NULL,
    tsv          tsvector GENERATED ALWAYS AS (
                     to_tsvector('english', heading_path || ' ' || content)) STORED,
    UNIQUE (chunker, doc_path, chunk_idx)
);
CREATE INDEX IF NOT EXISTS ch04_chunks_hnsw ON ch04_doc_chunks
    USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
CREATE INDEX IF NOT EXISTS ch04_chunks_fts ON ch04_doc_chunks USING gin (tsv);
"""


def open_db() -> psycopg.Connection:
    """Admin connection (we create and fill our own table) with the pgvector adapter loaded."""
    conn = connect()
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    # pgvector 0.8: if a filter (here: chunker) throws away index hits, keep scanning the graph
    # instead of returning fewer rows than LIMIT.
    conn.execute("SET hnsw.iterative_scan = strict_order")
    return conn


def setup(conn: psycopg.Connection, reset: bool = False) -> None:
    """Idempotent. `reset=True` drops only ch04's own table first."""
    if reset:
        conn.execute("DROP TABLE IF EXISTS ch04_doc_chunks")
    conn.execute(DDL)


def load_docs(root: Path = DOCS_ROOT) -> dict[str, str]:
    """{relative path: text} for every Markdown file under the docs root, sorted."""
    return {p.relative_to(root).as_posix(): p.read_text() for p in sorted(root.rglob("*.md"))}


def make_chunks(docs: dict[str, str], chunker: str) -> list[Chunk]:
    fn = CHUNKERS[chunker]
    return [c for path, text in docs.items() for c in fn(text, path)]


def embed_texts(
    texts: Sequence[str], prefix: str = DOC_PREFIX, batch: int = 32
) -> np.ndarray:
    """Embed through the OpenAI-compatible endpoint (Ollama `nomic-embed-text`, 768-d)."""
    client, model = embed_client(), get_settings().embed_model
    vecs: list[list[float]] = []
    for i in range(0, len(texts), batch):
        resp = client.embeddings.create(
            model=model, input=[prefix + t for t in texts[i : i + batch]]
        )
        vecs.extend(d.embedding for d in resp.data)
    arr = np.asarray(vecs, dtype=np.float32)
    if arr.shape != (len(texts), EMBED_DIM):
        raise RuntimeError(f"expected {(len(texts), EMBED_DIM)} embeddings, got {arr.shape}")
    return arr


def ingest(
    conn: psycopg.Connection,
    docs: dict[str, str],
    chunker: str,
    prefix: str = DOC_PREFIX,
) -> dict:
    """Replace all rows of one chunker with freshly chunked + embedded docs. Returns stats."""
    t0 = time.perf_counter()
    chunks = make_chunks(docs, chunker)
    vecs = embed_texts([c.embed_text for c in chunks], prefix)
    with conn.transaction():
        conn.execute("DELETE FROM ch04_doc_chunks WHERE chunker = %s", (chunker,))
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO ch04_doc_chunks (chunker, doc_path, chunk_idx, heading_path, content,"
                " token_count, embedding) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                [
                    (c.chunker, c.doc_path, c.chunk_idx, c.heading_path, c.content,
                     c.token_count, v)
                    for c, v in zip(chunks, vecs, strict=True)
                ],
            )
    tokens = [c.token_count for c in chunks]
    return {
        "chunker": chunker,
        "docs": len(docs),
        "chunks": len(chunks),
        "tokens_min": min(tokens),
        "tokens_mean": round(sum(tokens) / len(tokens), 1),
        "tokens_max": max(tokens),
        "seconds": round(time.perf_counter() - t0, 1),
    }


def counts(conn: psycopg.Connection) -> dict[str, int]:
    rows = conn.execute(
        "SELECT chunker, count(*) FROM ch04_doc_chunks GROUP BY chunker ORDER BY chunker"
    ).fetchall()
    return {r[0]: r[1] for r in rows}


def fetch_chunks(conn: psycopg.Connection, chunker: str) -> Iterable[tuple]:
    return conn.execute(
        "SELECT id, doc_path, chunk_idx, heading_path, content, embedding"
        " FROM ch04_doc_chunks WHERE chunker = %s ORDER BY id",
        (chunker,),
    ).fetchall()
