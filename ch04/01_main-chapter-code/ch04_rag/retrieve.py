"""Four ways to ask "which chunks answer this question?".

    vector         nearest neighbours by meaning (pgvector, cosine, HNSW)
    fts            Postgres full-text search: matching words, ranked by ts_rank_cd
    hybrid         vector + fts, fused with Reciprocal Rank Fusion (k=60)
    hybrid_rerank  hybrid candidates re-scored by a cross-encoder
"""

from __future__ import annotations

import re
import threading
import time
from collections import defaultdict
from collections.abc import Hashable, Sequence
from dataclasses import dataclass, field

import psycopg

from .store import QUERY_PREFIX, embed_texts

MODES = ("vector", "fts", "hybrid", "hybrid_rerank")
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RRF_K = 60


@dataclass
class Hit:
    chunk_id: int
    doc_path: str
    chunk_idx: int
    heading_path: str
    content: str
    score: float = 0.0
    rank: int = 0
    source: str = ""  # which mode produced this ranking
    extra: dict = field(default_factory=dict)

    @property
    def citation(self) -> str:
        return f"[doc:{self.doc_path}#{self.chunk_idx}]"

    @property
    def embed_text(self) -> str:
        return f"{self.heading_path}\n{self.content}" if self.heading_path else self.content


def rrf(
    rankings: Sequence[Sequence[Hashable]], k: int = RRF_K
) -> list[tuple[Hashable, float]]:
    """Reciprocal Rank Fusion: score(d) = sum over rankings of 1 / (k + rank(d)), rank from 1.

    Only ranks matter, never raw scores, so a cosine distance and a ts_rank can be fused without
    putting them on one scale. An item missing from a ranking contributes nothing from it.
    Ties keep first-seen order, so the result is deterministic.
    """
    scores: dict[Hashable, float] = defaultdict(float)
    first_seen: dict[Hashable, int] = {}
    for ranking in rankings:
        seen: set[Hashable] = set()
        for rank, item in enumerate(ranking, start=1):
            if item in seen:  # a duplicate in one list must not count twice
                continue
            seen.add(item)
            scores[item] += 1.0 / (k + rank)
            first_seen.setdefault(item, len(first_seen))
    return sorted(scores.items(), key=lambda kv: (-kv[1], first_seen[kv[0]]))


_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.\-/$<>]*")


def or_terms(question: str) -> list[str]:
    """Words of the question, each of which becomes one OR-ed full-text term."""
    return [w.rstrip(".-") for w in _WORD.findall(question)]


_COLS = "id, doc_path, chunk_idx, heading_path, content"


def _hits(rows, source: str) -> list[Hit]:
    return [
        Hit(r[0], r[1], r[2], r[3], r[4], score=float(r[5]), rank=i, source=source)
        for i, r in enumerate(rows, start=1)
    ]


def vector_search(
    conn: psycopg.Connection, query_vec, chunker: str, k: int
) -> list[Hit]:
    rows = conn.execute(
        f"SELECT {_COLS}, 1 - (embedding <=> %s) AS cos FROM ch04_doc_chunks"
        " WHERE chunker = %s ORDER BY embedding <=> %s LIMIT %s",
        (query_vec, chunker, query_vec, k),
    ).fetchall()
    return _hits(rows, "vector")


def fts_search(conn: psycopg.Connection, question: str, chunker: str, k: int) -> list[Hit]:
    """OR over the question's words. `websearch_to_tsquery` ANDs them, and an AND of every word
    in a natural-language question matches almost nothing."""
    terms = or_terms(question)
    if not terms:
        return []
    query = " || ".join(["plainto_tsquery('english', %s)"] * len(terms))
    rows = conn.execute(
        f"WITH q AS (SELECT {query} AS tsq) SELECT {_COLS}, ts_rank_cd(tsv, q.tsq) AS score"
        " FROM ch04_doc_chunks, q WHERE chunker = %s AND tsv @@ q.tsq"
        " ORDER BY score DESC, id LIMIT %s",
        (*terms, chunker, k),
    ).fetchall()
    return _hits(rows, "fts")


class Retriever:
    """Holds the DB connection and (lazily) the cross-encoder. One instance per thread of work."""

    _ce_lock = threading.Lock()
    _ce = None

    def __init__(
        self,
        conn: psycopg.Connection,
        pool: int = 30,
        query_prefix: str = QUERY_PREFIX,
        cache_queries: bool = False,
    ):
        self.conn = conn
        self.pool = pool  # candidates per retriever before fusion / reranking
        self.query_prefix = query_prefix
        self.last_timing: dict[str, float] = {}
        # Eval runs the same question under 8 configs; cache so we embed it once.
        self._qcache: dict[str, object] | None = {} if cache_queries else None

    @classmethod
    def cross_encoder(cls):
        with cls._ce_lock:
            if cls._ce is None:
                from sentence_transformers import CrossEncoder

                cls._ce = CrossEncoder(RERANK_MODEL, max_length=512)
            return cls._ce

    def embed_query(self, question: str):
        if self._qcache is not None and question in self._qcache:
            return self._qcache[question]
        vec = embed_texts([question], prefix=self.query_prefix)[0]
        if self._qcache is not None:
            self._qcache[question] = vec
        return vec

    def retrieve(self, question: str, mode: str, chunker: str, k: int = 5) -> list[Hit]:
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}; choose from {MODES}")
        timing: dict[str, float] = {}
        t0 = time.perf_counter()
        if mode == "fts":
            hits = fts_search(self.conn, question, chunker, k)
            timing["fts_ms"] = (time.perf_counter() - t0) * 1000
            self.last_timing = timing
            return hits
        qv = self.embed_query(question)
        timing["embed_ms"] = (time.perf_counter() - t0) * 1000
        t1 = time.perf_counter()
        if mode == "vector":
            hits = vector_search(self.conn, qv, chunker, k)
            timing["vector_ms"] = (time.perf_counter() - t1) * 1000
            self.last_timing = timing
            return hits
        vec = vector_search(self.conn, qv, chunker, self.pool)
        timing["vector_ms"] = (time.perf_counter() - t1) * 1000
        t2 = time.perf_counter()
        fts = fts_search(self.conn, question, chunker, self.pool)
        timing["fts_ms"] = (time.perf_counter() - t2) * 1000
        by_id = {h.chunk_id: h for h in [*vec, *fts]}
        fused = rrf([[h.chunk_id for h in vec], [h.chunk_id for h in fts]])
        ranked = []
        for i, (cid, score) in enumerate(fused, start=1):
            h = by_id[cid]
            ranked.append(Hit(h.chunk_id, h.doc_path, h.chunk_idx, h.heading_path, h.content,
                              score=score, rank=i, source="hybrid"))
        if mode == "hybrid":
            self.last_timing = timing
            return ranked[:k]
        t3 = time.perf_counter()
        candidates = ranked[: self.pool]
        scores = self.cross_encoder().predict(
            [(question, h.embed_text) for h in candidates], show_progress_bar=False
        )
        order = sorted(zip(candidates, scores, strict=True), key=lambda p: -float(p[1]))
        timing["rerank_ms"] = (time.perf_counter() - t3) * 1000
        self.last_timing = timing
        return [
            Hit(h.chunk_id, h.doc_path, h.chunk_idx, h.heading_path, h.content,
                score=float(s), rank=i, source="hybrid_rerank")
            for i, (h, s) in enumerate(order[:k], start=1)
        ]
