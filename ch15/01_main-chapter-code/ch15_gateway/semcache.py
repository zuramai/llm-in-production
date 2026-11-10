"""A semantic cache on Postgres + pgvector: serve a stored answer when a new question means the same.

An exact cache (Redis, in the gateway) only hits on byte-identical requests. A semantic cache
embeds the question and returns the nearest stored answer if its cosine similarity is at least
`threshold`. That's the whole trick, and the threshold is the whole risk: too low and it serves
the answer to a *different* question (a false hit).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass

import psycopg
from pgvector.psycopg import register_vector

Embedder = Callable[[Sequence[str]], list[list[float]]]

TABLE = "ch15_semcache"
DIM = 768  # nomic-embed-text

# Questions that mention a specific order, email or amount are personal: never share their answers.
_PERSONAL = re.compile(
    r"[\w.+-]+@[\w-]+\.[\w.]+"        # email address
    r"|#\s?\d{3,}"                     # "#10234"
    r"|\border\s+(?:no\.?|number)?\s*\d{3,}"
    r"|\$\s?\d",                       # amounts
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Hit:
    prompt: str
    response: str
    similarity: float


def setup(conn: psycopg.Connection) -> None:
    """Create the cache table and its HNSW index (idempotent)."""
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE} (
            id         bigserial PRIMARY KEY,
            namespace  text        NOT NULL,   -- model + system prompt version
            prompt     text        NOT NULL,
            embedding  vector({DIM}) NOT NULL,
            response   text        NOT NULL,
            created_at timestamptz NOT NULL DEFAULT now()
        )""")
    conn.execute(f"""
        CREATE INDEX IF NOT EXISTS {TABLE}_hnsw
        ON {TABLE} USING hnsw (embedding vector_cosine_ops)""")
    conn.execute(f"CREATE INDEX IF NOT EXISTS {TABLE}_ns ON {TABLE} (namespace, created_at)")
    register_vector(conn)


def is_cacheable(messages: Sequence[dict[str, str]]) -> bool:
    """Only single-turn, impersonal questions may be served from a shared cache.

    A follow-up ("and for VIPs?") means nothing without its conversation, and a question about
    one customer's order must never be answered with another customer's answer.
    """
    turns = [m for m in messages if m["role"] != "system"]
    if len(turns) != 1 or turns[0]["role"] != "user":
        return False
    return _PERSONAL.search(turns[0]["content"]) is None


def normalize(text: str) -> str:
    """Collapse whitespace and case so trivial variants embed (and hash) identically."""
    return " ".join(text.split()).lower()


class SemanticCache:
    def __init__(self, conn: psycopg.Connection, embed: Embedder, namespace: str,
                 threshold: float = 0.9, ttl_seconds: int | None = 86_400) -> None:
        if not 0 < threshold <= 1:
            raise ValueError(f"threshold must be in (0, 1], got {threshold}")
        self.conn, self.embed, self.namespace = conn, embed, namespace
        self.threshold, self.ttl_seconds = threshold, ttl_seconds
        register_vector(conn)

    def nearest(self, prompt: str) -> Hit | None:
        """The closest live entry in this namespace, whatever its similarity."""
        [vec] = self.embed([normalize(prompt)])
        row = self.conn.execute(
            f"""SELECT prompt, response, 1 - (embedding <=> %(v)s::vector) AS similarity
                FROM {TABLE}
                WHERE namespace = %(ns)s
                  AND (%(ttl)s::int IS NULL OR created_at > now() - make_interval(secs => %(ttl)s))
                ORDER BY embedding <=> %(v)s::vector
                LIMIT 1""",
            {"v": vec, "ns": self.namespace, "ttl": self.ttl_seconds},
        ).fetchone()
        return Hit(row[0], row[1], float(row[2])) if row else None

    def lookup(self, prompt: str) -> Hit | None:
        """Return a hit only if the nearest entry is at least `threshold` similar."""
        hit = self.nearest(prompt)
        return hit if hit is not None and hit.similarity >= self.threshold else None

    def store(self, prompt: str, response: str) -> None:
        [vec] = self.embed([normalize(prompt)])
        self.conn.execute(
            f"INSERT INTO {TABLE} (namespace, prompt, embedding, response) "
            "VALUES (%s, %s, %s::vector, %s)",
            (self.namespace, prompt, vec, response),
        )

    def clear(self) -> None:
        self.conn.execute(f"DELETE FROM {TABLE} WHERE namespace = %s", (self.namespace,))


@dataclass(frozen=True)
class SweepPoint:
    threshold: float
    served: int         # probes answered from the cache
    correct: int        # served with an answer that is right for the probe
    false_hits: int     # served with the answer to a different question
    should_hit: int     # probes for which some cached answer is right

    @property
    def hit_rate(self) -> float:
        return self.correct / self.should_hit if self.should_hit else 0.0

    @property
    def false_hit_rate(self) -> float:
        """Share of *served* answers that answer a different question."""
        return self.false_hits / self.served if self.served else 0.0


def sweep(nearest: Sequence[tuple[Collection[str], str, float]],
          thresholds: Sequence[float]) -> list[SweepPoint]:
    """Score thresholds offline from (acceptable_ids, nearest_id, similarity) per probe.

    `acceptable_ids` are the cached entries whose answer is right for the probe (empty when none
    is). A probe is served if similarity >= threshold; a served probe is correct only if its
    nearest entry is acceptable. Everything else served is a false hit.
    """
    should_hit = sum(1 for ok, _, _ in nearest if ok)
    out = []
    for t in thresholds:
        served = [(ok, got) for ok, got, sim in nearest if sim >= t]
        correct = sum(1 for ok, got in served if got in ok)
        out.append(SweepPoint(t, len(served), correct, len(served) - correct, should_hit))
    return out
