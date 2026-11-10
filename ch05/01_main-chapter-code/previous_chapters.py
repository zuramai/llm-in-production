"""Code from earlier chapters that this chapter reuses, copied so ch05 can be read on its own.

- Chapter 4 (ch04/01_main-chapter-code/ch04_rag/): heading-aware chunking, the pgvector +
  full-text store, hybrid retrieval with Reciprocal Rank Fusion and a cross-encoder reranker, and
  the cited-answer prompt. Changes: only the `heading` chunker is kept, the table is
  `ch05_doc_chunks` so this chapter doesn't depend on ch04's table, and the reranker falls back
  to the CPU when the shared GPU is full (`rerank_device`).
- Chapter 6 (ch06/01_main-chapter-code/sqlbot/): the sqlglot AST guard (`validate_sql`, verbatim)
  and `run_query` (read-only role, 5 s statement timeout, verbatim).
- Chapter 1: the OpenRouter price table.

ch05 code loads this file under the unique module name `ch05_previous_chapters`
(see `ch05_agentic/prev.py`), because every chapter ships a `previous_chapters.py` and the
book's tests run in one Python process.
"""

from __future__ import annotations

import re
import threading
import time
from collections import defaultdict
from collections.abc import Hashable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Protocol

import numpy as np
import psycopg
import sqlglot
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from sqlglot import exp
from sqlglot.errors import ParseError

from llmprod import connect, embed_client, get_settings

# --------------------------------------------------------------------------------------------
# Chapter 1: prices. $ per 1M tokens (input, output) on OpenRouter, as of 2026-11-07.
# --------------------------------------------------------------------------------------------
PRICES = {
    "openai/gpt-6-luna": (0.10, 0.50),
    "anthropic/claude-haiku-5.5": (0.10, 0.50),
    "anthropic/claude-sonnet-5.5": (2.00, 10.00),
    "qwen/qwen3.8-flash": (0.15, 0.47),
}

# --------------------------------------------------------------------------------------------
# Chapter 4: chunking (ch04_rag/chunking.py, heading chunker only)
# --------------------------------------------------------------------------------------------
HEADING_MAX = 512
FIXED_OVERLAP = 64
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")


class Tokenizer(Protocol):
    def token_starts(self, text: str) -> list[int]:
        """Character offset where each token starts."""


class TiktokenTokenizer:
    def __init__(self, encoding: str = "o200k_base") -> None:
        import tiktoken

        self._enc = tiktoken.get_encoding(encoding)

    def token_starts(self, text: str) -> list[int]:
        _, offsets = self._enc.decode_with_offsets(self._enc.encode(text))
        return offsets


@dataclass(frozen=True)
class Chunk:
    doc_path: str
    chunk_idx: int
    chunker: str
    heading_path: str
    content: str
    token_count: int

    @property
    def embed_text(self) -> str:
        return f"{self.heading_path}\n{self.content}" if self.heading_path else self.content


def _window_spans(text: str, size: int, overlap: int, tok: Tokenizer) -> list[tuple[int, int, int]]:
    starts = tok.token_starts(text)
    n = len(starts)
    if n == 0:
        return []
    step = size - overlap
    spans = []
    i = 0
    while True:
        j = min(i + size, n)
        end_char = starts[j] if j < n else len(text)
        spans.append((starts[i], end_char, j - i))
        if j >= n:
            break
        i += step
    return spans


def split_sections(text: str) -> list[tuple[str, str]]:
    """Cut Markdown into (heading_path, body) sections. Headings inside code fences are text."""
    sections: list[tuple[str, list[str]]] = []
    stack: list[tuple[int, str]] = []
    path = ""
    body: list[str] = []
    in_fence = False
    for line in text.splitlines():
        if _FENCE.match(line):
            in_fence = not in_fence
        m = None if in_fence else _HEADING.match(line)
        if m:
            sections.append((path, body))
            level, title = len(m.group(1)), m.group(2)
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            path = " > ".join(t for _, t in stack)
            body = []
        else:
            body.append(line)
    sections.append((path, body))
    return [(p, "\n".join(b).strip()) for p, b in sections if "\n".join(b).strip()]


def _pack_paragraphs(body: str, max_tokens: int, tok: Tokenizer) -> list[str]:
    pieces: list[str] = []
    current: list[str] = []
    used = 0
    for para in re.split(r"\n\s*\n", body):
        para = para.strip()
        if not para:
            continue
        n = len(tok.token_starts(para))
        if n > max_tokens:
            if current:
                pieces.append("\n\n".join(current))
                current, used = [], 0
            pieces.extend(
                para[a:b].strip()
                for a, b, _ in _window_spans(para, max_tokens, FIXED_OVERLAP, tok)
            )
            continue
        if used + n > max_tokens and current:
            pieces.append("\n\n".join(current))
            current, used = [], 0
        current.append(para)
        used += n
    if current:
        pieces.append("\n\n".join(current))
    return pieces


def heading_chunks(text: str, doc_path: str, max_tokens: int = HEADING_MAX,
                   tok: Tokenizer | None = None) -> list[Chunk]:
    tok = tok or TiktokenTokenizer()
    out: list[Chunk] = []
    for path, body in split_sections(text):
        for piece in _pack_paragraphs(body, max_tokens, tok):
            out.append(
                Chunk(doc_path, len(out), "heading", path, piece, len(tok.token_starts(piece)))
            )
    return out


# --------------------------------------------------------------------------------------------
# Chapter 4: storage (ch04_rag/store.py, table renamed to ch05_doc_chunks)
# --------------------------------------------------------------------------------------------
DOCS_ROOT = Path(__file__).resolve().parents[2] / "datasets" / "shopflow" / "docs"
EMBED_DIM = 768
DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "
TABLE = "ch05_doc_chunks"

DDL = f"""
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS {TABLE} (
    id           bigserial PRIMARY KEY,
    chunker      text    NOT NULL,
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
CREATE INDEX IF NOT EXISTS {TABLE}_hnsw ON {TABLE}
    USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
CREATE INDEX IF NOT EXISTS {TABLE}_fts ON {TABLE} USING gin (tsv);
"""


def open_db() -> psycopg.Connection:
    conn = connect()
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    conn.execute("SET hnsw.iterative_scan = strict_order")
    return conn


def setup(conn: psycopg.Connection, reset: bool = False) -> None:
    if reset:
        conn.execute(f"DROP TABLE IF EXISTS {TABLE}")
    conn.execute(DDL)


def load_docs(root: Path = DOCS_ROOT) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): p.read_text() for p in sorted(root.rglob("*.md"))}


def embed_texts(texts: Sequence[str], prefix: str = DOC_PREFIX, batch: int = 32) -> np.ndarray:
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


def ingest(conn: psycopg.Connection, docs: dict[str, str], chunker: str = "heading") -> dict:
    t0 = time.perf_counter()
    chunks = [c for path, text in docs.items() for c in heading_chunks(text, path)]
    vecs = embed_texts([c.embed_text for c in chunks])
    with conn.transaction():
        conn.execute(f"DELETE FROM {TABLE} WHERE chunker = %s", (chunker,))
        with conn.cursor() as cur:
            cur.executemany(
                f"INSERT INTO {TABLE} (chunker, doc_path, chunk_idx, heading_path, content,"
                " token_count, embedding) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                [(c.chunker, c.doc_path, c.chunk_idx, c.heading_path, c.content,
                  c.token_count, v) for c, v in zip(chunks, vecs, strict=True)],
            )
    return {"chunker": chunker, "docs": len(docs), "chunks": len(chunks),
            "seconds": round(time.perf_counter() - t0, 1)}


# --------------------------------------------------------------------------------------------
# Chapter 4: retrieval (ch04_rag/retrieve.py)
# --------------------------------------------------------------------------------------------
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
    source: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def citation(self) -> str:
        return f"[doc:{self.doc_path}#{self.chunk_idx}]"

    @property
    def embed_text(self) -> str:
        return f"{self.heading_path}\n{self.content}" if self.heading_path else self.content


def rrf(rankings: Sequence[Sequence[Hashable]], k: int = RRF_K) -> list[tuple[Hashable, float]]:
    """Reciprocal Rank Fusion: score(d) = sum over rankings of 1 / (k + rank(d))."""
    scores: dict[Hashable, float] = defaultdict(float)
    first_seen: dict[Hashable, int] = {}
    for ranking in rankings:
        seen: set[Hashable] = set()
        for rank, item in enumerate(ranking, start=1):
            if item in seen:
                continue
            seen.add(item)
            scores[item] += 1.0 / (k + rank)
            first_seen.setdefault(item, len(first_seen))
    return sorted(scores.items(), key=lambda kv: (-kv[1], first_seen[kv[0]]))


_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.\-/$<>]*")
_COLS = "id, doc_path, chunk_idx, heading_path, content"


def or_terms(question: str) -> list[str]:
    return [w.rstrip(".-") for w in _WORD.findall(question)]


def _hits(rows, source: str) -> list[Hit]:
    return [Hit(r[0], r[1], r[2], r[3], r[4], score=float(r[5]), rank=i, source=source)
            for i, r in enumerate(rows, start=1)]


def vector_search(conn: psycopg.Connection, query_vec, chunker: str, k: int) -> list[Hit]:
    rows = conn.execute(
        f"SELECT {_COLS}, 1 - (embedding <=> %s) AS cos FROM {TABLE}"
        " WHERE chunker = %s ORDER BY embedding <=> %s LIMIT %s",
        (query_vec, chunker, query_vec, k),
    ).fetchall()
    return _hits(rows, "vector")


def fts_search(conn: psycopg.Connection, question: str, chunker: str, k: int) -> list[Hit]:
    terms = or_terms(question)
    if not terms:
        return []
    query = " || ".join(["plainto_tsquery('english', %s)"] * len(terms))
    rows = conn.execute(
        f"WITH q AS (SELECT {query} AS tsq) SELECT {_COLS}, ts_rank_cd(tsv, q.tsq) AS score"
        f" FROM {TABLE}, q WHERE chunker = %s AND tsv @@ q.tsq"
        " ORDER BY score DESC, id LIMIT %s",
        (*terms, chunker, k),
    ).fetchall()
    return _hits(rows, "fts")


def rerank_device(min_free_bytes: int = 1 << 30) -> str:
    """ch05 change: ch04 picked "cuda" whenever a GPU existed. On a shared GPU that another job
    has filled, the 90 MB reranker then dies with CUDA out-of-memory mid-eval (it happened while
    writing this chapter). Use the GPU only if 1 GiB is free; the CPU scores 30 pairs fine."""
    import torch

    if not torch.cuda.is_available():
        return "cpu"
    free, _ = torch.cuda.mem_get_info()
    return "cuda" if free >= min_free_bytes else "cpu"


class Retriever:
    """Hybrid (vector + full-text, fused with RRF) candidates, re-scored by a cross-encoder."""

    _ce_lock = threading.Lock()
    _ce = None

    def __init__(self, conn: psycopg.Connection, pool: int = 30):
        self.conn = conn
        self.pool = pool
        self.last_timing: dict[str, float] = {}

    @classmethod
    def cross_encoder(cls):
        with cls._ce_lock:
            if cls._ce is None:
                from sentence_transformers import CrossEncoder

                cls._ce = CrossEncoder(RERANK_MODEL, max_length=512, device=rerank_device())
            return cls._ce

    def retrieve(self, question: str, chunker: str = "heading", k: int = 5) -> list[Hit]:
        """ch04's `hybrid_rerank` mode."""
        t0 = time.perf_counter()
        qv = embed_texts([question], prefix=QUERY_PREFIX)[0]
        vec = vector_search(self.conn, qv, chunker, self.pool)
        fts = fts_search(self.conn, question, chunker, self.pool)
        by_id = {h.chunk_id: h for h in [*vec, *fts]}
        fused = rrf([[h.chunk_id for h in vec], [h.chunk_id for h in fts]])
        candidates = [by_id[cid] for cid, _ in fused][: self.pool]
        t1 = time.perf_counter()
        scores = self.cross_encoder().predict(
            [(question, h.embed_text) for h in candidates], show_progress_bar=False
        )
        order = sorted(zip(candidates, scores, strict=True), key=lambda p: -float(p[1]))
        self.last_timing = {"search_ms": (t1 - t0) * 1000,
                            "rerank_ms": (time.perf_counter() - t1) * 1000}
        return [Hit(h.chunk_id, h.doc_path, h.chunk_idx, h.heading_path, h.content,
                    score=float(s), rank=i, source="hybrid_rerank")
                for i, (h, s) in enumerate(order[:k], start=1)]


# --------------------------------------------------------------------------------------------
# Chapter 4: the cited-answer prompt (ch04_rag/answer.py)
# --------------------------------------------------------------------------------------------
REFUSAL = "I can't find that in the ShopFlow documents."

RAG_K = 10  # ch04's decision: heading chunks, hybrid + rerank, k = 10
# ch04's final prompt ("v2" in ch04_rag/answer.py)
RAG_SYSTEM = f"""You are ShopFlow's internal documentation assistant.
Answer ONLY from the numbered context chunks below. Rules:
1. After every factual sentence, cite the chunk(s) it came from, exactly like [doc:path#N].
   Copy the tag from the chunk header. Never invent a tag.
2. If the context does not contain the answer, reply exactly: {REFUSAL}
   Do not guess, and do not use outside knowledge, even if you are sure.
3. If chunks disagree, say so and cite both.
4. Answer every part of the question. For each part, include the conditions, exceptions and
   limits the context attaches to it (who, how much, how long, starting when, only if), and
   quote commands, flags and names exactly as written.
5. If the context answers only part of the question, answer that part and say what is missing.
6. No greetings and no advice the context doesn't give. Usually 2-6 sentences."""

_CITE = re.compile(r"\[doc:([^\]#\s]+)#(\d+)\]")


def format_context(hits: list[Hit]) -> str:
    parts = []
    for h in hits:
        head = f" ({h.heading_path})" if h.heading_path else ""
        parts.append(f"{h.citation}{head}\n{h.content}")
    return "\n\n---\n\n".join(parts)


def extract_citations(text: str) -> list[tuple[str, int]]:
    return [(m.group(1), int(m.group(2))) for m in _CITE.finditer(text)]


# --------------------------------------------------------------------------------------------
# Chapter 6: the SQL guard (sqlbot/guard.py, verbatim)
# --------------------------------------------------------------------------------------------
BLOCKED_FUNCTIONS = frozenset({
    "pg_sleep", "pg_read_file", "pg_read_binary_file", "pg_ls_dir", "pg_stat_file",
    "lo_import", "lo_export", "dblink", "dblink_exec", "copy", "pg_terminate_backend",
    "pg_cancel_backend", "set_config", "pg_reload_conf",
})
WRITE_NODES = (
    exp.Insert, exp.Update, exp.Delete, exp.Merge, exp.Drop, exp.Create, exp.Alter,
    exp.TruncateTable, exp.Command, exp.Copy, exp.Grant,
)


class UnsafeSQLError(ValueError):
    """The SQL breaks a guard rule. The message is safe to show to the model."""


def _function_name(node: exp.Func) -> str:
    if isinstance(node, exp.Anonymous):
        return node.name.lower()
    return node.sql_name().lower()


def validate_sql(sql: str, allowed_tables: set[str], max_limit: int = 1000) -> str:
    """Return normalized, LIMIT-capped Postgres SQL, or raise UnsafeSQLError."""
    try:
        statements = [s for s in sqlglot.parse(sql, dialect="postgres") if s is not None]
    except ParseError as e:
        raise UnsafeSQLError(f"rule=parse: SQL does not parse as Postgres: {e}") from None
    if len(statements) != 1:
        raise UnsafeSQLError(
            f"rule=single_statement: expected exactly 1 statement, got {len(statements)}")
    tree = statements[0]

    if not isinstance(tree, (exp.Select, exp.SetOperation)):
        raise UnsafeSQLError(
            f"rule=select_only: only SELECT queries are allowed, got {tree.key.upper()}")
    for node in tree.walk():
        if isinstance(node, WRITE_NODES):
            raise UnsafeSQLError(
                f"rule=select_only: {node.key.upper()} is not allowed anywhere in the query")
        if isinstance(node, exp.Select) and node.args.get("into"):
            raise UnsafeSQLError("rule=no_select_into: SELECT ... INTO creates a table")
        if isinstance(node, exp.Func) and _function_name(node) in BLOCKED_FUNCTIONS:
            raise UnsafeSQLError(
                f"rule=blocked_function: function {_function_name(node)}() is not allowed")

    cte_names = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
    allowed = {t.lower() for t in allowed_tables} | cte_names
    for table in tree.find_all(exp.Table):
        name = table.name.lower()
        schema = table.db.lower()
        if not name or schema not in ("", "public") or name not in allowed:
            shown = f"{schema}.{name}" if schema else (name or table.sql())
            raise UnsafeSQLError(
                f"rule=allowed_tables: table {shown} is not allowed; "
                f"use only: {', '.join(sorted(allowed_tables))}")

    limit = tree.args.get("limit")
    if limit is None:
        tree = tree.limit(max_limit)
    else:
        value = limit.expression
        if not (isinstance(value, exp.Literal) and value.is_int):
            raise UnsafeSQLError("rule=limit: LIMIT must be an integer literal")
        if int(value.this) > max_limit:
            tree = tree.limit(max_limit)
    return tree.sql(dialect="postgres")


# --------------------------------------------------------------------------------------------
# Chapter 6: run guarded SQL as the read-only role (sqlbot/db.py, verbatim)
# --------------------------------------------------------------------------------------------
MAX_ROWS = 1000


class QueryError(Exception):
    """Postgres rejected or failed the query. The message is fed back to the model."""


def _jsonable(value):
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def run_query(sql: str) -> list[dict]:
    try:
        with connect(readonly=True) as conn:
            conn.execute("SET statement_timeout = '5s'")
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(sql)
                rows = cur.fetchmany(MAX_ROWS)
    except psycopg.Error as e:
        raise QueryError(str(e).strip()) from None
    return [{k: _jsonable(v) for k, v in row.items()} for row in rows]
