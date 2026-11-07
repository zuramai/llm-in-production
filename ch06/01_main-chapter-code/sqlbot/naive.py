"""v0: the naive text-to-SQL bot. DO NOT DEPLOY.

Raw DDL in the prompt, free-text SQL out, executed as the admin user with no guard. The chapter
uses it to show what goes wrong. Every query runs inside a transaction that is rolled back, so a
destructive query can "succeed" without damaging the shared seed.
"""

from __future__ import annotations

import re
import time
from functools import cache

import openai
import psycopg
from psycopg.rows import dict_row

from llmprod import chat_client, get_settings

from .db import _jsonable


@cache
def raw_ddl() -> str:
    """Columns of every table, straight from information_schema."""
    with psycopg.connect(get_settings().database_url) as conn:
        rows = conn.execute(
            "SELECT table_name, column_name, data_type FROM information_schema.columns "
            "WHERE table_schema = 'public' ORDER BY table_name, ordinal_position"
        ).fetchall()
    tables: dict[str, list[str]] = {}
    for table, col, typ in rows:
        tables.setdefault(table, []).append(f"  {col} {typ}")
    return "\n".join(f"CREATE TABLE {t} (\n" + ",\n".join(cols) + "\n);"
                     for t, cols in tables.items())


def generate_sql(question: str, model: str | None = None, client: openai.OpenAI | None = None,
                 reasoning_effort: str | None = None) -> tuple[str, dict]:
    client = client or chat_client()
    options = {"reasoning_effort": reasoning_effort} if reasoning_effort else {}
    resp = client.chat.completions.create(
        model=model or get_settings().llm_model,
        max_tokens=4000,
        **options,
        messages=[
            {"role": "system", "content": "You write PostgreSQL queries for this database:\n\n"
                                          + raw_ddl()
                                          + "\n\nReply with only the SQL, no explanation."},
            {"role": "user", "content": question},
        ],
    )
    text = resp.choices[0].message.content or ""
    match = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL)
    sql = (match.group(1) if match else text).strip()
    return sql, resp.usage.model_dump(exclude_none=True) if resp.usage else {}


def execute_rolled_back(sql: str) -> tuple[list[dict], int]:
    """Run as admin (no guard!) and roll back. Returns rows and the affected row count."""
    with psycopg.connect(get_settings().database_url) as conn:  # not autocommit: a transaction
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql)
            affected = cur.rowcount
            rows = cur.fetchmany(1000) if cur.description else []
        conn.rollback()
    return [{k: _jsonable(v) for k, v in r.items()} for r in rows], affected


def answer(question: str, model: str | None = None, client: openai.OpenAI | None = None,
           reasoning_effort: str | None = None) -> dict:
    started = time.perf_counter()
    sql, usage = generate_sql(question, model, client, reasoning_effort)
    try:
        rows, affected = execute_rolled_back(sql)
        error = None
    except psycopg.Error as e:
        rows, affected, error = [], 0, str(e).strip()
    return {
        "question": question,
        "sql": sql,
        "rows": rows,
        "affected": affected,
        "error": error,
        "usage": usage,
        "latency_ms": round((time.perf_counter() - started) * 1000),
    }
