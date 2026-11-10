"""Run guarded SQL as the read-only role (copied from ch06/01_main-chapter-code/sqlbot/db.py).

Change for production: the connection comes from the service's own settings, not the book's
`llmprod.connect()`.
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal

import psycopg
from psycopg import sql as pgsql
from psycopg.rows import dict_row

MAX_ROWS = 1000


class QueryError(Exception):
    """Postgres rejected or failed the query. The message is fed back to the model."""


def _jsonable(value):
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def connect(dsn: str, timezone: str) -> psycopg.Connection:
    conn = psycopg.connect(dsn, autocommit=True, connect_timeout=3)
    conn.execute(pgsql.SQL("SET TIME ZONE {}").format(pgsql.Literal(timezone)))
    return conn


def run_query(sql: str, dsn: str, timezone: str) -> list[dict]:
    try:
        with connect(dsn, timezone) as conn:
            conn.execute("SET statement_timeout = '5s'")
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(sql)
                rows = cur.fetchmany(MAX_ROWS)
    except psycopg.Error as e:
        raise QueryError(str(e).strip()) from None
    return [{k: _jsonable(v) for k, v in row.items()} for row in rows]
