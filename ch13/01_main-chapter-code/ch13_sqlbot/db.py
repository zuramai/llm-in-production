"""Run guarded SQL as a read-only role.

Copied from ch06/01_main-chapter-code/sqlbot/db.py. Chapter 13 change: `conninfo` lets the
defended bot connect as the narrower `ch13_analytics_ro` role (privilege separation).
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal

import psycopg
from psycopg import sql as pgsql
from psycopg.rows import dict_row

from llmprod import connect, get_settings

MAX_ROWS = 1000


class QueryError(Exception):
    """Postgres rejected or failed the query. The message is fed back to the model."""


def _jsonable(value):
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def _connect(conninfo: str | None) -> psycopg.Connection:
    if conninfo is None:
        return connect(readonly=True)
    conn = psycopg.connect(conninfo, autocommit=True)
    tz = get_settings().business_timezone
    conn.execute(pgsql.SQL("SET TIME ZONE {}").format(pgsql.Literal(tz)))
    return conn


def run_query(sql: str, conninfo: str | None = None) -> list[dict]:
    try:
        with _connect(conninfo) as conn:
            conn.execute("SET statement_timeout = '5s'")
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(sql)
                rows = cur.fetchmany(MAX_ROWS)
    except psycopg.Error as e:
        raise QueryError(str(e).strip()) from None
    return [{k: _jsonable(v) for k, v in row.items()} for row in rows]
