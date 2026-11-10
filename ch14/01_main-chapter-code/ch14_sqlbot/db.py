"""Run guarded SQL as the read-only role."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal

import psycopg
from psycopg.rows import dict_row

from llmprod import connect

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
