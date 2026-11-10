"""Run compiled SQL as the read-only role (same layer as Chapter 6: grants are the boundary)."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal

import psycopg
from psycopg.rows import dict_row

from llmprod import connect

MAX_ROWS = 1000


class DatabaseError(Exception):
    """Postgres rejected or failed the query."""


def jsonable(value):
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
        raise DatabaseError(str(e).strip()) from None
    return [{k: jsonable(v) for k, v in row.items()} for row in rows]
