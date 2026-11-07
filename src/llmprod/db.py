"""Postgres connections to the ShopFlow database."""

from __future__ import annotations

import psycopg
from psycopg import sql

from .config import get_settings


def connect(readonly: bool = False) -> psycopg.Connection:
    """Open an autocommit connection whose session time zone is the business time zone."""
    s = get_settings()
    conn = psycopg.connect(s.database_url_ro if readonly else s.database_url, autocommit=True)
    conn.execute(sql.SQL("SET TIME ZONE {}").format(sql.Literal(s.business_timezone)))
    return conn
