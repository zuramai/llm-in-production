"""Conversation stores. A conversation is the list of user/assistant turns, without the system
prompt: the prompt version is chosen per request, so it must not be frozen into history.

- MemorySessionStore: Chapter 1's in-process LRU. Correct for one process; with N replicas
  behind a load balancer a follow-up question lands on a pod that has never seen the session.
- PostgresSessionStore: one row per session in `ch20.chat_sessions`, shared by every replica.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Protocol

import psycopg
from psycopg.types.json import Jsonb

Message = dict[str, str]


class SessionStore(Protocol):
    def get(self, session_id: str) -> list[Message]: ...
    def put(self, session_id: str, turns: list[Message]) -> None: ...
    async def aping(self) -> None: ...


class MemorySessionStore:
    def __init__(self, max_sessions: int = 1_000):
        self._data: OrderedDict[str, list[Message]] = OrderedDict()
        self._max = max_sessions
        self._lock = threading.Lock()

    def get(self, session_id: str) -> list[Message]:
        with self._lock:
            if session_id in self._data:
                self._data.move_to_end(session_id)
                return list(self._data[session_id])
            return []

    def put(self, session_id: str, turns: list[Message]) -> None:
        with self._lock:
            self._data[session_id] = list(turns)
            self._data.move_to_end(session_id)
            while len(self._data) > self._max:
                self._data.popitem(last=False)

    async def aping(self) -> None:
        return None


class PostgresSessionStore:
    """One short connection per operation: simple and stateless. A pool is the next step."""

    def __init__(self, dsn: str, connect_timeout: int = 3):
        self._dsn = dsn
        self._timeout = connect_timeout

    def _connect(self) -> psycopg.Connection:
        return psycopg.connect(self._dsn, autocommit=True, connect_timeout=self._timeout)

    def get(self, session_id: str) -> list[Message]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT messages FROM ch20.chat_sessions WHERE session_id = %s", (session_id,)
            ).fetchone()
        return list(row[0]) if row else []

    def put(self, session_id: str, turns: list[Message]) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO ch20.chat_sessions (session_id, messages) VALUES (%s, %s) "
                "ON CONFLICT (session_id) DO UPDATE "
                "SET messages = EXCLUDED.messages, updated_at = now()",
                (session_id, Jsonb(turns)),
            )

    async def aping(self) -> None:
        """Readiness check, async so it never waits behind busy worker threads."""
        async with await psycopg.AsyncConnection.connect(
                self._dsn, autocommit=True, connect_timeout=self._timeout) as conn:
            await conn.execute("SELECT 1 FROM ch20.chat_sessions LIMIT 1")
