"""Percentage router for prompt versions: deterministic, sticky per session, monotonic.

A session lands in one of 10,000 buckets by hashing the canary version name with the session
id. Sessions whose bucket falls under `percent` get the canary prompt. Two properties follow:

- Sticky: the same session always gets the same version while the percentage is unchanged.
- Monotonic: raising the percentage only moves sessions from stable to canary, never back,
  and setting it to 0 moves everyone back to stable (rollback).

Salting with the canary name means each new canary picks a different slice of users.
"""

from __future__ import annotations

import hashlib

BUCKETS = 10_000


def bucket(session_id: str, salt: str) -> int:
    digest = hashlib.sha256(f"{salt}:{session_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % BUCKETS


def choose_version(session_id: str, stable: str, canary: str | None, percent: float) -> str:
    """Return the prompt version for this session: `canary` for `percent`% of sessions."""
    if not canary or percent <= 0:
        return stable
    if percent >= 100:
        return canary
    return canary if bucket(session_id, canary) < percent * BUCKETS / 100 else stable
