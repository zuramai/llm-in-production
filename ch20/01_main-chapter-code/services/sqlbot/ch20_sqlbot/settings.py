"""Service configuration, read from environment variables only.

The data bot needs the read-only database URL and an LLM key. It never sees the admin URL,
which the book's shared `llmprod.get_settings()` would require.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    database_url_ro: str
    business_timezone: str = "America/New_York"
    reasoning_effort: str | None = None
    llm_timeout_s: float = 60.0


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    env = os.environ if env is None else env

    def get(name: str) -> str | None:
        value = env.get(name, "").strip()
        return value or None

    required = ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL", "DATABASE_URL_RO")
    missing = [n for n in required if not get(n)]
    if missing:
        raise ValueError(f"missing required environment variables: {', '.join(missing)}")
    return Settings(
        llm_base_url=get("LLM_BASE_URL"),
        llm_api_key=get("LLM_API_KEY"),
        llm_model=get("LLM_MODEL"),
        database_url_ro=get("DATABASE_URL_RO"),
        business_timezone=get("BUSINESS_TIMEZONE") or "America/New_York",
        reasoning_effort=get("REASONING_EFFORT"),
        llm_timeout_s=float(get("LLM_TIMEOUT_S") or 60),
    )
