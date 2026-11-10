"""Service configuration, read from environment variables only (12-factor).

Each service declares exactly the settings it needs. Locally, pass them with
`uvicorn --env-file`; in Compose and Kubernetes they come from a ConfigMap and a Secret.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

SESSION_STORES = ("memory", "postgres")


@dataclass(frozen=True)
class Settings:
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    prompt_stable: str = "v1"
    prompt_canary: str | None = None
    canary_percent: float = 0.0
    session_store: str = "memory"
    sessions_database_url: str | None = None
    reasoning_effort: str | None = None
    llm_timeout_s: float = 30.0
    max_context_tokens: int = 8_000
    max_output_tokens: int = 800


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    """Build Settings from `env` (default: os.environ). Raise ValueError on any bad value.

    Failing here, at startup, turns a bad ConfigMap into a pod that never becomes ready instead
    of a pod that serves the wrong prompt.
    """
    env = os.environ if env is None else env

    def get(name: str) -> str | None:
        value = env.get(name, "").strip()
        return value or None

    missing = [n for n in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL") if not get(n)]
    if missing:
        raise ValueError(f"missing required environment variables: {', '.join(missing)}")

    percent = float(get("CANARY_PERCENT") or 0)
    if not 0 <= percent <= 100:
        raise ValueError(f"CANARY_PERCENT must be between 0 and 100, got {percent}")
    canary = get("PROMPT_CANARY")
    if percent > 0 and not canary:
        raise ValueError("CANARY_PERCENT > 0 needs PROMPT_CANARY")

    store = get("SESSION_STORE") or "memory"
    if store not in SESSION_STORES:
        raise ValueError(f"SESSION_STORE must be one of {SESSION_STORES}, got {store!r}")
    sessions_url = get("SESSIONS_DATABASE_URL")
    if store == "postgres" and not sessions_url:
        raise ValueError("SESSION_STORE=postgres needs SESSIONS_DATABASE_URL")

    return Settings(
        llm_base_url=get("LLM_BASE_URL"),
        llm_api_key=get("LLM_API_KEY"),
        llm_model=get("LLM_MODEL"),
        prompt_stable=get("PROMPT_STABLE") or "v1",
        prompt_canary=canary,
        canary_percent=percent,
        session_store=store,
        sessions_database_url=sessions_url,
        reasoning_effort=get("REASONING_EFFORT"),
        llm_timeout_s=float(get("LLM_TIMEOUT_S") or 30),
        max_context_tokens=int(get("MAX_CONTEXT_TOKENS") or 8_000),
        max_output_tokens=int(get("MAX_OUTPUT_TOKENS") or 800),
    )
