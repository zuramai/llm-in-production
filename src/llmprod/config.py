"""Single source of truth for environment configuration used by every chapter."""

from __future__ import annotations

import os
from functools import cache

from dotenv import find_dotenv, load_dotenv
from pydantic import BaseModel

_REQUIRED = (
    "llm_base_url",
    "llm_api_key",
    "llm_model",
    "llm_model_strong",
    "embed_base_url",
    "embed_api_key",
    "embed_model",
    "database_url",
    "database_url_ro",
)


class Settings(BaseModel):
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    llm_model_strong: str
    embed_base_url: str
    embed_api_key: str
    embed_model: str
    database_url: str
    database_url_ro: str
    business_timezone: str = "America/New_York"


@cache
def get_settings() -> Settings:
    """Load the repo-root `.env` (searched upward from the CWD) and validate it."""
    load_dotenv(find_dotenv(usecwd=True))
    values: dict[str, str] = {}
    for field in Settings.model_fields:
        value = os.environ.get(field.upper(), "").strip()
        if value:
            values[field] = value
        elif field in _REQUIRED:
            raise RuntimeError(
                f"Missing {field.upper()}: copy .env.example to .env and fill it in "
                "(see setup/README.md)"
            )
    return Settings(**values)
