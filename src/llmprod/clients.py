"""OpenAI-compatible clients pointed at whatever provider `.env` configures."""

from __future__ import annotations

import openai

from .config import get_settings


def chat_client(timeout: float = 60, max_retries: int = 2) -> openai.OpenAI:
    s = get_settings()
    return openai.OpenAI(
        base_url=s.llm_base_url,
        api_key=s.llm_api_key,
        timeout=timeout,
        max_retries=max_retries,
    )


def embed_client() -> openai.OpenAI:
    s = get_settings()
    return openai.OpenAI(base_url=s.embed_base_url, api_key=s.embed_api_key)
