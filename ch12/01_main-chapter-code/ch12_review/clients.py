"""Model clients for the chapter: hosted (OpenRouter by default) or local Ollama.

Inside this book's repo the hosted client comes from ``llmprod``. Copied into another repo (for
example into a GitHub Action), it falls back to the plain ``LLM_BASE_URL`` / ``LLM_API_KEY`` /
``LLM_MODEL`` environment variables, so the bot has no dependency on the rest of the book.
"""

from __future__ import annotations

import os

import openai

OLLAMA_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")


def make_client(provider: str = "hosted", timeout: float = 120) -> openai.OpenAI:
    if provider == "ollama":
        return openai.OpenAI(base_url=OLLAMA_URL, api_key="ollama", timeout=timeout,
                             max_retries=1)
    try:
        from llmprod import chat_client
    except ImportError:
        return openai.OpenAI(base_url=os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1"),
                             api_key=os.environ["LLM_API_KEY"], timeout=timeout, max_retries=2)
    return chat_client(timeout=timeout)


def default_model() -> str:
    try:
        from llmprod import get_settings
    except ImportError:
        return os.environ["LLM_MODEL"]
    return get_settings().llm_model


def model_options(model: str, provider: str) -> dict:
    """Per-model request options. Local reasoning models must not 'think' for minutes.

    qwen3 on Ollama honours reasoning_effort="none". gpt-oss can't switch reasoning off; "low"
    is its smallest budget, and its thinking tokens count against MAX_TOKENS.
    """
    if provider == "ollama" and model.startswith("qwen3"):
        return {"reasoning_effort": "none"}
    if provider == "ollama" and model.startswith("gpt-oss"):
        return {"reasoning_effort": "low"}
    return {}
