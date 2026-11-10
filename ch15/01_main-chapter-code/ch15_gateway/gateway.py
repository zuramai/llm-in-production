"""Talking to the LiteLLM proxy: an OpenAI client for apps, and a few admin calls for keys."""

from __future__ import annotations

import os
from typing import Any

import httpx
import openai

GATEWAY_URL = os.environ.get("CH15_GATEWAY_URL", "http://127.0.0.1:4015")
MASTER_KEY = os.environ.get("LITELLM_MASTER_KEY", "sk-ch15-dev-master")

# Measurement calls must not be answered by the gateway's own cache, or a re-run of the
# notebook "costs" $0 and takes 5 ms. LiteLLM's per-request cache control:
NO_CACHE = {"cache": {"no-cache": True}}


def client(api_key: str = MASTER_KEY, timeout: float = 120, max_retries: int = 0) -> openai.OpenAI:
    """An ordinary OpenAI client. The app only knows the gateway URL, a key and an alias."""
    return openai.OpenAI(base_url=f"{GATEWAY_URL}/v1", api_key=api_key,
                         timeout=timeout, max_retries=max_retries)


def _admin(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
    r = httpx.request(method, f"{GATEWAY_URL}{path}", timeout=30,
                      headers={"Authorization": f"Bearer {MASTER_KEY}"}, **kwargs)
    r.raise_for_status()
    return r.json()


def generate_key(alias: str, models: list[str], max_budget: float | None = None,
                 rpm_limit: int | None = None, tpm_limit: int | None = None,
                 budget_duration: str | None = None) -> str:
    body = {"key_alias": alias, "models": models, "max_budget": max_budget,
            "rpm_limit": rpm_limit, "tpm_limit": tpm_limit, "budget_duration": budget_duration}
    return _admin("POST", "/key/generate",
                  json={k: v for k, v in body.items() if v is not None})["key"]


def key_info(key: str) -> dict[str, Any]:
    return _admin("GET", "/key/info", params={"key": key})["info"]


def delete_keys_by_alias(prefix: str) -> int:
    """Remove keys a previous notebook run created, so re-runs start clean."""
    keys = _admin("GET", "/key/list", params={"return_full_object": "true", "size": 100})["keys"]
    aliases = [k["key_alias"] for k in keys if (k.get("key_alias") or "").startswith(prefix)]
    if aliases:
        _admin("POST", "/key/delete", json={"key_aliases": aliases})
    return len(aliases)
