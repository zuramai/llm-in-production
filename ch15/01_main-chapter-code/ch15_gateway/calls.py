"""One chat call with everything we need for a cost table: text, cost, tokens, cache info, latency.

Works against the LiteLLM gateway (cost in the `x-litellm-response-cost` header) and directly
against OpenRouter (cost in `usage.cost`).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import openai


@dataclass(frozen=True)
class Call:
    model: str
    text: str
    cost_usd: float
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int = 0
    cache_write_tokens: int = 0
    gateway_cache_hit: bool = False
    served_by: str = ""          # the deployment that actually answered (differs after a fallback)
    gateway_overhead_ms: float | None = None


def plain_text_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten content blocks (e.g. with `cache_control`) to plain strings for servers that only
    accept string content, such as some OpenAI-compatible local servers."""
    out = []
    for m in messages:
        content = m["content"]
        if isinstance(content, list):
            content = "".join(block.get("text", "") for block in content)
        out.append({**m, "content": content})
    return out


def _usage_field(obj: Any, name: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    value = getattr(obj, name, None)
    if value is None and getattr(obj, "model_extra", None):
        value = obj.model_extra.get(name)
    return value


def parse_cost(headers: dict[str, str] | Any, usage: Any) -> float:
    """What this call cost us.

    A gateway cache hit costs nothing: any `usage` in the cached body describes the original call.
    Otherwise prefer the gateway's cost header, then OpenRouter's `usage.cost`.
    """
    headers = headers if headers is not None else {}
    if headers.get("x-litellm-cache-key") is not None:
        return 0.0
    header = headers.get("x-litellm-response-cost")
    if header not in (None, "", "None"):
        return float(header)
    cost = _usage_field(usage, "cost")
    return float(cost) if cost is not None else 0.0


def chat(client: openai.OpenAI, model: str, messages: list[dict[str, Any]],
         max_tokens: int = 400, **kwargs: Any) -> Call:
    t0 = time.perf_counter()
    raw = client.chat.completions.with_raw_response.create(
        model=model, messages=messages, max_tokens=max_tokens, **kwargs)
    latency = time.perf_counter() - t0
    resp = raw.parse()
    usage = resp.usage
    details = _usage_field(usage, "prompt_tokens_details")
    return Call(
        model=model,
        text=resp.choices[0].message.content or "",
        cost_usd=parse_cost(raw.headers, usage),
        latency_s=latency,
        prompt_tokens=usage.prompt_tokens if usage else 0,
        completion_tokens=usage.completion_tokens if usage else 0,
        cached_tokens=int(_usage_field(details, "cached_tokens") or 0),
        cache_write_tokens=int(_usage_field(details, "cache_write_tokens") or 0),
        gateway_cache_hit=raw.headers.get("x-litellm-cache-key") is not None,
        # The gateway reports the alias as `model`; the real deployment is in a header.
        served_by=raw.headers.get("x-litellm-model-name") or resp.model or "",
        gateway_overhead_ms=(float(raw.headers["x-litellm-overhead-duration-ms"])
                             if raw.headers.get("x-litellm-overhead-duration-ms") else None),
    )
