"""Observability helpers for the instrumented ShopFlow data bot (Chapter 14).

Everything that turns an LLM call into telemetry lives here, so it can be unit-tested without a
Langfuse server: client setup, PII masking, OpenRouter usage -> Langfuse usage/cost, the
OpenTelemetry GenAI attributes we add to every generation, and user feedback scores.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from typing import Any

# The local stack from langfuse/docker-compose.yml, initialized headlessly by langfuse.env.
# Real deployments set LANGFUSE_BASE_URL / LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY instead.
LOCAL_BASE_URL = "http://localhost:3014"
LOCAL_PUBLIC_KEY = "pk-lf-ch14-local"
LOCAL_SECRET_KEY = "sk-lf-ch14-local"
LOCAL_PROJECT_ID = "sqlbot"

APP_VERSION = "sqlbot-ch14-1.0"
FEEDBACK_SCORE = "user_feedback"

_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def langfuse_settings() -> dict[str, str]:
    return {
        "base_url": os.getenv("LANGFUSE_BASE_URL", LOCAL_BASE_URL),
        "public_key": os.getenv("LANGFUSE_PUBLIC_KEY", LOCAL_PUBLIC_KEY),
        "secret_key": os.getenv("LANGFUSE_SECRET_KEY", LOCAL_SECRET_KEY),
    }


def mask(*, data: Any, **_: Any) -> Any:
    """Langfuse `mask=` hook: redact e-mail addresses anywhere in trace inputs and outputs.

    The bot can't read customers.email (column grant), but users paste addresses into questions
    and the phrase step echoes questions back. Traces are a second copy of your data.
    """
    if isinstance(data, str):
        return _EMAIL.sub("<email>", data)
    if isinstance(data, Mapping):
        return {k: mask(data=v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [mask(data=v) for v in data]
    return data


_client = None


def client():
    """The process-wide Langfuse client (created once; Langfuse batches and flushes in the
    background, so never create one per request)."""
    global _client
    if _client is None:
        from langfuse import Langfuse

        # OTel resource attribute on every span; the default is "unknown_service:python".
        os.environ.setdefault("OTEL_SERVICE_NAME", "shopflow-sqlbot")
        _client = Langfuse(**langfuse_settings(), mask=mask, release=APP_VERSION)
    return _client


def _get(obj: Any, key: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, Mapping):
        return obj.get(key)
    value = getattr(obj, key, None)
    if value is None:
        value = (getattr(obj, "model_extra", None) or {}).get(key)
    return value


def usage_details(usage: Any) -> dict[str, int]:
    """OpenAI-style `usage` -> Langfuse usage_details (flat ints; zero breakdowns dropped).

    Langfuse treats usage keys as *additive* types: "input tokens" is the sum of every key that
    starts with "input". OpenAI's prompt_tokens already includes the cached ones, so we split
    them out (input = uncached part) instead of reporting them twice. Same for reasoning.
    """
    if usage is None:
        return {}
    prompt = int(_get(usage, "prompt_tokens") or 0)
    completion = int(_get(usage, "completion_tokens") or 0)
    cached = int(_get(_get(usage, "prompt_tokens_details"), "cached_tokens") or 0)
    reasoning = int(_get(_get(usage, "completion_tokens_details"), "reasoning_tokens") or 0)
    out = {"input": prompt - cached, "output": completion - reasoning,
           "total": int(_get(usage, "total_tokens") or prompt + completion)}
    if cached:
        out["input_cached_tokens"] = cached
    if reasoning:
        out["output_reasoning_tokens"] = reasoning
    return out


def cost_details(usage: Any, provider: str) -> dict[str, float] | None:
    """The provider-reported cost, when there is one.

    OpenRouter returns `usage.cost` in USD. Ollama costs $0 per token (the GPU is already
    paid for), which we record explicitly so dashboards don't show "unknown". For anything
    else we return None and let Langfuse infer cost from its model price table.
    """
    if provider == "ollama":
        return {"total": 0.0}
    cost = _get(usage, "cost")
    if cost is None:
        return None
    return {"total": float(cost)}


def provider_of(base_url: str | None) -> str:
    """Value for gen_ai.provider.name, from the client's base URL."""
    url = (base_url or "").lower()
    if "openrouter.ai" in url:
        return "openrouter"
    if ":11434" in url or "ollama" in url:
        return "ollama"
    if "api.openai.com" in url:
        return "openai"
    return "openai_compatible"


def genai_attributes(*, operation: str, provider: str, request_model: str,
                     max_tokens: int | None = None, response: Any = None) -> dict[str, Any]:
    """OpenTelemetry GenAI semantic-convention attributes for one inference span.

    Required: gen_ai.operation.name, gen_ai.provider.name. Conditionally required:
    gen_ai.request.model. Recommended: max_tokens, response id/model, finish reasons, usage.
    Message contents (gen_ai.input.messages) are opt-in and deliberately left out: Langfuse
    already stores input/output, and other backends shouldn't get a second copy by default.
    """
    attrs: dict[str, Any] = {
        "gen_ai.operation.name": operation,
        "gen_ai.provider.name": provider,
        "gen_ai.request.model": request_model,
    }
    if max_tokens is not None:
        attrs["gen_ai.request.max_tokens"] = max_tokens
    if response is None:
        return attrs
    if _get(response, "id"):
        attrs["gen_ai.response.id"] = _get(response, "id")
    if _get(response, "model"):
        attrs["gen_ai.response.model"] = _get(response, "model")
    reasons = [_get(c, "finish_reason") for c in (_get(response, "choices") or [])]
    if any(reasons):
        attrs["gen_ai.response.finish_reasons"] = [r for r in reasons if r]
    usage = _get(response, "usage")
    if usage is not None:
        # Unlike Langfuse's usage types, the GenAI conventions count inclusively: input_tokens
        # includes cached tokens, output_tokens includes reasoning tokens.
        details = usage_details(usage)
        cached = details.get("input_cached_tokens", 0)
        reasoning = details.get("output_reasoning_tokens", 0)
        attrs["gen_ai.usage.input_tokens"] = details["input"] + cached
        attrs["gen_ai.usage.output_tokens"] = details["output"] + reasoning
        if cached:
            attrs["gen_ai.usage.cache_read.input_tokens"] = cached
        if reasoning:
            attrs["gen_ai.usage.reasoning.output_tokens"] = reasoning
    return attrs


def parse_feedback(text: str) -> bool | None:
    """CLI answer -> thumbs up (True), thumbs down (False) or skipped (None)."""
    t = text.strip().lower()
    if t in {"y", "yes", "+", "+1", "up", "👍"}:
        return True
    if t in {"n", "no", "-", "-1", "down", "👎"}:
        return False
    return None


def record_feedback(trace_id: str, thumbs_up: bool, comment: str | None = None,
                    langfuse=None) -> None:
    """Attach a 👍/👎 to a trace as a BOOLEAN score (1 = helpful)."""
    lf = langfuse or client()
    lf.create_score(name=FEEDBACK_SCORE, value=1 if thumbs_up else 0, data_type="BOOLEAN",
                    trace_id=trace_id, comment=mask(data=comment) if comment else None)


def trace_url(trace_id: str, base_url: str | None = None,
              project_id: str = LOCAL_PROJECT_ID) -> str:
    base = (base_url or langfuse_settings()["base_url"]).rstrip("/")
    return f"{base}/project/{project_id}/traces/{trace_id}"
