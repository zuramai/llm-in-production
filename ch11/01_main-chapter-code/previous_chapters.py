"""Code from earlier chapters that this chapter reuses, copied so ch11 can be read on its own.

From Chapter 1 (ch01/01_main-chapter-code/ch01.ipynb, "Cost napkin math") and Chapter 6
(sqlbot/agent.py, tolerant JSON parsing and the usage/cost bookkeeping).
"""

from __future__ import annotations

# $ per 1M tokens (input, output) on OpenRouter, as of 2026-11-07.
PRICES = {
    "openai/gpt-6-luna": (0.10, 0.50),
    "anthropic/claude-haiku-5.5": (0.10, 0.50),
    "anthropic/claude-sonnet-5.5": (2.00, 10.00),
}


def estimate_cost(prompt_tokens: int, completion_tokens: int,
                  usd_per_m_in: float, usd_per_m_out: float) -> float:
    return (prompt_tokens * usd_per_m_in + completion_tokens * usd_per_m_out) / 1_000_000


def usage_cost(usage, model: str) -> float:
    """Prefer the provider-reported `cost` (OpenRouter); fall back to our dated price table."""
    reported = getattr(usage, "cost", None)
    if reported is None and getattr(usage, "model_extra", None):
        reported = usage.model_extra.get("cost")
    if reported is not None:
        return float(reported)
    p_in, p_out = PRICES.get(model, (0.0, 0.0))
    return estimate_cost(usage.prompt_tokens, usage.completion_tokens, p_in, p_out)


def extract_json(text: str) -> str:
    """Return the outermost {...} in text (models sometimes wrap JSON in prose)."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end < start:
        raise ValueError("no JSON object found in model output")
    return text[start:end + 1]
