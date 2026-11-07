"""Code from earlier chapters that this chapter reuses, copied so ch04 can be read on its own.

From Chapter 1 (ch01/01_main-chapter-code/ch01.ipynb, section "Cost napkin math") and
Chapter 6 (the strict JSON-schema request shape used by the SQL planner).
"""

from __future__ import annotations

# $ per 1M tokens (input, output) on OpenRouter, as of 2026-11-07.
PRICES = {
    "openai/gpt-6-luna": (0.10, 0.50),
    "anthropic/claude-haiku-5.5": (0.10, 0.50),
    "anthropic/claude-sonnet-5.5": (2.00, 10.00),
    "qwen/qwen3.8-flash": (0.15, 0.47),
}


def estimate_cost(prompt_tokens: int, completion_tokens: int,
                  usd_per_m_in: float, usd_per_m_out: float) -> float:
    return (prompt_tokens * usd_per_m_in + completion_tokens * usd_per_m_out) / 1_000_000


def json_schema_format(name: str, schema: dict) -> dict:
    """`response_format` for strict structured output (works on gpt-6-luna, Claude 5.5, Ollama)."""
    return {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": schema}}
