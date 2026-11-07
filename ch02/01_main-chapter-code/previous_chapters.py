"""Code from earlier chapters that this chapter reuses, copied so ch02 can be read on its own.

From Chapter 1 (ch01/01_main-chapter-code/ch01.ipynb, section "Cost napkin math").
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
