"""Napkin math for capacity planning: monthly users → peak load → tokens → dollars → rate limits."""

from __future__ import annotations

from dataclasses import dataclass

SECONDS_PER_DAY = 86_400


@dataclass(frozen=True)
class Workload:
    mau: int  # monthly active users
    dau_ratio: float  # daily / monthly active users ("stickiness")
    sessions_per_dau: float  # chats per active user per day
    turns_per_session: float  # user messages per chat
    system_tokens: int  # system prompt (FAQ) tokens, resent every turn
    user_tokens: int  # tokens per user message
    output_tokens: int  # tokens per assistant reply
    peak_to_avg: float  # busiest-hour rate / average rate
    cached_share: float = 0.0  # share of input tokens served from the provider's prompt cache


@dataclass(frozen=True)
class Price:
    usd_per_m_in: float
    usd_per_m_out: float
    cached_discount: float = 1.0  # cached input price as a fraction of the input price


def input_tokens_per_session(system: int, user: int, output: int, turns: int) -> int:
    """Total prompt tokens for a chat of `turns` messages that resends the whole history.

    Turn k (1-based) sends the system prompt, k user messages and k-1 replies, so input grows
    linearly per turn and quadratically per session.
    """
    if turns < 1:
        raise ValueError("turns must be >= 1")
    return sum(system + k * user + (k - 1) * output for k in range(1, turns + 1))


def plan(w: Workload, latency_s: float) -> dict:
    """Daily volumes, peak rates and peak concurrency (Little's law: L = λ · W)."""
    if not 0 < w.dau_ratio <= 1 or w.peak_to_avg < 1 or not 0 <= w.cached_share <= 1:
        raise ValueError("dau_ratio in (0, 1], peak_to_avg >= 1, cached_share in [0, 1]")
    turns = round(w.turns_per_session)
    dau = w.mau * w.dau_ratio
    sessions = dau * w.sessions_per_dau
    messages = sessions * turns
    tokens_in = sessions * input_tokens_per_session(w.system_tokens, w.user_tokens,
                                                    w.output_tokens, turns)
    tokens_out = messages * w.output_tokens
    avg_rps = messages / SECONDS_PER_DAY
    peak_rps = avg_rps * w.peak_to_avg
    peak_tpm_in = tokens_in / SECONDS_PER_DAY * w.peak_to_avg * 60
    return {
        "dau": dau,
        "sessions_per_day": sessions,
        "messages_per_day": messages,
        "input_tokens_per_day": tokens_in,
        "output_tokens_per_day": tokens_out,
        "avg_rps": avg_rps,
        "peak_rps": peak_rps,
        "peak_rpm": peak_rps * 60,
        "peak_input_tpm": peak_tpm_in,
        "peak_uncached_input_tpm": peak_tpm_in * (1 - w.cached_share),
        "peak_output_tpm": tokens_out / SECONDS_PER_DAY * w.peak_to_avg * 60,
        "peak_concurrency": peak_rps * latency_s,
    }


def daily_cost(p: dict, w: Workload, price: Price) -> float:
    """USD per day for the volumes in `plan(...)` output `p`."""
    tin = p["input_tokens_per_day"]
    cached = tin * w.cached_share
    return ((tin - cached) * price.usd_per_m_in
            + cached * price.usd_per_m_in * price.cached_discount
            + p["output_tokens_per_day"] * price.usd_per_m_out) / 1_000_000


def peak_to_avg_from_hourly(counts_by_hour: dict[int, int]) -> float:
    """Busiest hour's share of the day × 24. A flat day gives 1.0."""
    if len(counts_by_hour) != 24 or any(c < 0 for c in counts_by_hour.values()):
        raise ValueError("need 24 non-negative hourly counts")
    total = sum(counts_by_hour.values())
    if total == 0:
        raise ValueError("no events")
    return max(counts_by_hour.values()) / total * 24


def servers_needed(peak_concurrency: float, per_server: float, headroom: float = 0.7) -> int:
    """Replicas to keep each one at most `headroom` of its measured max concurrent streams."""
    if per_server <= 0 or not 0 < headroom <= 1:
        raise ValueError("per_server > 0 and headroom in (0, 1]")
    n = peak_concurrency / (per_server * headroom)
    return max(1, int(n) + (n > int(n)))
