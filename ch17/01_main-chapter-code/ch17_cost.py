"""Cost per million tokens: renting a GPU vs paying an API, plus a decode-speed roofline.

Pure Python, no network. Prices and throughputs live in ``gpu_prices.json`` with sources and
dates, because both go stale within weeks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ch17_vram import ModelSpec, active_params, kv_cache_bytes, total_params, weight_bytes

HERE = Path(__file__).parent


def load_prices(path: str | Path = HERE / "gpu_prices.json") -> dict:
    return json.loads(Path(path).read_text())


def cost_per_million_tokens(
    gpu_hourly_usd: float,
    n_gpus: int,
    tokens_per_second: float,
    utilization: float = 1.0,
) -> float:
    """USD to generate 1M tokens on a rented node.

    ``tokens_per_second`` is the whole node's throughput at saturation. ``utilization`` is the
    fraction of paid hours the node is actually busy: you pay for an idle GPU too, so the cost
    per *useful* token is the saturated cost divided by utilization.
    """
    if gpu_hourly_usd < 0 or n_gpus < 1:
        raise ValueError("gpu_hourly_usd must be >= 0 and n_gpus >= 1")
    if tokens_per_second <= 0:
        raise ValueError("tokens_per_second must be > 0")
    if not 0 < utilization <= 1:
        raise ValueError("utilization must be in (0, 1]")
    node_per_hour = gpu_hourly_usd * n_gpus
    tokens_per_hour = tokens_per_second * 3600 * utilization
    return node_per_hour / tokens_per_hour * 1_000_000


def breakeven_utilization(
    gpu_hourly_usd: float, n_gpus: int, tokens_per_second: float, api_usd_per_million: float
) -> float | None:
    """Utilization at which self-hosting costs the same as the API.

    Returns ``None`` when even a 100% busy node is more expensive than the API (no utilization
    can win), otherwise a value in (0, 1].
    """
    if api_usd_per_million <= 0:
        raise ValueError("api_usd_per_million must be > 0")
    saturated = cost_per_million_tokens(gpu_hourly_usd, n_gpus, tokens_per_second, 1.0)
    if saturated > api_usd_per_million:
        return None
    return saturated / api_usd_per_million


def blended_api_price(input_per_m: float, output_per_m: float, input_share: float = 0.75) -> float:
    """One number for an API's price given the share of tokens that are input tokens."""
    if not 0 <= input_share <= 1:
        raise ValueError("input_share must be in [0, 1]")
    return input_share * input_per_m + (1 - input_share) * output_per_m


@dataclass(frozen=True)
class Roofline:
    """Upper bound on decode speed when the GPU is memory-bandwidth bound.

    Each decode step streams the active weights once for the whole batch, plus every sequence's
    KV cache. ``tokens_per_second = batch / step_time`` with
    ``step_time = bytes_read_per_step / (bandwidth * efficiency)``.
    """

    bytes_per_step: float
    step_seconds: float
    tokens_per_second: float


def decode_roofline(
    spec: ModelSpec,
    bandwidth_gb_s: float,
    dtype: str | None = None,
    batch: int = 1,
    context: int = 0,
    kv_dtype: str = "bf16",
    efficiency: float = 1.0,
) -> Roofline:
    """Memory-bound decode speed.

    Dense models read all weights each step. MoE models read the weights of every expert any
    sequence in the batch routes to, which for a batch >= ~32 is close to all of them, so we read
    ``min(total, active * batch)`` bytes (a simple cap; real routing is workload dependent).
    """
    if batch < 1:
        raise ValueError("batch must be >= 1")
    if not 0 < efficiency <= 1:
        raise ValueError("efficiency must be in (0, 1]")
    total = weight_bytes(spec, dtype)
    if spec.is_moe:
        active_fraction = active_params(spec) / total_params(spec)
        weights_read = min(total, total * active_fraction * batch)
    else:
        weights_read = total
    kv_read = kv_cache_bytes(spec, context, batch, kv_dtype)
    bytes_per_step = weights_read + kv_read
    step = bytes_per_step / (bandwidth_gb_s * 1e9 * efficiency)
    return Roofline(bytes_per_step, step, batch / step)
