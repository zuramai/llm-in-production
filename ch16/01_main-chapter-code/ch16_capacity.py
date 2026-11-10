"""Napkin math for self-hosting: KV-cache memory per token and $ per million output tokens."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelShape:
    """The few numbers from a Hugging Face `config.json` that size the KV cache."""

    num_layers: int
    num_kv_heads: int
    head_dim: int

    @classmethod
    def from_config(cls, path: Path) -> ModelShape:
        cfg = json.loads(Path(path).read_text())
        heads = cfg["num_attention_heads"]
        head_dim = cfg.get("head_dim") or cfg["hidden_size"] // heads
        return cls(cfg["num_hidden_layers"], cfg.get("num_key_value_heads", heads), head_dim)


def kv_bytes_per_token(shape: ModelShape, dtype_bytes: float = 2) -> int:
    """Bytes of K and V cache one token occupies across all layers (2 = K and V)."""
    if dtype_bytes <= 0:
        raise ValueError("dtype_bytes must be positive")
    return round(2 * shape.num_layers * shape.num_kv_heads * shape.head_dim * dtype_bytes)


def max_cached_tokens(free_bytes: float, shape: ModelShape, dtype_bytes: float = 2) -> int:
    """How many tokens of context fit in `free_bytes` of KV-cache memory (all requests summed)."""
    if free_bytes < 0:
        raise ValueError("free_bytes must be >= 0")
    return int(free_bytes // kv_bytes_per_token(shape, dtype_bytes))


def usd_per_million_tokens(usd_per_hour: float, tokens_per_s: float) -> float:
    """Cost of one million tokens on a machine rented by the hour, kept busy at `tokens_per_s`."""
    if tokens_per_s <= 0:
        raise ValueError("tokens_per_s must be positive")
    if usd_per_hour < 0:
        raise ValueError("usd_per_hour must be >= 0")
    return usd_per_hour / (tokens_per_s * 3600) * 1_000_000


def breakeven_tokens_per_s(usd_per_hour: float, api_usd_per_million: float) -> float:
    """Sustained throughput at which the rented GPU costs the same as the API per token."""
    if api_usd_per_million <= 0:
        raise ValueError("api_usd_per_million must be positive")
    return usd_per_hour / 3600 / api_usd_per_million * 1_000_000
