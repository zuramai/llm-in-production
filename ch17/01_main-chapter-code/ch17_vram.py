"""VRAM calculator: will this model fit on this GPU, and with how much context and batch?

Everything is derived from a Hugging Face ``config.json``. The formulas are the ones in the
chapter:

    weights  = params * bytes_per_param
    kv_cache = 2 * layers * kv_heads * head_dim * bytes * seq_len * batch
    overhead = CUDA context + activation working set (an estimate, see ``activation_bytes``)

Pure Python, standard library only, no network.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

GIB = 1024**3

# Bytes per stored parameter. Block-quantized formats pay for their scales:
#   q8_0   = 32 int8 values + one fp16 scale        -> 34 bytes / 32 values = 8.5 bits
#   q4_k_m = ~4.85 bits per weight on average (llama.cpp mixes q4_K and q6_K tensors)
#   mxfp4  = 4-bit values + one 8-bit scale per 32  -> 4.25 bits
#   nvfp4  = 4-bit values + one fp8 scale per 16     -> 4.5 bits (plus a tiny per-tensor scale)
DTYPE_BYTES: dict[str, float] = {
    "fp32": 4.0,
    "fp16": 2.0,
    "bf16": 2.0,
    "fp8": 1.0,
    "int8": 1.0,
    "q8_0": 34 / 32,
    "q4_k_m": 4.85 / 8,
    "mxfp4": 4.25 / 8,
    "nvfp4": 4.5 / 8,
    "int4": 0.5,
}

# KV cache dtypes people actually use. fp8 halves the cache; the accuracy cost is model-specific.
KV_DTYPE_BYTES = {"fp32": 4.0, "fp16": 2.0, "bf16": 2.0, "fp8": 1.0, "int8": 1.0}

# Memory of common accelerators in GiB-ish marketing units (GB as printed on the spec sheet).
GPU_MEMORY_GB = {
    "RTX 4090": 24,
    "L40S": 48,
    "A100 80GB": 80,
    "H100 80GB": 80,
    "H200": 141,
    "B200": 180,
    "M2 Pro (unified)": 16,
}


@dataclass(frozen=True)
class ModelSpec:
    """The fields of a ``config.json`` that decide memory."""

    name: str
    hidden_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    tie_embeddings: bool = False
    attention_bias: bool = False
    # Mixture of experts: 0 means a dense model.
    num_experts: int = 0
    experts_per_token: int = 0
    # Per-layer attention type ("full_attention" / "sliding_attention") and the window size.
    layer_types: tuple[str, ...] = field(default_factory=tuple)
    sliding_window: int | None = None
    native_dtype: str = "bf16"
    # Checkpoints that store the expert weights in a different format than everything else.
    expert_dtype: str | None = None

    @property
    def is_moe(self) -> bool:
        return self.num_experts > 0


def load_spec(path: str | Path, name: str | None = None) -> ModelSpec:
    """Build a ModelSpec from a Hugging Face ``config.json`` file."""
    path = Path(path)
    cfg = json.loads(path.read_text())
    hidden = cfg["hidden_size"]
    heads = cfg["num_attention_heads"]
    num_experts = cfg.get("num_local_experts") or cfg.get("num_experts") or 0
    per_tok = cfg.get("num_experts_per_tok") or cfg.get("experts_per_token") or 0
    quant = cfg.get("quantization_config") or {}
    torch_dtype = cfg.get("torch_dtype") or cfg.get("dtype") or "bfloat16"
    return ModelSpec(
        name=name or path.stem,
        hidden_size=hidden,
        num_layers=cfg["num_hidden_layers"],
        num_heads=heads,
        num_kv_heads=cfg.get("num_key_value_heads", heads),
        head_dim=cfg.get("head_dim") or hidden // heads,
        intermediate_size=cfg["intermediate_size"],
        vocab_size=cfg["vocab_size"],
        tie_embeddings=bool(cfg.get("tie_word_embeddings", False)),
        attention_bias=bool(cfg.get("attention_bias", False)),
        num_experts=num_experts,
        experts_per_token=per_tok,
        layer_types=tuple(cfg.get("layer_types") or ()),
        sliding_window=cfg.get("sliding_window"),
        native_dtype={"bfloat16": "bf16", "float16": "fp16", "float32": "fp32"}.get(
            torch_dtype, "bf16"
        ),
        expert_dtype="mxfp4" if quant.get("quant_method") == "mxfp4" else None,
    )


# --------------------------------------------------------------------------- parameters


def embedding_params(spec: ModelSpec) -> int:
    """Input embedding plus output head (shared when ``tie_embeddings``)."""
    table = spec.vocab_size * spec.hidden_size
    return table if spec.tie_embeddings else 2 * table


def attention_params_per_layer(spec: ModelSpec) -> int:
    q = spec.hidden_size * spec.num_heads * spec.head_dim
    kv = 2 * spec.hidden_size * spec.num_kv_heads * spec.head_dim
    o = spec.num_heads * spec.head_dim * spec.hidden_size
    total = q + kv + o
    if spec.attention_bias:
        total += (spec.num_heads + 2 * spec.num_kv_heads) * spec.head_dim + spec.hidden_size
    return total


def mlp_params_per_expert(spec: ModelSpec) -> int:
    """SwiGLU MLP: gate, up and down projections."""
    return 3 * spec.hidden_size * spec.intermediate_size


def expert_params(spec: ModelSpec) -> int:
    """All expert weights in the model (0 for dense models)."""
    return spec.num_layers * spec.num_experts * mlp_params_per_expert(spec)


def total_params(spec: ModelSpec) -> int:
    """Every weight we must store. Ignores norms (< 0.01% of the total)."""
    attn = spec.num_layers * attention_params_per_layer(spec)
    if spec.is_moe:
        router = spec.num_layers * spec.hidden_size * spec.num_experts
        return embedding_params(spec) + attn + router + expert_params(spec)
    return embedding_params(spec) + attn + spec.num_layers * mlp_params_per_expert(spec)


def active_params(spec: ModelSpec) -> int:
    """Weights that take part in one decoded token.

    For a dense model that is everything except the input embedding lookup (a table read, not a
    matmul). For a MoE only ``experts_per_token`` of the experts run. Decode speed is bounded by
    *this* number, memory by ``total_params``.
    """
    attn = spec.num_layers * attention_params_per_layer(spec)
    out_head = spec.vocab_size * spec.hidden_size
    if spec.is_moe:
        router = spec.num_layers * spec.hidden_size * spec.num_experts
        experts = spec.num_layers * spec.experts_per_token * mlp_params_per_expert(spec)
        return out_head + attn + router + experts
    return out_head + attn + spec.num_layers * mlp_params_per_expert(spec)


# --------------------------------------------------------------------------- memory


def _bytes(dtype: str) -> float:
    try:
        return DTYPE_BYTES[dtype]
    except KeyError:
        raise ValueError(f"unknown dtype {dtype!r}; choose from {sorted(DTYPE_BYTES)}") from None


def weight_bytes(spec: ModelSpec, dtype: str | None = None) -> float:
    """Bytes of weights.

    ``dtype=None`` means "as shipped": a checkpoint like gpt-oss-120b keeps its experts in mxfp4
    and everything else in bf16. Passing a dtype quantizes *every* weight to it.
    """
    if dtype is not None:
        return total_params(spec) * _bytes(dtype)
    if spec.expert_dtype:
        experts = expert_params(spec)
        rest = total_params(spec) - experts
        return experts * _bytes(spec.expert_dtype) + rest * _bytes(spec.native_dtype)
    return total_params(spec) * _bytes(spec.native_dtype)


def kv_bytes_per_token(spec: ModelSpec, kv_dtype: str = "bf16") -> float:
    """2 (K and V) * layers * kv_heads * head_dim * bytes."""
    if kv_dtype not in KV_DTYPE_BYTES:
        raise ValueError(f"unknown kv dtype {kv_dtype!r}; choose from {sorted(KV_DTYPE_BYTES)}")
    return 2 * spec.num_layers * spec.num_kv_heads * spec.head_dim * KV_DTYPE_BYTES[kv_dtype]


def kv_cache_bytes(
    spec: ModelSpec,
    seq_len: int,
    batch: int = 1,
    kv_dtype: str = "bf16",
    respect_sliding_window: bool = True,
) -> float:
    """KV cache for ``batch`` sequences of ``seq_len`` tokens.

    Sliding-window layers only keep ``sliding_window`` tokens, which is why gpt-oss's cache is
    about half of what the plain formula says at long context. Engines that do not implement the
    window pay the full price, so set ``respect_sliding_window=False`` to see that number.
    """
    if seq_len < 0 or batch < 0:
        raise ValueError("seq_len and batch must be >= 0")
    if kv_dtype not in KV_DTYPE_BYTES:
        raise ValueError(f"unknown kv dtype {kv_dtype!r}; choose from {sorted(KV_DTYPE_BYTES)}")
    per_layer_token = 2 * spec.num_kv_heads * spec.head_dim * KV_DTYPE_BYTES[kv_dtype]
    if respect_sliding_window and spec.layer_types and spec.sliding_window:
        tokens_per_layer = [
            min(seq_len, spec.sliding_window) if t == "sliding_attention" else seq_len
            for t in spec.layer_types
        ]
    else:
        tokens_per_layer = [seq_len] * spec.num_layers
    return batch * per_layer_token * sum(tokens_per_layer)


def activation_bytes(
    spec: ModelSpec, max_batched_tokens: int = 8192, act_dtype: str = "bf16"
) -> float:
    """Rough peak activation memory of one forward pass over ``max_batched_tokens`` tokens.

    One layer's working set is reused layer after layer, so it does not scale with depth:
    roughly 4 hidden-size tensors (residual, norm output, attention output, MLP input) plus the
    gated MLP intermediate (2 tensors of the *active* intermediate width) per token, plus fp32
    logits for one position per sequence (approximated by one row of vocab size per 128 tokens).
    This is an estimate, not a profile. Real engines profile at startup (vLLM does a dummy
    forward pass) and you should trust that number over this one.
    """
    b = KV_DTYPE_BYTES[act_dtype]
    width = 4 * spec.hidden_size + 2 * spec.intermediate_size * max(spec.experts_per_token, 1)
    logits = (max_batched_tokens // 128 + 1) * spec.vocab_size * 4
    return max_batched_tokens * width * b + logits


CUDA_CONTEXT_BYTES = 1.0 * GIB  # CUDA context, cuBLAS workspace, graph pools: ~1 GiB in practice


@dataclass(frozen=True)
class Estimate:
    spec_name: str
    dtype: str
    kv_dtype: str
    seq_len: int
    batch: int
    weights: float
    kv_cache: float
    activations: float
    overhead: float

    @property
    def total(self) -> float:
        return self.weights + self.kv_cache + self.activations + self.overhead

    def row(self) -> dict[str, float | str | int]:
        return {
            "model": self.spec_name,
            "dtype": self.dtype,
            "kv_dtype": self.kv_dtype,
            "seq_len": self.seq_len,
            "batch": self.batch,
            "weights_GiB": round(self.weights / GIB, 2),
            "kv_GiB": round(self.kv_cache / GIB, 2),
            "activations_GiB": round(self.activations / GIB, 2),
            "overhead_GiB": round(self.overhead / GIB, 2),
            "total_GiB": round(self.total / GIB, 2),
        }


def estimate(
    spec: ModelSpec,
    seq_len: int,
    batch: int = 1,
    dtype: str | None = None,
    kv_dtype: str = "bf16",
    max_batched_tokens: int = 8192,
    respect_sliding_window: bool = True,
) -> Estimate:
    """Full memory estimate for serving ``batch`` concurrent sequences of ``seq_len`` tokens."""
    shipped = dtype or (spec.expert_dtype and "as shipped") or spec.native_dtype
    return Estimate(
        spec_name=spec.name,
        dtype=str(shipped),
        kv_dtype=kv_dtype,
        seq_len=seq_len,
        batch=batch,
        weights=weight_bytes(spec, dtype),
        kv_cache=kv_cache_bytes(spec, seq_len, batch, kv_dtype, respect_sliding_window),
        activations=activation_bytes(spec, max_batched_tokens),
        overhead=CUDA_CONTEXT_BYTES,
    )


def usable_bytes(gpu_gb: float, n_gpus: int = 1, utilization: float = 0.90) -> float:
    """What an engine may use: vLLM's ``--gpu-memory-utilization`` (default 0.9) of each GPU."""
    if not 0 < utilization <= 1:
        raise ValueError("utilization must be in (0, 1]")
    return gpu_gb * 1e9 * n_gpus * utilization


def max_batch(
    spec: ModelSpec,
    seq_len: int,
    gpu_gb: float,
    n_gpus: int = 1,
    dtype: str | None = None,
    kv_dtype: str = "bf16",
    utilization: float = 0.90,
) -> int:
    """Largest number of concurrent ``seq_len``-token sequences that fit. 0 means the weights
    alone (plus overhead) do not fit or not even one sequence fits."""
    base = estimate(spec, seq_len, batch=0, dtype=dtype, kv_dtype=kv_dtype)
    free = usable_bytes(gpu_gb, n_gpus, utilization) - base.total
    per_seq = kv_cache_bytes(spec, seq_len, 1, kv_dtype)
    if free <= 0 or per_seq == 0:
        return 0
    return int(free // per_seq)


def max_context(
    spec: ModelSpec,
    gpu_gb: float,
    batch: int = 1,
    n_gpus: int = 1,
    dtype: str | None = None,
    kv_dtype: str = "bf16",
    utilization: float = 0.90,
) -> int:
    """Longest context (tokens) that fits for ``batch`` sequences, found by bisection so
    sliding-window models are handled correctly. Capped at 10 million tokens."""
    budget = usable_bytes(gpu_gb, n_gpus, utilization)

    def fits(seq: int) -> bool:
        return estimate(spec, seq, batch, dtype, kv_dtype).total <= budget

    if not fits(0):
        return 0
    lo, hi = 0, 10_000_000
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if fits(mid):
            lo = mid
        else:
            hi = mid - 1
    return lo


def load_all(config_dir: str | Path) -> dict[str, ModelSpec]:
    """Load every ``*.json`` in a directory, keyed by file stem."""
    return {p.stem: load_spec(p) for p in sorted(Path(config_dir).glob("*.json"))}
