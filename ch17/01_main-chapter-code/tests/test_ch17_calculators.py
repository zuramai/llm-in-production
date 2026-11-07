"""Unit tests for the VRAM and cost calculators. Hand-computed expectations, no network."""

import json
import sys
from pathlib import Path

import pytest

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

import ch17_cost as cost
import ch17_vram as vram

CONFIGS = CODE_DIR / "configs"


@pytest.fixture(scope="module")
def qwen():
    return vram.load_spec(CONFIGS / "qwen3-8b.json")


@pytest.fixture(scope="module")
def llama70():
    return vram.load_spec(CONFIGS / "llama-3.3-70b.json")


@pytest.fixture(scope="module")
def gptoss():
    return vram.load_spec(CONFIGS / "gpt-oss-120b.json")


# ---------------------------------------------------------------- parameters


def test_qwen3_8b_param_count_matches_hand_computation(qwen):
    # embed 2*151936*4096 + 36 * (attn 41,943,040 + mlp 150,994,944)
    assert vram.total_params(qwen) == 8_190_427_136


def test_llama70b_params_close_to_published(llama70):
    assert vram.total_params(llama70) == pytest.approx(70.55e9, rel=0.001)


def test_tied_embeddings_count_once():
    base = {
        "name": "t",
        "hidden_size": 64,
        "num_layers": 2,
        "num_heads": 4,
        "num_kv_heads": 2,
        "head_dim": 16,
        "intermediate_size": 128,
        "vocab_size": 1000,
    }
    untied = vram.ModelSpec(**base, tie_embeddings=False)
    tied = vram.ModelSpec(**base, tie_embeddings=True)
    assert vram.total_params(untied) - vram.total_params(tied) == 1000 * 64


def test_moe_total_vs_active(gptoss):
    total = vram.total_params(gptoss)
    active = vram.active_params(gptoss)
    assert total == pytest.approx(116.8e9, rel=0.005)
    assert active == pytest.approx(5.1e9, rel=0.02)
    # 4 of 128 experts run per token
    assert active < total / 20


def test_dense_active_excludes_only_input_embedding(qwen):
    diff = vram.total_params(qwen) - vram.active_params(qwen)
    assert diff == qwen.vocab_size * qwen.hidden_size


# ---------------------------------------------------------------- weights


def test_weight_bytes_dtypes(qwen):
    n = vram.total_params(qwen)
    assert vram.weight_bytes(qwen, "bf16") == n * 2
    assert vram.weight_bytes(qwen, "fp8") == n
    assert vram.weight_bytes(qwen, "int4") == n / 2
    assert vram.weight_bytes(qwen, None) == n * 2  # shipped as bf16


def test_gptoss_as_shipped_is_mixed_precision(gptoss):
    experts = vram.expert_params(gptoss)
    rest = vram.total_params(gptoss) - experts
    expected = experts * 4.25 / 8 + rest * 2
    assert vram.weight_bytes(gptoss) == pytest.approx(expected)
    # about 60 GiB: this is why the model fits on one 80 GB GPU
    assert 58 < vram.weight_bytes(gptoss) / vram.GIB < 63


def test_unknown_dtype_raises(qwen):
    with pytest.raises(ValueError, match="unknown dtype"):
        vram.weight_bytes(qwen, "fp3")


# ---------------------------------------------------------------- KV cache


def test_qwen_kv_per_token(qwen):
    # 2 * 36 layers * 8 kv heads * 128 dim * 2 bytes
    assert vram.kv_bytes_per_token(qwen) == 147_456
    assert vram.kv_bytes_per_token(qwen, "fp8") == 73_728


def test_qwen_kv_cache_32k(qwen):
    assert vram.kv_cache_bytes(qwen, 32_768) == 147_456 * 32_768 == 4_831_838_208
    assert vram.kv_cache_bytes(qwen, 32_768) / vram.GIB == pytest.approx(4.5)


def test_kv_scales_linearly_with_batch_and_seq(qwen):
    one = vram.kv_cache_bytes(qwen, 1000, 1)
    assert vram.kv_cache_bytes(qwen, 1000, 8) == 8 * one
    assert vram.kv_cache_bytes(qwen, 4000, 1) == 4 * one
    assert vram.kv_cache_bytes(qwen, 0, 8) == 0


def test_llama70b_kv_per_token(llama70):
    assert vram.kv_bytes_per_token(llama70) == 327_680


def test_gqa_cache_is_smaller_than_mha(qwen):
    mha = vram.ModelSpec(**{**qwen.__dict__, "num_kv_heads": qwen.num_heads})
    assert vram.kv_cache_bytes(mha, 1000) == 4 * vram.kv_cache_bytes(qwen, 1000)


def test_sliding_window_layers_cap_the_cache(gptoss):
    per_layer_token = 2 * 8 * 64 * 2  # K+V, 8 kv heads, head_dim 64, bf16
    seq = 32_768
    full = 18 * seq * per_layer_token
    sliding = 18 * 128 * per_layer_token
    assert vram.kv_cache_bytes(gptoss, seq) == full + sliding
    assert (
        vram.kv_cache_bytes(gptoss, seq, respect_sliding_window=False) == 36 * seq * per_layer_token
    )


def test_sliding_window_does_not_inflate_short_sequences(gptoss):
    # below the window every layer holds the whole sequence
    assert vram.kv_cache_bytes(gptoss, 100) == vram.kv_cache_bytes(
        gptoss, 100, respect_sliding_window=False
    )


def test_negative_inputs_rejected(qwen):
    with pytest.raises(ValueError):
        vram.kv_cache_bytes(qwen, -1)
    with pytest.raises(ValueError):
        vram.kv_cache_bytes(qwen, 10, kv_dtype="fp3")


# ---------------------------------------------------------------- estimate and fit


def test_estimate_total_is_sum_of_parts(qwen):
    e = vram.estimate(qwen, seq_len=4096, batch=4)
    assert e.total == e.weights + e.kv_cache + e.activations + e.overhead
    assert e.kv_cache == vram.kv_cache_bytes(qwen, 4096, 4)


def test_max_batch_is_tight(qwen):
    gpu_gb = 80
    n = vram.max_batch(qwen, 4096, gpu_gb)
    budget = vram.usable_bytes(gpu_gb)
    assert n > 0
    assert vram.estimate(qwen, 4096, n).total <= budget
    assert vram.estimate(qwen, 4096, n + 1).total > budget


def test_max_batch_zero_when_weights_do_not_fit(llama70):
    assert vram.max_batch(llama70, 4096, gpu_gb=80) == 0  # 131 GiB of bf16 weights


def test_70b_fits_on_two_h100_in_fp8_but_not_one(llama70):
    assert vram.max_batch(llama70, 8192, 80, dtype="fp8") == 0
    assert vram.estimate(llama70, 8192, 1, dtype="fp8").total > vram.usable_bytes(80)
    assert vram.max_batch(llama70, 8192, 80, n_gpus=2, dtype="fp8") > 0


def test_fp8_kv_cache_roughly_doubles_batch(qwen):
    bf16 = vram.max_batch(qwen, 8192, 80)
    fp8 = vram.max_batch(qwen, 8192, 80, kv_dtype="fp8")
    assert fp8 > 1.9 * bf16


def test_max_context_is_tight(qwen):
    ctx = vram.max_context(qwen, 24, batch=1)
    budget = vram.usable_bytes(24)
    assert vram.estimate(qwen, ctx, 1).total <= budget
    assert vram.estimate(qwen, ctx + 1, 1).total > budget


def test_max_context_zero_when_nothing_fits(llama70):
    assert vram.max_context(llama70, 24) == 0


def test_usable_bytes_validates_utilization():
    assert vram.usable_bytes(80, 2, 0.5) == 80e9
    with pytest.raises(ValueError):
        vram.usable_bytes(80, utilization=1.5)


def test_load_all_reads_every_config():
    specs = vram.load_all(CONFIGS)
    assert {"qwen3-8b", "llama-3.3-70b", "gpt-oss-120b"} <= set(specs)


def test_load_spec_defaults_head_dim_and_kv_heads(tmp_path):
    cfg = {
        "hidden_size": 512,
        "num_hidden_layers": 2,
        "num_attention_heads": 8,
        "intermediate_size": 1024,
        "vocab_size": 100,
    }
    p = tmp_path / "tiny.json"
    p.write_text(json.dumps(cfg))
    spec = vram.load_spec(p)
    assert spec.head_dim == 64
    assert spec.num_kv_heads == 8  # no GQA: one kv head per query head


# ---------------------------------------------------------------- cost


def test_cost_per_million_hand_computed():
    # $3.99/h, 1000 tok/s -> 3.6M tokens/h -> 3.99 / 3.6 = 1.10833 per 1M
    assert cost.cost_per_million_tokens(3.99, 1, 1000) == pytest.approx(1.108333, rel=1e-5)
    assert cost.cost_per_million_tokens(3.99, 2, 1000) == pytest.approx(2.216667, rel=1e-5)


def test_idle_gpu_multiplies_cost():
    full = cost.cost_per_million_tokens(2.0, 1, 500, utilization=1.0)
    half = cost.cost_per_million_tokens(2.0, 1, 500, utilization=0.5)
    assert half == pytest.approx(2 * full)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"gpu_hourly_usd": 1, "n_gpus": 0, "tokens_per_second": 10},
        {"gpu_hourly_usd": 1, "n_gpus": 1, "tokens_per_second": 0},
        {"gpu_hourly_usd": 1, "n_gpus": 1, "tokens_per_second": 10, "utilization": 0},
        {"gpu_hourly_usd": 1, "n_gpus": 1, "tokens_per_second": 10, "utilization": 1.1},
        {"gpu_hourly_usd": -1, "n_gpus": 1, "tokens_per_second": 10},
    ],
)
def test_cost_rejects_bad_inputs(kwargs):
    with pytest.raises(ValueError):
        cost.cost_per_million_tokens(**kwargs)


def test_breakeven_utilization():
    # saturated cost 1.108333; API 2.216666 -> break even at 50% utilization
    u = cost.breakeven_utilization(3.99, 1, 1000, 2.216667)
    assert u == pytest.approx(0.5, rel=1e-4)
    # self-hosting at that utilization costs exactly the API price
    assert cost.cost_per_million_tokens(3.99, 1, 1000, u) == pytest.approx(2.216667, rel=1e-4)


def test_breakeven_none_when_api_is_cheaper_even_at_full_load():
    assert cost.breakeven_utilization(7.98, 2, 7648.67, 0.067) is None


def test_blended_price():
    assert cost.blended_api_price(1.0, 5.0, input_share=0.75) == pytest.approx(2.0)
    assert cost.blended_api_price(1.0, 5.0, input_share=1.0) == 1.0
    with pytest.raises(ValueError):
        cost.blended_api_price(1.0, 5.0, input_share=1.5)


# ---------------------------------------------------------------- roofline


def test_roofline_dense_batch_one(qwen):
    r = cost.decode_roofline(qwen, bandwidth_gb_s=200, dtype="bf16")
    weights = 8_190_427_136 * 2
    assert r.bytes_per_step == weights
    assert r.tokens_per_second == pytest.approx(200e9 / weights)  # about 12.2 tok/s


def test_roofline_batching_multiplies_throughput_until_kv_dominates(qwen):
    one = cost.decode_roofline(qwen, 3350, batch=1).tokens_per_second
    eight = cost.decode_roofline(qwen, 3350, batch=8).tokens_per_second
    assert eight == pytest.approx(8 * one)  # weights are read once per step
    long_ctx = cost.decode_roofline(qwen, 3350, batch=8, context=32_768).tokens_per_second
    assert long_ctx < eight


def test_roofline_efficiency_scales_speed(qwen):
    full = cost.decode_roofline(qwen, 200, efficiency=1.0).tokens_per_second
    half = cost.decode_roofline(qwen, 200, efficiency=0.5).tokens_per_second
    assert half == pytest.approx(full / 2)


def test_roofline_moe_reads_few_experts_at_batch_one_all_at_large_batch(gptoss):
    b1 = cost.decode_roofline(gptoss, 3350, batch=1)
    b256 = cost.decode_roofline(gptoss, 3350, batch=256)
    assert b1.bytes_per_step < 0.1 * vram.weight_bytes(gptoss)
    assert b256.bytes_per_step == pytest.approx(vram.weight_bytes(gptoss))


def test_roofline_rejects_bad_inputs(qwen):
    with pytest.raises(ValueError):
        cost.decode_roofline(qwen, 200, batch=0)
    with pytest.raises(ValueError):
        cost.decode_roofline(qwen, 200, efficiency=0)


def test_price_file_is_well_formed():
    prices = cost.load_prices()
    assert prices["accessed"] == "2026-11-07"
    for name, run in prices["measured_throughput"].items():
        assert run["output_tokens_per_second"] > 0 and run["n_gpus"] >= 1, name
        assert run["source"].startswith("https://")
