import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ch16_capacity import (
    ModelShape,
    breakeven_tokens_per_s,
    kv_bytes_per_token,
    max_cached_tokens,
    usd_per_million_tokens,
)

QWEN3_8B = ModelShape(num_layers=36, num_kv_heads=8, head_dim=128)


def test_kv_bytes_per_token_qwen3_8b_bf16():
    # 2 (K and V) x 36 layers x 8 KV heads x 128 dims x 2 bytes = 144 KiB per token
    assert kv_bytes_per_token(QWEN3_8B) == 147_456
    assert kv_bytes_per_token(QWEN3_8B, dtype_bytes=1) == 73_728  # FP8 KV cache halves it
    with pytest.raises(ValueError):
        kv_bytes_per_token(QWEN3_8B, dtype_bytes=0)


def test_max_cached_tokens_floors():
    assert max_cached_tokens(147_456 * 10 + 147_455, QWEN3_8B) == 10
    assert max_cached_tokens(0, QWEN3_8B) == 0
    with pytest.raises(ValueError):
        max_cached_tokens(-1, QWEN3_8B)


def test_model_shape_from_config_handles_missing_head_dim_and_gqa(tmp_path):
    explicit = tmp_path / "a.json"
    explicit.write_text(json.dumps({"num_hidden_layers": 36, "num_attention_heads": 32,
                                    "num_key_value_heads": 8, "head_dim": 128,
                                    "hidden_size": 4096}))
    assert ModelShape.from_config(explicit) == QWEN3_8B
    derived = tmp_path / "b.json"  # older configs: no head_dim, no GQA
    derived.write_text(json.dumps({"num_hidden_layers": 2, "num_attention_heads": 16,
                                   "hidden_size": 1024}))
    assert ModelShape.from_config(derived) == ModelShape(2, 16, 64)


def test_usd_per_million_tokens_and_breakeven_are_inverse():
    # $0.36/h at 1,000 tok/s -> 3.6M tokens per hour -> $0.10 per 1M
    assert usd_per_million_tokens(0.36, 1000) == pytest.approx(0.10)
    assert breakeven_tokens_per_s(0.36, 0.10) == pytest.approx(1000)
    with pytest.raises(ValueError):
        usd_per_million_tokens(0.36, 0)
    with pytest.raises(ValueError):
        usd_per_million_tokens(-1, 10)
    with pytest.raises(ValueError):
        breakeven_tokens_per_s(0.36, 0)
