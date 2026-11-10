"""Static checks on the gateway config: the mistakes here cost money or availability silently."""

import re
from pathlib import Path

import pytest
import yaml

HERE = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((HERE / "config.yaml").read_text())
COMPOSE = yaml.safe_load((HERE / "docker-compose.yml").read_text())
MODELS = {m["model_name"]: m["litellm_params"] for m in CONFIG["model_list"]}


def test_aliases_point_at_the_models_the_chapter_promises():
    assert MODELS["fast"]["model"] == "openrouter/openai/gpt-6-luna"
    assert MODELS["smart"]["model"] == "openrouter/anthropic/claude-sonnet-5.5"
    assert MODELS["local"]["model"].endswith("qwen3:8b")


def test_every_deployment_has_a_price_and_an_output_cap():
    # Without a price, spend stays 0 and budgets never trigger; without max_tokens, OpenRouter
    # reserves credit for 65k output tokens per request.
    for name, params in MODELS.items():
        assert "input_cost_per_token" in params and "output_cost_per_token" in params, name
        assert 0 < params["max_tokens"] <= 4096, name


def test_prices_match_the_book_table():
    per_m = {n: (p["input_cost_per_token"] * 1e6, p["output_cost_per_token"] * 1e6)
             for n, p in MODELS.items()}
    assert per_m["fast"] == pytest.approx((0.10, 0.50))
    assert per_m["smart"] == pytest.approx((2.0, 10.0))
    assert per_m["local"] == (0, 0)


def test_reasoning_defaults_are_set_at_the_gateway():
    assert MODELS["local"]["reasoning_effort"] == "none"     # qwen3 thinks for minutes otherwise
    assert MODELS["smart"]["reasoning_effort"] == "low"      # hidden tokens ate max_tokens


def test_fallbacks_reference_existing_aliases_and_never_loop():
    fallbacks = {k: v for d in CONFIG["router_settings"]["fallbacks"] for k, v in d.items()}
    assert fallbacks["fast"] == ["local"]
    for src, targets in fallbacks.items():
        assert src in MODELS and all(t in MODELS for t in targets)
        assert src not in targets
        # a fallback target that itself falls back to the source would ping-pong
        for t in targets:
            assert src not in fallbacks.get(t, [])


def test_exact_cache_is_redis_with_a_ttl():
    s = CONFIG["litellm_settings"]
    assert s["cache"] is True
    assert s["cache_params"]["type"] == "redis"
    assert 0 < s["cache_params"]["ttl"] <= 86_400


def test_secrets_come_from_the_environment():
    for name, params in MODELS.items():
        key = params.get("api_key", "")
        assert key == "ollama" or key.startswith("os.environ/"), name
    assert CONFIG["general_settings"]["master_key"].startswith("os.environ/")


def test_gateway_image_is_pinned_to_a_release():
    image = COMPOSE["services"]["litellm"]["image"]
    assert image.startswith("ghcr.io/berriai/litellm:")
    assert re.fullmatch(r"v\d+\.\d+\.\d+(-lts)?", image.split(":")[1]), image


def test_ports_are_the_chapter_ports_and_local_only():
    assert "--port" in (cmd := COMPOSE["services"]["litellm"]["command"])
    assert cmd[cmd.index("--port") + 1] == "4015"
    assert COMPOSE["services"]["litellm-db"]["ports"] == ["127.0.0.1:5435:5432"]
    assert COMPOSE["services"]["redis"]["ports"] == ["127.0.0.1:6385:6379"]
