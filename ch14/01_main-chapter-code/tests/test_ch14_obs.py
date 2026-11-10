import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ch14_obs as obs


def _usage(**kw):
    """An OpenAI SDK CompletionUsage lookalike, with OpenRouter's `cost` in model_extra."""
    extra = {"cost": kw.pop("cost")} if "cost" in kw else {}
    return SimpleNamespace(model_extra=extra, **kw)


def test_mask_redacts_emails_in_nested_structures():
    data = {"question": "Orders for jane.doe+vip@example.co.uk?",
            "rows": [{"note": "cc bob@shop.io"}, 42, None],
            "tuple": ("x@y.com", "no email")}
    masked = obs.mask(data=data)
    assert masked == {"question": "Orders for <email>?",
                      "rows": [{"note": "cc <email>"}, 42, None],
                      "tuple": ["<email>", "no email"]}
    assert data["question"].startswith("Orders for jane")  # input not mutated


def test_mask_leaves_non_email_at_signs_alone():
    assert obs.mask(data="SELECT 1 @ 2; handle @shopflow") == "SELECT 1 @ 2; handle @shopflow"


def test_usage_details_maps_openrouter_usage_with_breakdowns():
    usage = _usage(prompt_tokens=2100, completion_tokens=150, total_tokens=2250,
                   prompt_tokens_details=SimpleNamespace(cached_tokens=1800),
                   completion_tokens_details=SimpleNamespace(reasoning_tokens=64), cost=0.000285)
    details = obs.usage_details(usage)
    # Additive types: cached and reasoning tokens are split out, not counted twice.
    assert details == {"input": 300, "output": 86, "total": 2250,
                       "input_cached_tokens": 1800, "output_reasoning_tokens": 64}
    assert sum(v for k, v in details.items() if k.startswith("input")) == 2100
    assert sum(v for k, v in details.items() if k.startswith("output")) == 150


def test_genai_usage_attributes_are_inclusive():
    resp = SimpleNamespace(usage=_usage(
        prompt_tokens=2100, completion_tokens=150, total_tokens=2250,
        prompt_tokens_details=SimpleNamespace(cached_tokens=1800),
        completion_tokens_details=SimpleNamespace(reasoning_tokens=64)))
    attrs = obs.genai_attributes(operation="chat", provider="openrouter", request_model="m",
                                 response=resp)
    assert attrs["gen_ai.usage.input_tokens"] == 2100  # SHOULD include cached tokens
    assert attrs["gen_ai.usage.cache_read.input_tokens"] == 1800
    assert attrs["gen_ai.usage.output_tokens"] == 150
    assert attrs["gen_ai.usage.reasoning.output_tokens"] == 64


def test_usage_details_handles_ollama_usage_without_details_or_total():
    usage = {"prompt_tokens": 900, "completion_tokens": 40, "total_tokens": None,
             "prompt_tokens_details": None}
    assert obs.usage_details(usage) == {"input": 900, "output": 40, "total": 940}
    assert obs.usage_details(None) == {}


def test_cost_details_prefers_provider_cost_and_zeroes_local_models():
    assert obs.cost_details(_usage(cost=0.000285), "openrouter") == {"total": 0.000285}
    # OpenRouter reports 0 for free models: that's a real $0, not "unknown".
    assert obs.cost_details(_usage(cost=0), "openrouter") == {"total": 0.0}
    assert obs.cost_details(_usage(), "ollama") == {"total": 0.0}
    # No cost reported: let Langfuse infer it from its model price table.
    assert obs.cost_details(_usage(), "openai") is None


@pytest.mark.parametrize(("url", "provider"), [
    ("https://openrouter.ai/api/v1/", "openrouter"),
    ("http://localhost:11434/v1/", "ollama"),
    ("https://api.openai.com/v1/", "openai"),
    ("http://vllm.internal:8000/v1", "openai_compatible"),
    (None, "openai_compatible"),
])
def test_provider_of(url, provider):
    assert obs.provider_of(url) == provider


def test_genai_attributes_request_only_has_required_attributes():
    attrs = obs.genai_attributes(operation="chat", provider="openrouter",
                                 request_model="openai/gpt-6-luna", max_tokens=4000)
    assert attrs == {"gen_ai.operation.name": "chat", "gen_ai.provider.name": "openrouter",
                     "gen_ai.request.model": "openai/gpt-6-luna",
                     "gen_ai.request.max_tokens": 4000}


def test_genai_attributes_from_response():
    resp = SimpleNamespace(
        id="gen-123", model="openai/gpt-6-luna-2026-09",
        choices=[SimpleNamespace(finish_reason="stop"), SimpleNamespace(finish_reason=None)],
        usage=_usage(prompt_tokens=10, completion_tokens=5, total_tokens=15,
                     prompt_tokens_details=SimpleNamespace(cached_tokens=8)))
    attrs = obs.genai_attributes(operation="chat", provider="openrouter",
                                 request_model="openai/gpt-6-luna", response=resp)
    assert attrs["gen_ai.response.id"] == "gen-123"
    assert attrs["gen_ai.response.model"] == "openai/gpt-6-luna-2026-09"
    assert attrs["gen_ai.response.finish_reasons"] == ["stop"]
    assert attrs["gen_ai.usage.input_tokens"] == 10
    assert attrs["gen_ai.usage.output_tokens"] == 5
    assert attrs["gen_ai.usage.cache_read.input_tokens"] == 8
    assert "gen_ai.usage.reasoning.output_tokens" not in attrs
    # Message content is opt-in in the conventions; we never put it on the span.
    assert not any(k.startswith(("gen_ai.input", "gen_ai.output")) for k in attrs)


@pytest.mark.parametrize(("text", "expected"), [
    ("y\n", True), ("YES", True), ("👍", True), ("up", True),
    ("n", False), ("no\n", False), ("👎", False), ("-1", False),
    ("", None), ("\n", None), ("maybe", None),
])
def test_parse_feedback(text, expected):
    assert obs.parse_feedback(text) is expected


def test_record_feedback_sends_boolean_score_with_masked_comment():
    calls = []
    fake = SimpleNamespace(create_score=lambda **kw: calls.append(kw))
    obs.record_feedback("abc123", False, "wrong number, ask me at me@corp.com", langfuse=fake)
    obs.record_feedback("abc123", True, langfuse=fake)
    assert calls[0] == {"name": "user_feedback", "value": 0, "data_type": "BOOLEAN",
                        "trace_id": "abc123", "comment": "wrong number, ask me at <email>"}
    assert calls[1]["value"] == 1 and calls[1]["comment"] is None


def test_trace_url():
    assert (obs.trace_url("t1", base_url="http://localhost:3014/")
            == "http://localhost:3014/project/sqlbot/traces/t1")
