"""The target model is routed by name (Ollama tag vs OpenRouter id); the judge never moves."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ch13_harness as H

HOSTED = SimpleNamespace(name="openrouter-client")


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(H, "chat_client", lambda *a, **k: HOSTED)
    monkeypatch.setattr(H, "get_settings", lambda: SimpleNamespace(llm_model="openai/gpt-6-luna"))
    monkeypatch.delenv("OLLAMA_CHAT_BASE_URL", raising=False)


@pytest.mark.parametrize("model,local", [
    ("qwen3:8b", True), ("llama-guard3:1b", True), ("qwen3", True),
    ("openai/gpt-6-luna", False), ("qwen/qwen3.8-flash", False), (None, False), ("", False),
])
def test_is_local_model(model, local):
    assert H.is_local_model(model) is local


def test_local_tag_goes_to_ollama_and_hosted_id_to_openrouter(monkeypatch):
    client = H.target_client("qwen3:8b")
    assert client is not HOSTED
    assert str(client.base_url).startswith("http://localhost:11434/v1")
    monkeypatch.setenv("OLLAMA_CHAT_BASE_URL", "http://gpu-box:11434/v1")
    assert str(H.target_client("qwen3:8b").base_url).startswith("http://gpu-box:11434/v1")
    assert H.target_client("openai/gpt-6-luna") is HOSTED
    assert H.target_client(None) is HOSTED


def _capture_targets(monkeypatch):
    calls = {}

    def fake_sqlbot(prompt, **kw):
        calls["sqlbot"] = kw
        return SimpleNamespace(answer="42", sql="SELECT 42", usage={})

    def fake_agent(prompt, toolbox, **kw):
        calls["agent"] = kw
        return SimpleNamespace(answer="ok", cost_usd=0.0, tool_calls=[], stop_reason="final")

    monkeypatch.setattr(H, "sqlbot_answer", fake_sqlbot)
    monkeypatch.setattr(H.ch13_agent, "run_agent", fake_agent)
    monkeypatch.setattr(H.ch13_agent, "system_for", lambda role: "system")
    return calls


@pytest.mark.parametrize("model", ["qwen3:8b", "openai/gpt-6-luna"])
def test_both_targets_use_the_routed_client(monkeypatch, model):
    calls = _capture_targets(monkeypatch)
    local = H.is_local_model(model)
    for target in ("sqlbot", "agent"):
        out = H.run_case({"id": "B1", "target": target, "category": "benign", "prompt": "hi"},
                         H.NONE, {}, model)
        assert out.error is None
    sql, agent = calls["sqlbot"], calls["agent"]
    assert sql["model"] == agent["model"] == model
    assert (sql["client"] is HOSTED) is (not local)
    assert (agent["client"] is HOSTED) is (not local)
    # qwen3 must not think on Ollama; hosted sqlbot rows keep the provider default.
    assert sql["reasoning_effort"] == ("none" if local else None)
    assert agent["extra"] == {"reasoning_effort": "none"}


def test_judge_stays_hosted_for_local_target(monkeypatch):
    judged = []
    monkeypatch.setattr(H.ch13_setup, "reset_refunds", lambda: None)
    monkeypatch.setattr(H, "run_case", lambda c, d, ctx, model: H.Outcome(
        c["id"], c["target"], c["category"], output="Here you go."))

    def fake_judge(prompt, output, model=None):
        judged.append(model)
        return {"refused": False, "reason": "answered", "cost_usd": 0.001}

    monkeypatch.setattr(H, "judge_refusal", fake_judge)
    rows = H.run_suite([{"id": "B1", "target": "agent", "category": "benign", "prompt": "hi"}],
                       H.NONE, {}, model="qwen3:8b", reps=2, workers=1, judge=True)
    assert judged == [None, None]  # None = the judge's own default (hosted LLM_MODEL)
    assert [r["model"] for r in rows] == ["qwen3:8b", "qwen3:8b"]
    assert all(r["refused"] is False for r in rows)
