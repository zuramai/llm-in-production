import sys
from pathlib import Path

import openai
import pytest
from fastapi.testclient import TestClient

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "services" / "sqlbot"))

from ch20_sqlbot.agent import SqlAnswer
from ch20_sqlbot.app import create_app
from ch20_sqlbot.guard import UnsafeSQLError, validate_sql
from ch20_sqlbot.settings import load_settings

SETTINGS = load_settings({"LLM_BASE_URL": "http://x/v1", "LLM_API_KEY": "k", "LLM_MODEL": "m",
                          "DATABASE_URL_RO": "postgresql://ro@127.0.0.1:1/none"})


def client_with(ask):
    return TestClient(create_app(SETTINGS, client=object(), ask=ask))


def fake_answer(n_rows):
    def ask(question, *, settings, client):
        assert settings is SETTINGS
        return SqlAnswer(answer=f"{n_rows} rows", sql="SELECT 1", attempts=1, model="m",
                         latency_ms=5, usage={"cost_usd": 0.0001},
                         rows=[{"i": i} for i in range(n_rows)])
    return ask


def test_ask_returns_answer_and_caps_rows_but_reports_the_true_count():
    r = client_with(fake_answer(120)).post("/ask", json={"question": "top products?"})
    assert r.status_code == 200
    body = r.json()
    assert body["row_count"] == 120
    assert len(body["rows"]) == 50
    assert body["sql"] == "SELECT 1"


def test_provider_error_is_a_502_not_a_500():
    def ask(question, *, settings, client):
        raise openai.APIConnectionError(request=None)

    r = client_with(ask).post("/ask", json={"question": "q"})
    assert r.status_code == 502


@pytest.mark.parametrize("question", ["", "x" * 1001])
def test_question_length_limits(question):
    assert client_with(fake_answer(1)).post("/ask", json={"question": question}).status_code == 422


def test_readiness_fails_when_database_is_unreachable():
    c = client_with(fake_answer(1))
    assert c.get("/readyz").status_code == 503
    assert c.get("/healthz").status_code == 200


def test_guard_copy_still_blocks_writes_and_caps_limit():
    with pytest.raises(UnsafeSQLError, match="select_only"):
        validate_sql("DELETE FROM orders", {"orders"})
    assert validate_sql("SELECT id FROM orders LIMIT 5000", {"orders"}).endswith("LIMIT 1000")


def test_answer_survives_a_phrase_call_with_no_text(monkeypatch):
    # Providers can return content=None (refusal, length cut); that must not become a 500.
    import json
    from types import SimpleNamespace

    from ch20_sqlbot import agent

    plan = json.dumps({"assumptions": [], "sql": "SELECT 1 AS n", "needs_clarification": None,
                       "refusal": None})
    replies = iter([plan, None])

    def create(**kwargs):
        msg = SimpleNamespace(content=next(replies))
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="length")],
                               usage=None)

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(agent, "run_query", lambda sql, dsn, tz: [{"n": 1}])
    result = agent.answer("How many?", settings=SETTINGS, client=client)
    assert result.rows == [{"n": 1}]
    assert "couldn't phrase an answer (length)" in result.answer
