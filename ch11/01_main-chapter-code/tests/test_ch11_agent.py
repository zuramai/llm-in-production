import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ch11_agent import agent

REPORT = {"summary": "s", "impact": "i", "timeline": ["13:00 deploy"],
          "hypotheses": [{"hypothesis": "DB pool exhausted", "evidence": ["in_use=10"],
                          "confidence": 0.9}],
          "next_actions": ["Roll back"]}


class Msg:
    def __init__(self, content=None, tool_calls=None):
        self.content, self.tool_calls = content, tool_calls

    def model_dump(self, exclude_none=True):
        return {"role": "assistant", "content": self.content,
                "tool_calls": [{"id": c.id, "type": "function",
                                "function": {"name": c.function.name,
                                             "arguments": c.function.arguments}}
                               for c in self.tool_calls or []]}


def call(i, name, args):
    return SimpleNamespace(id=f"c{i}", function=SimpleNamespace(name=name, arguments=args))


def resp(msg, prompt=100):
    usage = SimpleNamespace(prompt_tokens=prompt, completion_tokens=10, cost=0.001)
    return SimpleNamespace(usage=usage, choices=[SimpleNamespace(message=msg)])


class FakeClient:
    """Plays back scripted responses and records every request."""

    def __init__(self, script):
        self.script, self.requests = list(script), []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.requests.append({**kw, "messages": list(kw["messages"])})
        return self.script.pop(0)


def run(monkeypatch, script, **kw):
    fake = FakeClient(script)
    monkeypatch.setattr(agent, "client_for", lambda model: fake)
    return agent.run_agent("5xx high", "openai/test", **kw), fake


def test_tool_results_go_back_to_the_model_and_the_report_is_parsed(monkeypatch):
    res, fake = run(monkeypatch, [
        resp(Msg(tool_calls=[call(1, "search_runbooks", '{"query": "db pool exhausted"}'),
                             call(2, "rm_rf", "{}")])),
        resp(Msg(content="done")),
        resp(Msg(content="Here you go: " + json.dumps(REPORT)), prompt=300),
    ])
    assert res.error is None and res.report.hypotheses[0].hypothesis == "DB pool exhausted"
    assert res.steps == 1 and [t["tool"] for t in res.trace] == ["search_runbooks", "rm_rf"]
    tool_msgs = [m for m in fake.requests[1]["messages"] if m["role"] == "tool"]
    assert "checkout-db-pool-exhaustion.md" in tool_msgs[0]["content"]
    assert tool_msgs[1]["content"].startswith("ERROR: unknown tool")
    assert res.prompt_tokens == 500 and res.max_prompt_tokens == 300
    assert abs(res.cost_usd - 0.003) < 1e-9


def test_step_cap_still_produces_a_report_call_without_tools(monkeypatch):
    looping = [resp(Msg(tool_calls=[call(i, "get_deploys", '{"since": "1h"}')])) for i in range(3)]
    res, fake = run(monkeypatch, [*looping, resp(Msg(content=json.dumps(REPORT)))], max_steps=3)
    assert res.steps == 3 and res.report is not None
    final = fake.requests[-1]
    assert "tools" not in final and final["response_format"]["json_schema"]["strict"] is True
    assert final["messages"][-1]["role"] == "user"


def test_bad_report_is_recorded_as_an_error_not_raised(monkeypatch):
    res, _ = run(monkeypatch, [resp(Msg(content="no tools needed")),
                               resp(Msg(content='{"summary": "missing fields"}'))])
    assert res.report is None and res.error.startswith("ValidationError")


def test_routing_and_reasoning_effort_per_model():
    assert agent.client_for("gpt-oss:20b").base_url.host == "localhost"
    assert agent.reasoning_for("gpt-oss:20b") == "low"
    assert agent.reasoning_for("openai/gpt-6-luna") == "none"
