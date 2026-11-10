"""The graph's control flow with a scripted fake model and fake tools: no network, no database."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ch05_agentic import graph as G
from ch05_agentic.tools import DocEvidence, SqlEvidence
from langgraph.checkpoint.memory import InMemorySaver

TODAY = "Saturday 2026-11-07 09:00 (America/New_York)"


class FakeLLM:
    """Returns scripted replies per step ('route', 'grade', 'rewrite', 'answer') and records
    every call, so tests can assert what ran."""

    model = "fake"

    def __init__(self, script: dict):
        self.script = {k: list(v) for k, v in script.items()}
        self.calls: list[str] = []

    def _next(self, step: str):
        self.calls.append(step)
        replies = self.script[step]
        reply = replies.pop(0) if len(replies) > 1 else replies[0]
        if isinstance(reply, Exception):
            raise reply
        return reply

    def structured(self, messages, schema, usage, step, **kw):
        return schema.model_validate(self._next(step))

    def text(self, messages, usage, step, **kw):
        return self._next(step)


class FakeDocs:
    def __init__(self):
        self.queries: list[str] = []

    def search(self, query, k=None):
        self.queries.append(query)
        n = len(self.queries)
        return [DocEvidence(f"[doc:policies/returns-policy.md#{n}]", "policies/returns-policy.md",
                            "Returns", "Return within 30 days of delivery.", query)]


@pytest.fixture
def sql_calls(monkeypatch):
    calls: list[str] = []

    def fake_ask_sql(question, llm, usage, citation, today, max_attempts=2):
        calls.append(citation)
        return SqlEvidence(citation, question, "SELECT 1", [{"delivered": "2026-10-20"}])

    monkeypatch.setattr(G, "ask_sql", fake_ask_sql)
    return calls


ROUTE_BOTH = {"doc_query": "return window", "sql_question": "order 1 delivery date"}
OK = {"relevant": ["[doc:policies/returns-policy.md#1]", "[sql:1]"], "sufficient": True,
      "missing": ""}
NOT_OK = {"relevant": [], "sufficient": False, "missing": "the delivery date"}
ANSWER = "Yes, until 2026-11-19 [sql:1] [doc:policies/returns-policy.md#1]."


def run(llm, docs, **kw):
    app = G.build_graph(llm, docs, **kw)
    return app.invoke(G.initial_state("Can customer 1 return order 1?", TODAY),
                      {"recursion_limit": G.RECURSION_LIMIT,
                       "configurable": {"thread_id": "t"}})


def test_route_fans_out_to_both_tools_then_answers(sql_calls):
    llm, docs = FakeLLM({"route": [ROUTE_BOTH], "grade": [OK], "answer": [ANSWER]}), FakeDocs()
    state = run(llm, docs)
    assert llm.calls == ["route", "grade", "answer"]  # grade ran ONCE after both tools
    assert docs.queries == ["return window"] and sql_calls == ["[sql:1]"]
    assert {e["kind"] for e in state["evidence"]} == {"doc", "sql"}
    assert state["citations"]["invalid"] == []
    assert state["citations"]["cites_doc"] and state["citations"]["cites_sql"]


def test_rewrite_loop_is_capped_at_max_loops(sql_calls):
    rewrite = {"doc_query": "other words", "sql_question": "order 1 shipments"}
    llm = FakeLLM({"route": [ROUTE_BOTH], "grade": [NOT_OK], "rewrite": [rewrite],
                   "answer": ["I can't tell: the delivery date is missing."]})
    state = run(llm, FakeDocs())
    assert llm.calls.count("rewrite") == G.MAX_LOOPS
    assert llm.calls.count("grade") == G.MAX_LOOPS + 1
    assert llm.calls[-1] == "answer"
    assert state["loops"] == G.MAX_LOOPS
    # SQL citations keep counting across loops instead of reusing [sql:1]
    assert sql_calls == ["[sql:1]", "[sql:2]", "[sql:3]"]


def test_rewrite_can_target_one_tool(sql_calls):
    llm = FakeLLM({"route": [ROUTE_BOTH], "grade": [NOT_OK, OK],
                   "rewrite": [{"doc_query": None, "sql_question": "order 1 shipments"}],
                   "answer": [ANSWER]})
    docs = FakeDocs()
    run(llm, docs)
    assert docs.queries == ["return window"]  # docs not searched again
    assert sql_calls == ["[sql:1]", "[sql:2]"]


def test_no_tools_needed_goes_straight_to_answer(sql_calls):
    llm = FakeLLM({"route": [{"doc_query": None, "sql_question": None}],
                   "answer": ["Hello! What do you need?"]})
    state = run(llm, FakeDocs())
    assert llm.calls == ["route", "answer"]
    assert state.get("evidence", []) == []


def test_invented_citation_is_flagged(sql_calls):
    llm = FakeLLM({"route": [ROUTE_BOTH], "grade": [OK],
                   "answer": ["Yes [doc:policies/vip-program.md#3] [sql:9]."]})
    state = run(llm, FakeDocs())
    assert state["citations"]["invalid"] == ["[doc:policies/vip-program.md#3]", "[sql:9]"]
    assert not state["citations"]["cites_doc"]


def test_grader_cannot_mark_unknown_evidence_relevant(sql_calls):
    bogus = {**OK, "relevant": ["[sql:1]", "[doc:made/up.md#0]"]}
    llm = FakeLLM({"route": [ROUTE_BOTH], "grade": [bogus], "answer": [ANSWER]})
    state = run(llm, FakeDocs())
    assert state["relevant"] == ["[sql:1]"]


def test_checkpoint_resume_skips_finished_nodes(sql_calls):
    """The answer node fails once (e.g. a provider timeout). Resuming the same thread from its
    checkpoint re-runs only `answer`: no second route, retrieval or SQL call."""
    llm = FakeLLM({"route": [ROUTE_BOTH], "grade": [OK],
                   "answer": [TimeoutError("provider timeout"), ANSWER]})
    docs = FakeDocs()
    app = G.build_graph(llm, docs, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "q1"}}
    with pytest.raises(TimeoutError):
        app.invoke(G.initial_state("Can customer 1 return order 1?", TODAY), config)
    state = app.invoke(None, config)
    assert state["answer"] == ANSWER
    assert llm.calls == ["route", "grade", "answer", "answer"]
    assert docs.queries == ["return window"] and sql_calls == ["[sql:1]"]
