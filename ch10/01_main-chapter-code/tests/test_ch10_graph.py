"""The approval graph with a scripted fake model, an in-memory checkpointer and a fake ledger:
no network, no database. These pin the safety properties the chapter relies on."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ch10_graph
from ch10_graph import AgentConfig, build_graph, run
from ch10_tools import RefundError, idempotency_key, prev
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import ValidationError


def _resp(content="", calls=()):
    tool_calls = [
        SimpleNamespace(id=f"call_{i}_{name}", function=SimpleNamespace(name=name, arguments=args))
        for i, (name, args) in enumerate(calls)
    ]
    msg = SimpleNamespace(content=content, tool_calls=tool_calls or None)
    usage = SimpleNamespace(prompt_tokens=100, completion_tokens=10, model_extra={"cost": 0.001})
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=usage)


class FakeLLM:
    """Replays scripted responses; records every request so tests can inspect prompts."""

    def __init__(self, script):
        self.script = list(script)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        if not self.script:
            return _resp("done")
        return self.script.pop(0)


def refund_call(order_id=1, amount=2500, reason="damaged"):
    args = {"order_id": order_id, "amount_cents": amount, "reason": reason}
    return _resp(calls=[("issue_refund", json.dumps(args))])


@pytest.fixture
def ledger(monkeypatch):
    """Fake DB: orders 1..9 delivered, $1,000 total each; order 5 is pending."""
    rows = []

    def get_order(order_id: int):
        if order_id > 9:
            raise prev.ToolError(f"order {order_id} not found")
        return {"id": order_id, "status": "delivered", "total_cents": 100_000}

    def validate(order_id, amount_cents, reason):
        if order_id == 5:
            raise RefundError("order status is 'pending'")
        if any(r["key"] == idempotency_key(order_id, reason) for r in rows):
            raise RefundError("already issued")
        approver = prev.evaluate_refund(status="delivered", total_cents=100_000,
                                        refunded_cents=0, open_cents=0, amount_cents=amount_cents)
        return {"order_id": order_id, "amount_cents": amount_cents, "reason": reason,
                "amount_usd": prev.usd(amount_cents), "required_approver": approver,
                "idempotency_key": idempotency_key(order_id, reason)}  # fmt: skip

    def execute(proposal, *, thread_id, tool_call_id, role, note=""):
        if any(r["key"] == proposal["idempotency_key"] for r in rows):
            return {"status": "already_issued", "idempotent_replay": True}
        rows.append({"key": proposal["idempotency_key"], "amount": proposal["amount_cents"],
                     "role": role, "thread": thread_id})  # fmt: skip
        return {"status": "issued", "approved_by": role}

    get_order.__doc__ = "Look up one order."
    get_order.__annotations__ = {"order_id": int}
    monkeypatch.setattr(ch10_graph, "READ_TOOLS", {"get_order": get_order})
    monkeypatch.setattr(ch10_graph, "validate_refund", validate)
    monkeypatch.setattr(ch10_graph, "execute_refund", execute)
    return rows


def make(llm, saver=None, **kw):
    return build_graph(AgentConfig(model="fake", client=llm, **kw), saver or InMemorySaver())


def last_tool_result(turn):
    tool_msgs = [m for m in turn.state["messages"] if m["role"] == "tool"]
    return json.loads(tool_msgs[-1]["content"])


def test_refund_pauses_and_nothing_is_written_before_approval(ledger):
    g = make(FakeLLM([refund_call(), _resp("Refund issued.")]))
    turn = run(g, "t1", user="refund $25 on order 1")
    assert turn.answer is None
    assert turn.pending[0]["amount_cents"] == 2500
    assert turn.pending[0]["required_approver"] == "agent"
    assert ledger == []  # the side effect has not happened
    turn = run(g, "t1", decision={"approved": True, "role": "agent"})
    assert [r["role"] for r in ledger] == ["agent"]
    assert last_tool_result(turn)["status"] == "issued"
    assert turn.answer == "Refund issued."


def test_rejection_writes_nothing_and_tells_the_model(ledger):
    g = make(FakeLLM([refund_call(), _resp("It was rejected.")]))
    run(g, "t2", user="refund")
    turn = run(g, "t2", decision={"approved": False, "role": "team_lead", "note": "no receipt"})
    assert ledger == []
    body = last_tool_result(turn)
    assert body["status"] == "rejected" and body["note"] == "no receipt"


def test_insufficient_role_asks_again_then_executes(ledger):
    g = make(FakeLLM([refund_call(amount=25_000), _resp("ok")]))
    first = run(g, "t3", user="refund $250")
    assert first.pending[0]["required_approver"] == "team_lead"
    second = run(g, "t3", decision={"approved": True, "role": "agent"})
    assert ledger == []
    assert second.pending[0]["attempt"] == 2
    assert "needs 'team_lead'" in second.pending[0]["previous"]
    run(g, "t3", decision={"approved": True, "role": "team_lead"})
    assert [r["role"] for r in ledger] == ["team_lead"]


def test_insufficient_role_twice_is_not_executed(ledger):
    g = make(FakeLLM([refund_call(amount=60_000), _resp("needs finance")]))
    run(g, "t4", user="refund $600")
    run(g, "t4", decision={"approved": True, "role": "agent"})
    turn = run(g, "t4", decision={"approved": True, "role": "team_lead"})
    assert ledger == []
    assert last_tool_result(turn)["status"] == "not_executed"
    assert turn.pending == []


def test_invalid_proposal_never_reaches_a_human(ledger):
    g = make(FakeLLM([refund_call(order_id=5), _resp("pending order")]))
    turn = run(g, "t5", user="refund order 5")
    assert turn.pending == [] and ledger == []
    assert "not executed" in last_tool_result(turn)["error"]


def test_invalid_arguments_never_reach_a_human(ledger):
    bad = _resp(calls=[("issue_refund", json.dumps({"order_id": 1, "amount_cents": -5,
                                                     "reason": "damaged"}))])  # fmt: skip
    g = make(FakeLLM([bad, _resp("bad amount")]))
    turn = run(g, "t5b", user="refund")
    assert turn.pending == [] and ledger == []
    assert "invalid arguments" in last_tool_result(turn)["error"]


def test_ungated_v0_writes_without_anyone_approving(ledger):
    g = make(FakeLLM([refund_call(amount=90_000), _resp("done")]), gate=False)
    turn = run(g, "t6", user="refund $900")
    assert turn.pending == []
    assert ledger == [{"key": "refund-1-damaged", "amount": 90_000, "role": None, "thread": "t6"}]


def test_resume_from_a_new_graph_object_with_the_same_checkpointer(ledger):
    saver = InMemorySaver()
    run(make(FakeLLM([refund_call()]), saver), "t7", user="refund")
    # "Restart": a new graph, a new client; only the checkpointer is shared.
    turn = run(make(FakeLLM([_resp("issued")]), saver), "t7",
               decision={"approved": True, "role": "agent"})  # fmt: skip
    assert len(ledger) == 1 and turn.answer == "issued"


def test_bad_resume_payload_fails_and_the_approval_stays_pending(ledger):
    g = make(FakeLLM([refund_call(), _resp("ok")]))
    run(g, "t8", user="refund")
    with pytest.raises(ValidationError):
        run(g, "t8", decision={"approved": "maybe", "role": "intern"})
    assert ledger == []
    assert g.get_state({"configurable": {"thread_id": "t8"}}).interrupts
    run(g, "t8", decision={"approved": True, "role": "agent"})
    assert len(ledger) == 1


def test_two_refunds_in_one_turn_are_approved_one_at_a_time(ledger):
    both = _resp(calls=[
        ("issue_refund", json.dumps({"order_id": 1, "amount_cents": 1000, "reason": "late"})),
        ("issue_refund", json.dumps({"order_id": 2, "amount_cents": 2000, "reason": "late"})),
    ])  # fmt: skip
    g = make(FakeLLM([both, _resp("one of two")]))
    t = run(g, "t9", user="refund both")
    assert t.pending[0]["order_id"] == 1
    t = run(g, "t9", decision={"approved": True, "role": "agent"})
    assert t.pending[0]["order_id"] == 2 and len(ledger) == 1
    t = run(g, "t9", decision={"approved": False, "role": "agent"})
    assert len(ledger) == 1 and t.answer == "one of two"
    # Every tool_call_id got exactly one tool message, as Chat Completions requires.
    ids = [m["tool_call_id"] for m in t.state["messages"] if m["role"] == "tool"]
    assert sorted(ids) == ["call_0_issue_refund", "call_1_issue_refund"]


def test_step_limit_stops_a_looping_model(ledger):
    loop = [_resp(calls=[("get_order", '{"order_id": 1}')]) for _ in range(10)]
    llm = FakeLLM(loop)
    turn = run(make(llm, max_llm_calls=3), "t10", user="loop")
    assert len(llm.requests) == 3
    assert turn.answer == "Stopped: step limit reached."


def test_bookkeeping_never_reaches_the_provider(ledger):
    llm = FakeLLM([_resp(calls=[("get_order", '{"order_id": 1}')]), _resp("ok")])
    run(make(llm), "t11", user="look up 1")
    second = llm.requests[1]["messages"]
    assert second[0]["role"] == "system"
    assert all(not k.startswith("_") for m in second for k in m)
    assert second[1]["role"] == "user"


def test_run_requires_exactly_one_input(ledger):
    g = make(FakeLLM([]))
    with pytest.raises(ValueError):
        run(g, "t12")
