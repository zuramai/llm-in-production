"""The real ledger in Postgres: idempotent writes and re-checks at execution time."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ch10_setup
from ch10_scenarios import pick_context
from ch10_tools import RefundError, execute_refund, ledger_rows, validate_refund

pytestmark = pytest.mark.docker


@pytest.fixture
def order():
    ch10_setup.setup()
    o = pick_context()["small3"]  # delivered, $60-90, no historical refunds
    ch10_setup.clear_orders([o["id"]])
    yield o
    ch10_setup.clear_orders([o["id"]])


def test_execute_twice_writes_once(order):
    p = validate_refund(order["id"], 1000, "late")
    first = execute_refund(p, thread_id="t", tool_call_id="c1", role="agent")
    again = execute_refund(p, thread_id="t", tool_call_id="c1", role="agent")
    assert first["status"] == "issued"
    assert again["status"] == "already_issued" and again["idempotent_replay"]
    assert len(ledger_rows([order["id"]])) == 1


def test_a_second_proposal_for_the_same_order_and_reason_is_refused(order):
    p = validate_refund(order["id"], 1000, "late")
    execute_refund(p, thread_id="t", tool_call_id="c1", role="agent")
    with pytest.raises(RefundError, match="already issued"):
        validate_refund(order["id"], 1000, "late")
    assert validate_refund(order["id"], 1000, "damaged")["required_approver"] == "agent"


def test_limits_are_rechecked_when_the_approval_arrives(order):
    # Two proposals validated before either executes: together they exceed the order total.
    total = order["total_cents"]
    a = validate_refund(order["id"], total - 100, "damaged")
    b = validate_refund(order["id"], 500, "late")
    execute_refund(a, thread_id="t", tool_call_id="a", role="agent")
    with pytest.raises(RefundError, match="exceed the order total"):
        execute_refund(b, thread_id="t", tool_call_id="b", role="agent")
    assert [r["reason"] for r in ledger_rows([order["id"]])] == ["damaged"]
