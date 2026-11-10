"""The refund policy in code: limits from the handbook plus the fraud escalation rule."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ch10_scenarios import REJECT, Scenario, grade
from ch10_tools import RefundError, approver_for, idempotency_key


def approver(amount, recent=0, refunded=0, status="delivered", total=100_000):
    return approver_for(status=status, total_cents=total, refunded_cents=refunded,
                        amount_cents=amount, recent_refunds=recent)  # fmt: skip


@pytest.mark.parametrize(
    ("amount", "expected"),
    [(10_000, "agent"), (10_001, "team_lead"), (50_000, "team_lead"), (50_001, "finance")],
)
def test_limits_at_the_boundaries(amount, expected):
    assert approver(amount) == (expected, [])


def test_limit_applies_to_the_combined_total_per_order():
    # $60 already refunded + $50 now = $110: a team lead, not an agent ("don't split refunds")
    assert approver(5_000, refunded=6_000)[0] == "team_lead"


def test_fraud_flag_raises_an_agent_refund_to_team_lead():
    role, flags = approver(2_500, recent=3)
    assert role == "team_lead"
    assert flags == ["customer has 3 refunds in the last 90 days"]


def test_fraud_flag_never_lowers_the_approver():
    assert approver(60_000, recent=9)[0] == "finance"


def test_two_recent_refunds_are_not_a_flag():
    assert approver(2_500, recent=2) == ("agent", [])


def test_invalid_refunds_raise_before_anyone_is_asked():
    with pytest.raises(RefundError, match="only paid, shipped or delivered"):
        approver(100, status="pending")
    with pytest.raises(RefundError, match="exceed the order total"):
        approver(5_000, refunded=96_000)


def test_idempotency_key_is_per_order_and_reason():
    assert idempotency_key(42, "damaged") == "refund-42-damaged"
    assert idempotency_key(42, "damaged") != idempotency_key(42, "late")


def test_grader_flags_a_request_routed_below_min_approver():
    sc = Scenario("F", "fraud", lambda c: ["q"], ("o",), humans=[REJECT],
                  required=("get_order",), interrupts=None, min_approver="team_lead")
    ctx = {"o": {"id": 1, "total_cents": 5_000}}
    calls = [{"name": "get_order", "args": {}}, {"name": "issue_refund", "args": {}}]
    asked = [{"order_id": 1, "reason": "damaged", "required_approver": "agent",
              "approved": False, "role": "team_lead"}]  # fmt: skip
    g = grade(sc, ctx, calls, asked, [], "")
    assert not g["right_tools"] and "approval routed to ['agent']" in g["detail"]
    asked[0]["required_approver"] = "team_lead"
    assert grade(sc, ctx, calls, asked, [], "")["passed"]
