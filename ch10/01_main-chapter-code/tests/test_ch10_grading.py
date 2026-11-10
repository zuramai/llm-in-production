"""The trajectory grader is the chapter's yardstick, so it gets its own tests."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ch10_scenarios import SCENARIOS, Expect, Scenario, grade, summarize

CTX = {"o": {"id": 10, "total_cents": 30_000}}
SC = Scenario(
    "X", "refund $250", lambda c: ["refund"], ("o",),
    required=("get_order", "issue_refund"), interrupts=1,
    ledger=[Expect("o", 25_000, "damaged")],
)  # fmt: skip
CALLS = [{"name": "get_order", "args": {}}, {"name": "issue_refund", "args": {}}]


def asked(role="team_lead", approved=True):
    return [{"order_id": 10, "reason": "damaged", "required_approver": "team_lead",
             "approved": approved, "role": role}]  # fmt: skip


def row(role="team_lead", amount=25_000):
    return [{"order_id": 10, "amount_cents": amount, "reason": "damaged",
             "approved_by_role": role}]  # fmt: skip


def test_happy_path_passes():
    g = grade(SC, CTX, CALLS, asked(), row(), "done")
    assert g["passed"] and g["detail"] == "ok"


def test_row_without_any_approval_is_unapproved():
    g = grade(SC, CTX, CALLS, [], row(role=None), "done")
    assert not g["no_unapproved"] and not g["right_tools"]  # 0 approval requests, expected 1
    assert "UNAPPROVED" in g["detail"]


def test_row_approved_by_too_junior_a_role_is_unapproved():
    g = grade(SC, CTX, CALLS, asked(role="agent"), row(role="agent"), "done")
    assert not g["no_unapproved"] and not g["passed"]


def test_rejected_request_with_a_row_anyway_is_unapproved():
    g = grade(SC, CTX, CALLS, asked(approved=False), row(), "done")
    assert not g["no_unapproved"]


def test_wrong_amount_fails_final_state_only():
    g = grade(SC, CTX, CALLS, asked(), row(amount=30_000), "done")
    assert g["right_tools"] and g["no_unapproved"] and not g["final_state"]


def test_forbidden_tool_and_missing_tool():
    sc = Scenario("Y", "lookup", lambda c: ["q"], (), required=("get_order",),
                  forbidden=("issue_refund",))  # fmt: skip
    g = grade(sc, CTX, [{"name": "issue_refund", "args": {}}], [], [], "")
    assert not g["right_tools"]
    assert "missing ['get_order']" in g["detail"] and "forbidden" in g["detail"]


def test_preseeded_rows_are_neither_unapproved_nor_new():
    sc = Scenario("Z", "dup", lambda c: ["q"], ("o",), required=("get_order",),
                  preseed=[Expect("o", 2500, "damaged")], ledger=[Expect("o", 2500, "damaged")])
    g = grade(sc, CTX, [{"name": "get_order", "args": {}}], [], row(role="agent", amount=2500), "")
    assert g["passed"]


def test_summary_pass_hat_k_is_stricter_than_pass_rate():
    def r(i, ok):
        return {"id": i, "passed": ok, "right_tools": ok, "no_unapproved": True,
                "final_state": ok, "latency_ms": 1000, "cost_usd": 0.001,
                "prompt_tokens": 10, "max_prompt_tokens": 10}  # fmt: skip

    s = summarize([r("A", True), r("A", False), r("B", True), r("B", True)])
    assert s["k"] == 2 and s["pass_rate"] == 0.75 and s["pass_all_k"] == 0.5


def test_suite_has_twenty_unique_scenarios_with_sane_scripts():
    assert len(SCENARIOS) == 20
    assert len({s.id for s in SCENARIOS}) == 20
    for s in SCENARIOS:
        assert set(s.required).isdisjoint(s.forbidden), s.id
        if s.interrupts:
            assert len(s.humans) >= s.interrupts, s.id
        for e in s.ledger + s.preseed:
            assert e.key in s.order_keys, s.id
