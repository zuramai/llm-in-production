"""Approval limits (pure), plus idempotency and role enforcement against the ShopFlow DB."""

import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ch09_refunds as refunds
import ch09_setup
import ch09_tools

from llmprod import connect

# ----------------------------------------------------------------------------- pure logic


@pytest.mark.parametrize(
    ("cents", "who"),
    [
        (1, "agent"),
        (10_000, "agent"),  # "up to $100.00" is inclusive
        (10_001, "team_lead"),  # "$100.01 to $500.00"
        (50_000, "team_lead"),
        (50_001, "finance"),
    ],
)
def test_limit_boundaries(cents, who):
    assert refunds.required_approver(cents) == who


def ev(**kw):
    base = {"status": "delivered", "total_cents": 40_000, "refunded_cents": 0, "open_cents": 0}
    return refunds.evaluate_refund(**{**base, "amount_cents": 1_000, **kw})


def test_splitting_a_refund_does_not_dodge_the_limit():
    assert ev(amount_cents=6_000) == "agent"
    # A second $60 on the same order: combined $120 needs a team lead, per the handbook.
    assert ev(amount_cents=6_000, open_cents=6_000) == "team_lead"
    assert ev(amount_cents=6_000, refunded_cents=6_000) == "team_lead"


def test_cannot_refund_more_than_the_order_total():
    with pytest.raises(refunds.RefundError, match="at most 4000 cents"):
        ev(amount_cents=5_000, open_cents=36_000)
    # exactly the order total is allowed, and a $400.00 combined refund needs a team lead
    assert ev(amount_cents=4_000, open_cents=36_000) == "team_lead"


@pytest.mark.parametrize("status", ["pending", "cancelled", "refunded"])
def test_only_paid_shipped_delivered_orders_are_refundable(status):
    with pytest.raises(refunds.RefundError, match=status):
        ev(status=status)


@pytest.mark.parametrize("amount", [0, -5])
def test_amount_must_be_positive(amount):
    with pytest.raises(refunds.RefundError, match="positive"):
        ev(amount_cents=amount)


# --------------------------------------------------------------------------------- database


def _orders():
    """One delivered order per band, plus a pending one, chosen from the seeded data."""
    q = (
        "SELECT o.id, o.total_cents FROM orders o WHERE {cond} "
        "AND NOT EXISTS (SELECT 1 FROM refunds r WHERE r.order_id = o.id) ORDER BY o.id LIMIT 1 OFFSET {skip}"
    )
    mid = "o.status = 'delivered' AND o.total_cents BETWEEN 20000 AND 40000"
    conds = {
        "small": ("o.status = 'delivered' AND o.total_cents BETWEEN 5000 AND 9000", 0),
        "mid": (mid, 0),
        "mid_b": (mid, 1),
        "big": ("o.status = 'delivered' AND o.total_cents > 80000", 0),
        "pending": ("o.status = 'pending'", 0),
    }
    out = {}
    with connect() as conn:
        for k, (c, skip) in conds.items():
            out[k] = conn.execute(q.format(cond=c, skip=skip)).fetchone()
    return out


@pytest.fixture
def db():
    ch09_setup.setup_tables()
    orders = _orders()
    ids = [o[0] for o in orders.values()]

    def clean():
        with connect() as conn:
            conn.execute("DELETE FROM ch09_pending_refunds WHERE order_id = ANY(%s)", (ids,))

    clean()
    yield {k: {"id": v[0], "total": v[1]} for k, v in orders.items()}
    clean()


def count_rows(order_id):
    with connect() as conn:
        return conn.execute(
            "SELECT count(*) FROM ch09_pending_refunds WHERE order_id = %s", (order_id,)
        ).fetchone()[0]


@pytest.mark.docker
def test_same_key_replays_the_stored_result_and_creates_one_row(db):
    oid = db["small"]["id"]
    first = refunds.request_refund(oid, 1_500, "damaged", "t-key-1")
    second = refunds.request_refund(oid, 1_500, "damaged", "t-key-1")
    assert first["idempotent_replay"] is False and second["idempotent_replay"] is True
    assert second["refund_request_id"] == first["refund_request_id"]
    assert count_rows(oid) == 1


@pytest.mark.docker
def test_same_key_with_different_parameters_is_rejected_not_replayed(db):
    oid = db["small"]["id"]
    refunds.request_refund(oid, 1_500, "damaged", "t-key-2")
    with pytest.raises(refunds.RefundError, match="different parameters"):
        refunds.request_refund(oid, 2_500, "damaged", "t-key-2")
    assert count_rows(oid) == 1


@pytest.mark.docker
def test_fresh_key_for_the_same_order_and_reason_does_not_create_a_second_refund(db):
    """The model "retries" with a brand-new key: the handbook's order+reason rule catches it."""
    oid = db["small"]["id"]
    first = refunds.request_refund(oid, 1_500, "late", "t-key-3a")
    again = refunds.request_refund(oid, 1_500, "late", "t-key-3b")
    assert again["idempotent_replay"] is True
    assert again["refund_request_id"] == first["refund_request_id"]
    assert count_rows(oid) == 1


@pytest.mark.docker
def test_concurrent_retries_with_one_key_create_exactly_one_row(db):
    oid = db["small"]["id"]
    results, errors = [], []

    def worker():
        try:
            results.append(refunds.request_refund(oid, 1_000, "defective", "t-key-4"))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert len({r["refund_request_id"] for r in results}) == 1
    assert sum(not r["idempotent_replay"] for r in results) == 1
    assert count_rows(oid) == 1


@pytest.mark.docker
def test_limits_route_to_the_right_approver_and_cap_the_total(db):
    small, mid, mid_b, big = db["small"], db["mid"], db["mid_b"], db["big"]
    assert (
        refunds.request_refund(mid["id"], 10_000, "damaged", "t-a")["required_approver"] == "agent"
    )
    got = refunds.request_refund(mid_b["id"], 10_001, "damaged", "t-b")
    assert got["required_approver"] == "team_lead"
    assert (
        refunds.request_refund(big["id"], 50_001, "damaged", "t-c")["required_approver"]
        == "finance"
    )
    # the cap counts money already queued: after asking for all but $10, $20 more is too much
    refunds.request_refund(small["id"], small["total"] - 1_000, "damaged", "t-d")
    with pytest.raises(refunds.RefundError, match="exceed the order total"):
        refunds.request_refund(small["id"], 2_000, "other", "t-e")


@pytest.mark.docker
def test_unknown_and_unrefundable_orders_are_tool_errors_not_rows(db):
    box = ch09_tools.LocalToolbox()
    missing = box.call(
        "create_refund",
        '{"order_id": 99999999, "amount_cents": 500, "reason": "late", "idempotency_key": "t-e"}',
    )
    assert "order 99999999 not found" in missing
    pend = db["pending"]["id"]
    out = box.call(
        "create_refund",
        f'{{"order_id": {pend}, "amount_cents": 500, "reason": "late", "idempotency_key": "t-f"}}',
    )
    assert "only paid, shipped or delivered" in out
    assert count_rows(pend) == 0


@pytest.mark.docker
def test_role_enforcement_on_approval(db):
    mid = db["mid"]["id"]
    req = refunds.request_refund(mid, 25_000, "damaged", "t-g")  # needs team_lead
    rid = req["refund_request_id"]
    with pytest.raises(refunds.RefundError, match="requires 'team_lead' or higher"):
        refunds.decide(rid, "agent", approve=True)
    assert refunds.list_requests("pending")  # still pending after the refused attempt
    done = refunds.decide(rid, "team_lead", approve=True, note="ok")
    assert done["status"] == "approved" and done["decided_by_role"] == "team_lead"
    with pytest.raises(refunds.RefundError, match="already approved"):
        refunds.decide(rid, "finance", approve=True)


@pytest.mark.docker
def test_anyone_may_reject_and_a_rejected_reason_can_be_requested_again(db):
    oid = db["small"]["id"]
    rid = refunds.request_refund(oid, 3_000, "late", "t-h1")["refund_request_id"]
    done = refunds.decide(rid, "agent", approve=False, note="outside window")
    assert done["status"] == "rejected"
    again = refunds.request_refund(oid, 3_000, "late", "t-h2")
    assert again["idempotent_replay"] is False
    # but replaying the OLD key returns the old (rejected) outcome rather than a new refund
    old = refunds.request_refund(oid, 3_000, "late", "t-h1")
    assert old["status"] == "rejected" and old["idempotent_replay"] is True
