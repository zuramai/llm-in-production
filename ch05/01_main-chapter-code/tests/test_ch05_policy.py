"""The gold answers of the eval come from policy.py, so its boundaries must be right."""

import sys
from datetime import UTC, date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ch05_agentic import policy as P

TZ = "America/New_York"
ET = ZoneInfo(TZ)


def test_local_date_uses_business_time_zone_not_utc():
    # 02:30 UTC on Oct 21 is still Oct 20 in New York.
    ts = datetime(2026, 10, 21, 2, 30, tzinfo=UTC)
    assert P.local_date(ts, TZ) == date(2026, 10, 20)
    assert P.window_end(ts, 30, TZ) == date(2026, 11, 19)


def test_naive_timestamps_are_rejected():
    with pytest.raises(ValueError):
        P.local_date(datetime(2026, 10, 20, 12, 0), TZ)


def test_return_window_last_day_inclusive():
    delivered = datetime(2026, 10, 20, 15, 0, tzinfo=ET)
    assert P.is_open(delivered, 30, datetime(2026, 11, 19, 23, 59, tzinfo=ET), TZ)
    assert not P.is_open(delivered, 30, datetime(2026, 11, 20, 0, 1, tzinfo=ET), TZ)


def test_window_across_dst_change_counts_calendar_days():
    # DST ended on 2026-11-01. A 14-day window is 14 calendar days, not 14*24 hours.
    delivered = datetime(2026, 10, 25, 23, 30, tzinfo=ET)
    assert P.window_end(delivered, 14, TZ) == date(2026, 11, 8)


@pytest.mark.parametrize(("cents", "who"), [
    (1, "support agent"), (10_000, "support agent"), (10_001, "team lead"),
    (50_000, "team lead"), (50_001, "finance"),
])
def test_refund_approver_boundaries(cents, who):
    assert P.refund_approver(cents) == who


def test_refund_approver_rejects_non_positive():
    with pytest.raises(ValueError):
        P.refund_approver(0)


def test_free_shipping_threshold_is_after_discount():
    assert P.expected_shipping_cents(8199, 820, vip=False) == 599  # $73.79 after discount
    assert P.expected_shipping_cents(8199, 699, vip=False) == 0  # exactly $75.00
    assert P.expected_shipping_cents(1000, 0, vip=True) == 0


def test_standard_return_refund_deducts_label_not_shipping_twice():
    assert P.standard_return_refund_cents(999, 0) == 400
    assert P.standard_return_refund_cents(5000, 500) == 3901


def test_warranty_end_by_category():
    d = datetime(2026, 6, 29, 7, 0, tzinfo=ET)
    assert P.warranty_end(d, "Electronics", TZ) == date(2027, 6, 29)
    assert P.warranty_end(d, "Apparel", TZ) == date(2026, 9, 27)
    assert P.warranty_end(d, "Grocery", TZ) is None
    leap = datetime(2028, 2, 29, 12, 0, tzinfo=ET)
    assert P.warranty_end(leap, "Electronics", TZ) == date(2029, 2, 28)


def test_vip_gap_never_negative():
    assert P.vip_gap_cents(67_916) == 32_084
    assert P.vip_gap_cents(150_000) == 0


def test_usd_formatting():
    assert P.usd(57897) == "$578.97"
    assert P.usd(-599) == "-$5.99"
    assert P.usd(100_000) == "$1,000.00"
