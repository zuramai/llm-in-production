"""ShopFlow's written policies as plain functions. These compute the GOLD answers of the eval.

Every constant cites the document it comes from (datasets/shopflow/docs/). The agent never sees
this module: it has to find the same rules in the documents and the same facts in the database.
Windows are calendar days counted on the business-time-zone date: "within 30 days of delivery"
means the last eligible day is delivery date + 30 days.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

RETURN_WINDOW_DAYS = 30  # policies/returns-policy.md: "within 30 days of delivery"
DAMAGE_REPORT_DAYS = 30  # policies/damaged-or-defective-items.md
CARRIER_CLAIM_DAYS = 14  # policies/shipping-policy.md: "within 14 days of the delivery date"
PRICE_ADJUST_DAYS = 7  # policies/price-adjustment-policy.md: from when payment succeeds
PENDING_AUTOCANCEL_HOURS = 48  # policies/order-cancellation.md
RETURN_LABEL_CENTS = 599  # policies/refund-policy.md: "$5.99 return shipping is deducted"
STANDARD_SHIPPING_CENTS = 599  # policies/shipping-policy.md
FREE_SHIPPING_MIN_CENTS = 7500  # policies/shipping-policy.md: "$75.00 ... after discounts"
VIP_THRESHOLD_CENTS = 100_000  # policies/vip-program.md: "$1,000.00 or more in a calendar year"
AGENT_LIMIT_CENTS = 10_000  # handbook/refund-approval-limits.md: "Up to $100.00"
LEAD_LIMIT_CENTS = 50_000  # handbook/refund-approval-limits.md: "$100.01 to $500.00"
NOT_RETURNABLE_IF_OPENED = frozenset({"Beauty", "Grocery"})  # policies/returns-policy.md
WARRANTY_DAYS_90 = 90  # policies/warranty-policy.md (all categories except below)


def local_date(ts: datetime, tz: str) -> date:
    """The calendar date of `ts` in the business time zone (never the UTC date)."""
    if ts.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return ts.astimezone(ZoneInfo(tz)).date()


def window_end(start: datetime, days: int, tz: str) -> date:
    """Last eligible calendar day of a window of `days` days starting at `start`."""
    return local_date(start, tz) + timedelta(days=days)


def is_open(start: datetime, days: int, now: datetime, tz: str) -> bool:
    return local_date(now, tz) <= window_end(start, days, tz)


def warranty_end(delivered_at: datetime, category: str, tz: str) -> date | None:
    """policies/warranty-policy.md: Electronics 1 year, Grocery no warranty, the rest 90 days."""
    d = local_date(delivered_at, tz)
    if category == "Grocery":
        return None
    if category == "Electronics":
        try:
            return d.replace(year=d.year + 1)
        except ValueError:  # delivered on Feb 29
            return d.replace(year=d.year + 1, day=28)
    return d + timedelta(days=WARRANTY_DAYS_90)


def refund_approver(amount_cents: int) -> str:
    """handbook/refund-approval-limits.md. Limits apply to the total refund per order."""
    if amount_cents <= 0:
        raise ValueError("refund amount must be positive")
    if amount_cents <= AGENT_LIMIT_CENTS:
        return "support agent"
    if amount_cents <= LEAD_LIMIT_CENTS:
        return "team lead"
    return "finance"


def standard_return_refund_cents(subtotal_cents: int, discount_cents: int) -> int:
    """A full 'changed my mind' return: items as paid (after discount), original shipping not
    refunded, $5.99 return label deducted (policies/refund-policy.md)."""
    return subtotal_cents - discount_cents - RETURN_LABEL_CENTS


def expected_shipping_cents(subtotal_cents: int, discount_cents: int, vip: bool) -> int:
    """Standard shipping per policies/shipping-policy.md: free for VIP, free at $75.00 or more
    AFTER discounts, else $5.99."""
    if vip or subtotal_cents - discount_cents >= FREE_SHIPPING_MIN_CENTS:
        return 0
    return STANDARD_SHIPPING_CENTS


def vip_gap_cents(spent_cents: int) -> int:
    """How much more a customer must spend this calendar year to reach VIP (0 if already there).
    Only completed purchases count (policies/vip-program.md)."""
    return max(0, VIP_THRESHOLD_CENTS - spent_cents)


def auto_cancel_at(placed_at: datetime) -> datetime:
    return placed_at + timedelta(hours=PENDING_AUTOCANCEL_HOURS)


def usd(cents: int) -> str:
    sign = "-" if cents < 0 else ""
    return f"{sign}${abs(cents) / 100:,.2f}"
