"""The 25 hard questions: each needs a fact from the database AND a rule from the documents.

The questions are generated from the live ShopFlow database, because "can they still return it?"
depends on today's date. Each template picks a real order (or customer, or ticket) that fits
it, deterministically (`ORDER BY md5(salt || id)`), and computes the gold facts with hand-written
SQL plus `policy.py`. Day ranges keep every case at least 3 days away from a policy boundary, so
an off-by-one reading of "within 30 days" can't flip the verdict.

Gold facts are the claims an answer must make. Details in parentheses support a fact: the judge
accepts the fact if its main claim is stated and nothing in the answer contradicts the details.

    uv run python -m ch05_agentic.evalset                      # dev set, eval/questions.json
    uv run python -m ch05_agentic.evalset --salt holdout --out eval/questions_holdout.json
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from psycopg.rows import dict_row

from llmprod import connect, get_settings

from . import policy as P

QUESTIONS_PATH = Path(__file__).resolve().parents[1] / "eval" / "questions.json"


@dataclass
class Question:
    id: str
    template: str
    question: str
    gold: list[str]  # facts the answer must state (the judge checks each one)
    docs: list[str]  # documents that hold the rule (for retrieval diagnostics)
    data: dict = field(default_factory=dict)  # the DB facts behind the gold answer


def _d(x) -> str:
    """ISO date. (A first version printed the weekday with strftime('%a'), which follows the
    machine's locale: on this host it printed Indonesian day names.)"""
    return f"{x:%Y-%m-%d}"


# Single-item orders with their product and category, delivered, never refunded, no tickets.
_DELIVERED = """
WITH one AS (
    SELECT oi.order_id, min(p.name) AS product, min(c.name) AS category
    FROM order_items oi JOIN products p ON p.id = oi.product_id
    JOIN categories c ON c.id = p.category_id
    GROUP BY oi.order_id HAVING count(*) = 1)
SELECT o.id, o.customer_id, o.subtotal_cents, o.discount_cents, o.shipping_cents,
       o.total_cents, s.delivered_at, one.product, one.category,
       (now() AT TIME ZONE %(tz)s)::date - (s.delivered_at AT TIME ZONE %(tz)s)::date AS days
FROM orders o
JOIN shipments s ON s.order_id = o.id AND s.status = 'delivered'
JOIN one ON one.order_id = o.id
WHERE o.status = 'delivered'
  AND NOT EXISTS (SELECT 1 FROM refunds r WHERE r.order_id = o.id)
  AND NOT EXISTS (SELECT 1 FROM support_tickets t WHERE t.order_id = o.id)
"""


class _Picker:
    def __init__(self, conn, tz: str, salt: str = ""):
        self.conn, self.tz, self.salt, self.used = conn, tz, salt, set()

    def one(self, sql: str, **params) -> dict:
        """First row (in md5 order) whose id hasn't been used by another question."""
        with self.conn.cursor(row_factory=dict_row) as cur:
            rows = cur.execute(
                f"SELECT * FROM ({sql}) q ORDER BY md5(%(salt)s || q.id::text) LIMIT 50",
                {"tz": self.tz, "salt": self.salt, **params}).fetchall()
        for r in rows:
            if r["id"] not in self.used:
                self.used.add(r["id"])
                return r
        raise LookupError(f"no row fits template: {sql[:120]}...")

    def delivered(self, lo: int, hi: int, extra: str = "", **params) -> dict:
        sql = f"SELECT * FROM ({_DELIVERED}) x WHERE days BETWEEN %(lo)s AND %(hi)s {extra}"
        return self.one(sql, lo=lo, hi=hi, **params)


def build(now: datetime | None = None, salt: str = "") -> list[Question]:
    """`salt` changes which rows each template picks: "" is the dev set we iterated on,
    "holdout" a second set of 25 we only ran at the end."""
    tz = get_settings().business_timezone
    conn = connect()
    pk = _Picker(conn, tz, salt)
    qs: list[Question] = []
    add = qs.append
    excl = "AND category NOT IN ('Beauty', 'Grocery')"
    ret_docs = ["policies/returns-policy.md", "policies/refund-policy.md"]

    def day(ts) -> str:
        return _d(P.local_date(ts, tz))

    # 1-2: the return window (the chapter's headline question)
    r = pk.delivered(8, 15, excl)
    add(Question("r01", "return_open",
                 f"Customer {r['customer_id']} changed their mind about order {r['id']}. "
                 "Can they still return it, and what is the last day they can?",
                 ["Yes, it can still be returned (within 30 days of delivery)",
                  ("The last day to return it is "
                  f"{_d(P.window_end(r['delivered_at'], P.RETURN_WINDOW_DAYS, tz))}")],
                 ret_docs, r))
    r = pk.delivered(40, 60, excl)
    add(Question("r02", "return_closed",
                 f"Can customer {r['customer_id']} still return order {r['id']}?",
                 [("No, the 30-day return window has closed (delivered "
                  f"{day(r['delivered_at'])}, {r['days']} days ago)")], ret_docs, r))

    # 3-4: category exceptions
    r = pk.delivered(5, 15, "AND category = 'Beauty'")
    add(Question("r03", "return_opened_beauty",
                 f"Customer {r['customer_id']} opened the {r['product']} from order {r['id']} "
                 "and doesn't like it. Can they return it for a refund?",
                 [("No: opened Beauty items can't be returned unless they arrived damaged or "
                  "defective (even though it is within 30 days of delivery)")], ret_docs, r))
    r = pk.delivered(5, 15, "AND category = 'Grocery'")
    add(Question("r04", "damaged_opened_grocery",
                 f"The {r['product']} in order {r['id']} arrived damaged and customer "
                 f"{r['customer_id']} has already opened it. What can we offer?",
                 [("It is still eligible: damaged grocery items qualify even if opened (reported "
                  "within 30 days of delivery)"),
                  ("A free replacement or a full refund including shipping (the order total is "
                  f"{P.usd(r['total_cents'])})")],
                 ["policies/damaged-or-defective-items.md", "policies/returns-policy.md"], r))

    # 5-7: cancellation and pending orders
    c_docs = ["policies/order-cancellation.md"]
    r = pk.one("SELECT o.id, o.customer_id, o.total_cents, o.shipping_cents, o.placed_at"
               " FROM orders o WHERE o.status = 'paid'"
               " AND NOT EXISTS (SELECT 1 FROM shipments s WHERE s.order_id = o.id)")
    add(Question("c01", "cancel_paid",
                 f"Customer {r['customer_id']} wants to cancel order {r['id']}. "
                 "Is that possible, and how much do they get back?",
                 [("Yes, it can be cancelled free of charge because it hasn't shipped (status "
                  "paid)"),
                  (f"A full refund of {P.usd(r['total_cents'])} (including shipping, to the "
                  "original payment method within 5-7 business days)")], c_docs, r))
    r = pk.one("SELECT o.id, o.customer_id, s.shipped_at FROM orders o"
               " JOIN shipments s ON s.order_id = o.id"
               " WHERE o.status = 'shipped' AND s.status = 'in_transit'")
    add(Question("c02", "cancel_shipped",
                 f"The customer changed their mind. Can order {r['id']} still be cancelled?",
                 [f"No, it has already shipped (on {day(r['shipped_at'])})",
                  "The customer can return it after delivery (within 30 days of delivery)"],
                 c_docs, r))
    r = pk.one("SELECT o.id, o.customer_id, o.placed_at,"
               " (SELECT count(*) FROM payments p WHERE p.order_id = o.id"
               "   AND p.status = 'failed') AS failed_attempts"
               " FROM orders o WHERE o.status = 'pending' AND o.placed_at > now() - interval '40h'")
    at = P.auto_cancel_at(r["placed_at"]).astimezone(ZoneInfo(tz))
    add(Question("c03", "pending_autocancel",
                 f"Order {r['id']} is still pending. Was the customer charged, and what happens "
                 "to the order if nothing changes?",
                 [(f"The customer was not charged (no successful payment; {r['failed_attempts']} "
                  "failed attempts)"),
                  ("It is cancelled automatically 48 hours after it was placed (around "
                  f"{_d(at)} {at:%H:%M} ET)")], c_docs, r))

    # 8-10: warranty vs return window
    w_docs = ["policies/warranty-policy.md"]
    r = pk.delivered(100, 170, "AND category = 'Electronics'")
    add(Question("w01", "warranty_electronics",
                 f"The {r['product']} from order {r['id']} stopped charging. Customer "
                 f"{r['customer_id']} wants their money back. Return or warranty?",
                 ["The 30-day return window has closed",
                  ("It is covered by the 1-year Electronics warranty (until "
                  f"{_d(P.warranty_end(r['delivered_at'], 'Electronics', tz))})"),
                  ("Under warranty ShopFlow replaces it, or refunds the item price if no "
                  "replacement is available")], w_docs, r))
    cats90 = ("AND category IN ('Apparel', 'Footwear', 'Home & Kitchen', 'Sports & Outdoors',"
              " 'Toys & Games', 'Office', 'Garden', 'Pet Supplies')")
    r = pk.delivered(100, 170, cats90)
    add(Question("w02", "warranty_expired",
                 f"The {r['product']} from order {r['id']} has a manufacturing defect. "
                 "Is it still under warranty?",
                 [(f"No, it is no longer under warranty ({r['category']} has a 90-day warranty, "
                  f"which ended {_d(P.warranty_end(r['delivered_at'], r['category'], tz))})")],
                 w_docs, r))
    r = pk.delivered(40, 75, cats90)
    add(Question("w03", "warranty_90_open",
                 f"Customer {r['customer_id']} reports a defect in the {r['product']} from order "
                 f"{r['id']}. Can we still do anything?",
                 ["The 30-day return window has closed",
                  ("It is still covered by the 90-day warranty (until "
                  f"{_d(P.warranty_end(r['delivered_at'], r['category'], tz))})")], w_docs, r))

    # 11-13: refund approval limits (the full refund of a damaged order = order total)
    a_docs = ["handbook/refund-approval-limits.md", "policies/damaged-or-defective-items.md"]
    who_text = {"support agent": "A support agent can approve it (up to $100.00)",
                "team lead": "A team lead must approve it ($100.01 to $500.00)",
                "finance": "Finance must approve it (above $500.00, via the finance-approvals "
                           "channel)"}
    for qid, lo, hi in (("a01", 3000, 9500), ("a02", 15000, 45000), ("a03", 55000, 10**9)):
        r = pk.delivered(5, 20, "AND total_cents BETWEEN %(tlo)s AND %(thi)s", tlo=lo, thi=hi)
        who = P.refund_approver(r["total_cents"])
        add(Question(qid, f"approver_{who.replace(' ', '_')}",
                     f"Order {r['id']} arrived damaged and customer {r['customer_id']} prefers a "
                     "full refund. Who has to approve it?",
                     [f"The full refund is {P.usd(r['total_cents'])} (including shipping)",
                      who_text[who]], a_docs, r))

    # 14-15: "delivered but not received"
    m_docs = ["policies/shipping-policy.md", "handbook/carrier-claims.md"]
    r = pk.delivered(3, 9)
    add(Question("m01", "missing_open",
                 f"Order {r['id']} shows delivered but customer {r['customer_id']} says it never "
                 "arrived. Can we open a carrier investigation?",
                 [("Yes: it is within 14 days of the delivery date (deadline "
                  f"{_d(P.window_end(r['delivered_at'], P.CARRIER_CLAIM_DAYS, tz))}; the 24-hour"
                  " wait has passed)")], m_docs, r))
    r = pk.delivered(20, 28)
    add(Question("m02", "missing_closed",
                 f"Customer {r['customer_id']} just told us order {r['id']} never arrived, even "
                 "though tracking says delivered. Can we still open a carrier investigation?",
                 [("No: missing-package claims must be made within 14 days of the delivery date "
                  f"(delivered {day(r['delivered_at'])}; the window ended "
                  f"{_d(P.window_end(r['delivered_at'], P.CARRIER_CLAIM_DAYS, tz))})")],
                 m_docs, r))

    # 16-17: refund amounts
    f_docs = ["policies/refund-policy.md"]
    r = pk.delivered(5, 15, excl + " AND shipping_cents = 599 AND discount_cents = 0")
    add(Question("f01", "refund_changed_mind",
                 f"If customer {r['customer_id']} returns order {r['id']} because they changed "
                 "their mind, how much will they get back?",
                 [(f"{P.usd(P.standard_return_refund_cents(r['subtotal_cents'], 0))} (the "
                  f"{P.usd(r['subtotal_cents'])} item price minus the $5.99 return label; the "
                  "original shipping is not refunded)")], f_docs, r))
    r = pk.delivered(5, 20, "AND shipping_cents = 599")
    add(Question("f02", "refund_defective",
                 f"Customer {r['customer_id']} says the item in order {r['id']} is defective and "
                 "wants a refund, not a replacement. How much do we refund?",
                 [(f"The full {P.usd(r['total_cents'])} (including the original shipping; no "
                  "$5.99 return-label deduction)")],
                 f_docs + ["policies/damaged-or-defective-items.md"], r))

    # 18-19: price adjustment (7 days from the successful payment)
    p_sql = ("SELECT o.id, o.customer_id, o.status, p.created_at AS paid_at,"
             " (now() AT TIME ZONE %(tz)s)::date - (p.created_at AT TIME ZONE %(tz)s)::date"
             "   AS days"
             " FROM orders o JOIN payments p ON p.order_id = o.id AND p.status = 'succeeded'"
             " WHERE o.status IN ('paid', 'shipped', 'delivered')"
             " AND NOT EXISTS (SELECT 1 FROM refunds r WHERE r.order_id = o.id)")
    pa_docs = ["policies/price-adjustment-policy.md"]
    r = pk.one(f"SELECT * FROM ({p_sql}) x WHERE days BETWEEN 2 AND 4")
    add(Question("p01", "price_adjust_open",
                 f"Customer {r['customer_id']} saw the price of an item from order {r['id']} "
                 "drop today. Can they get the difference back, and by when must they ask?",
                 [("Yes, a one-time price adjustment is allowed (within 7 days of the successful "
                  f"payment on {day(r['paid_at'])})"),
                  f"They must ask by {_d(P.window_end(r['paid_at'], P.PRICE_ADJUST_DAYS, tz))}"],
                 pa_docs, r))
    r = pk.one(f"SELECT * FROM ({p_sql}) x WHERE days BETWEEN 11 AND 20")
    add(Question("p02", "price_adjust_closed",
                 f"The price of an item in order {r['id']} just dropped. Can customer "
                 f"{r['customer_id']} get a price adjustment?",
                 [("No, the 7-day price adjustment window has passed (payment succeeded "
                  f"{day(r['paid_at'])}; the window ended "
                  f"{_d(P.window_end(r['paid_at'], P.PRICE_ADJUST_DAYS, tz))})")], pa_docs, r))

    # 20-21: VIP (completed purchases in the calendar year)
    spend = ("SELECT c.id, c.segment,"
             " coalesce(sum(o.total_cents) FILTER (WHERE o.status IN"
             "   ('paid', 'shipped', 'delivered')), 0) AS spent_cents,"
             " coalesce(sum(o.total_cents) FILTER (WHERE o.status IN"
             "   ('cancelled', 'refunded')), 0) AS not_counted_cents"
             " FROM customers c JOIN orders o ON o.customer_id = c.id"
             " WHERE o.placed_at >= date_trunc('year', now() AT TIME ZONE %(tz)s)"
             "   AT TIME ZONE %(tz)s"
             " GROUP BY c.id, c.segment")
    v_docs = ["policies/vip-program.md"]
    r = pk.one(f"SELECT * FROM ({spend}) x WHERE segment = 'vip' AND spent_cents >= 100000")
    add(Question("v01", "vip_shipping",
                 f"Does customer {r['id']} pay for standard shipping?",
                 ["No: the customer is VIP, and VIPs get free standard shipping (on every order)"],
                 v_docs, r))
    r = pk.one(f"SELECT * FROM ({spend}) x WHERE segment <> 'vip'"
               " AND spent_cents BETWEEN 60000 AND 95000"
               " AND spent_cents + not_counted_cents >= 100000")
    add(Question("v02", "vip_gap",
                 f"How much more does customer {r['id']} need to spend this year to become VIP?",
                 [(f"{P.usd(P.vip_gap_cents(r['spent_cents']))} more (completed purchases this "
                  f"calendar year total {P.usd(r['spent_cents'])} of the $1,000.00 threshold; "
                  "cancelled and refunded orders don't count)")], v_docs, r))

    # 22-23: shipping charged vs the policy (the $75 threshold is AFTER discounts)
    s_sql = ("SELECT o.id, o.customer_id, o.subtotal_cents, o.discount_cents,"
             " o.shipping_cents, c.segment FROM orders o JOIN customers c"
             " ON c.id = o.customer_id WHERE c.segment <> 'vip' AND o.discount_cents > 0")
    s_docs = ["policies/shipping-policy.md"]
    # Discount pushed the order below $75: $5.99 shipping is correct, though the pre-discount
    # subtotal was over $75 (the trap: applying the threshold before discounts).
    r = pk.one(f"SELECT * FROM ({s_sql}) x WHERE subtotal_cents >= 7500"
               " AND subtotal_cents - discount_cents < 7500 AND shipping_cents = 599")
    net = r["subtotal_cents"] - r["discount_cents"]
    add(Question("s01", "shipping_after_discount",
                 f"Customer {r['customer_id']} says their cart was over $75 but they still paid "
                 f"shipping on order {r['id']}. Were they charged correctly?",
                 [(f"Yes: after the {P.usd(r['discount_cents'])} discount the order was "
                  f"{P.usd(net)}, below the $75.00 free-shipping threshold (which applies after "
                  "discounts; the customer isn't VIP)"),
                  "$5.99 standard shipping was correctly charged"],
                 s_docs, r))
    r = pk.one(f"SELECT * FROM ({s_sql}) x WHERE subtotal_cents - discount_cents >= 7600"
               " AND subtotal_cents - discount_cents < 8500 AND shipping_cents = 0")
    net = r["subtotal_cents"] - r["discount_cents"]
    add(Question("s02", "shipping_correct",
                 f"Customer {r['customer_id']} asks why they paid no shipping on order "
                 f"{r['id']}. Was that correct?",
                 [(f"Yes: after the {P.usd(r['discount_cents'])} discount the order was "
                  f"{P.usd(net)}, at or above the $75.00 free standard shipping threshold")],
                 s_docs, r))

    # 24: a support ticket that is a safety issue. The hand-written hard ticket: no "urgent" or
    # "safety" in its text, the agent has to infer it.
    r = pk.one("SELECT t.id, t.order_id, t.subject, t.body FROM support_tickets t"
               " WHERE t.order_id IS NOT NULL AND t.body ILIKE '%%swelled%%'"
               " AND t.body ILIKE '%%power bank%%'")
    add(Question("t01", "ticket_safety",
                 f"How should we handle support ticket {r['id']}, and how fast must we respond?",
                 ["It is a safety issue: priority urgent",
                  "First response within 1 hour",
                  "Tell the customer to stop using the product immediately"],
                 ["handbook/support-escalation.md", "policies/damaged-or-defective-items.md"],
                 r))

    # 25: international delivery estimate
    r = pk.one("SELECT o.id, o.customer_id, c.country, s.shipped_at FROM orders o"
               " JOIN customers c ON c.id = o.customer_id JOIN shipments s ON s.order_id = o.id"
               " WHERE o.status = 'shipped' AND s.status = 'in_transit'"
               " AND c.country IN ('GB', 'DE', 'AU', 'ID')")
    add(Question("i01", "intl_eta",
                 f"Customer {r['customer_id']} asks when order {r['id']} will arrive. "
                 "What do we tell them?",
                 [f"Delivery to {r['country']} takes 7-14 business days (customs can add time)",
                  f"It shipped on {day(r['shipped_at'])}"],
                 ["policies/international-shipping.md", "faq.md"], r))
    conn.close()
    return qs


def _jsonable(v):
    if isinstance(v, datetime):
        return v.isoformat()
    return v


def save(qs: list[Question], path: Path = QUESTIONS_PATH) -> None:
    out = {"built_at": datetime.now(ZoneInfo(get_settings().business_timezone))
           .isoformat(timespec="seconds"),
           "questions": [{**asdict(q), "data": {k: _jsonable(v) for k, v in q.data.items()}}
                         for q in qs]}
    path.write_text(json.dumps(out, indent=2) + "\n")


def load(path: Path = QUESTIONS_PATH) -> tuple[str, list[Question]]:
    raw = json.loads(path.read_text())
    return raw["built_at"], [Question(**q) for q in raw["questions"]]


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Build the eval questions from today's data")
    ap.add_argument("--salt", default="", help='"" = dev set, "holdout" = held-out set')
    ap.add_argument("--out", type=Path, default=QUESTIONS_PATH)
    args = ap.parse_args()
    qs = build(salt=args.salt)
    save(qs, args.out)
    for q in qs:
        print(f"{q.id} {q.template:24} {q.question}")
