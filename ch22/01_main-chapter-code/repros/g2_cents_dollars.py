"""G2: integer cents read as dollars (a 100x error), and float money that doesn't add up."""

from __future__ import annotations

import datetime as dt
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from zoneinfo import ZoneInfo

from ch22_common import LUNA, Result, chat, timed

from llmprod import connect, get_settings

ID, TITLE, LAYER, KIND = "G2", "Cents vs dollars", "Data & time", "hosted"

SYSTEM = ("You are ShopFlow's data assistant. Answer the question in one sentence using only the "
          "query result.")


def format_cents(cents: int) -> str:
    """Integer cents -> '$1,234.56' with exact arithmetic (no float anywhere)."""
    sign = "-" if cents < 0 else ""
    dollars = Decimal(abs(cents)) / 100
    return f"{sign}${dollars:,.2f}"


def phrase(row: dict) -> str:
    r = chat(LUNA, [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": f"Question: What was revenue today?\n"
                                                f"Query result: {row}"}],
             max_tokens=80, reasoning_effort="none")
    return (r.choices[0].message.content or "").replace(",", "")


@timed
def run(n: int = 10) -> Result:
    s = get_settings()
    with connect() as conn:
        last = conn.execute("SELECT max(placed_at) FROM orders").fetchone()[0]
        day = last.astimezone(ZoneInfo(s.business_timezone)).date()
        start = dt.datetime.combine(day, dt.time.min, tzinfo=ZoneInfo(s.business_timezone))
        cents = conn.execute(
            "SELECT sum(total_cents) FROM orders WHERE placed_at >= %s AND placed_at < %s "
            "AND status <> 'cancelled'", (start, start + dt.timedelta(days=1))).fetchone()[0]
    right = format_cents(cents).replace(",", "")
    with ThreadPoolExecutor(5) as pool:
        naive = list(pool.map(phrase, [{"revenue": cents}] * n))
        fixed = list(pool.map(phrase, [{"revenue_usd": format_cents(cents)}] * n))
    naive_wrong = sum(str(cents) in a and right not in a for a in naive)
    fixed_right = sum(right in a for a in fixed)
    total = 0.0
    for _ in range(3):  # three $0.10 items, the way a loop over line items adds them
        total += 0.10
    return Result(
        ID, TITLE, LAYER, KIND,
        reproduced=naive_wrong > 0 or total != 0.30,
        fixed=fixed_right == n and sum([Decimal("0.10")] * 3) == Decimal("0.30"),
        evidence=f"revenue on {day} = {cents} cents = {format_cents(cents)}. Row {{'revenue': "
                 f"{cents}}} -> luna reported ${cents:,} in {naive_wrong}/{n}; row "
                 f"{{'revenue_usd': '{format_cents(cents)}'}} -> right in {fixed_right}/{n}. "
                 f"Floats: 0.10 + 0.10 + 0.10 = {total!r}",
        data={"cents": cents, "naive": naive, "fixed": fixed},
    )
