"""G1: "today" depends on whose clock: a UTC session counts a different day than the business."""

from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import psycopg
from ch22_common import Result, timed

from llmprod import connect, get_settings

ID, TITLE, LAYER, KIND = "G1", 'Timezone "today"', "Data & time", "db"

NAIVE_SQL = "SELECT count(*) FROM orders WHERE placed_at::date = %s::date"
FIXED_SQL = "SELECT count(*) FROM orders WHERE placed_at >= %s AND placed_at < %s"


def business_day_bounds(day: dt.date, tz: str) -> tuple[dt.datetime, dt.datetime]:
    """Half-open [start, end) of a local calendar day as aware datetimes (23 or 25 h on DST)."""
    z = ZoneInfo(tz)
    start = dt.datetime.combine(day, dt.time.min, tzinfo=z)
    end = dt.datetime.combine(day + dt.timedelta(days=1), dt.time.min, tzinfo=z)
    return start, end


@timed
def run() -> Result:
    s = get_settings()
    with connect() as biz:
        # "Now" pinned to the seeder's last order, so the repro works on any later day too.
        last = biz.execute("SELECT max(placed_at) FROM orders").fetchone()[0]
        today = last.astimezone(ZoneInfo(s.business_timezone)).date()
        start, end = business_day_bounds(today, s.business_timezone)
        biz_session = biz.execute(NAIVE_SQL, (today,)).fetchone()[0]
        fixed_biz = biz.execute(FIXED_SQL, (start, end)).fetchone()[0]
    with psycopg.connect(s.database_url, autocommit=True) as raw:  # server default: UTC
        server_tz = raw.execute("SHOW TIME ZONE").fetchone()[0]
        utc_session = raw.execute(NAIVE_SQL, (today,)).fetchone()[0]
        fixed_utc = raw.execute(FIXED_SQL, (start, end)).fetchone()[0]
    return Result(
        ID, TITLE, LAYER, KIND,
        reproduced=utc_session != biz_session,
        fixed=fixed_utc == fixed_biz == biz_session,
        evidence=f"orders on {today} (last order {last:%H:%M %Z}): `placed_at::date = day` gives "
                 f"{utc_session} in a {server_tz} session, {biz_session} in "
                 f"{s.business_timezone}; explicit [{start:%H:%M %z}, next day) bounds give "
                 f"{fixed_utc} in both",
        data={"utc": utc_session, "business": biz_session, "fixed": fixed_utc},
    )
