"""G4: Python subtracts two aware datetimes in the same zone by wall clock, ignoring DST."""

from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

from ch22_common import Result, timed

ID, TITLE, LAYER, KIND = "G4", "Aware datetime arithmetic ignores DST", "Data & time", "local"

NY = ZoneInfo("America/New_York")
UTC = dt.UTC


def elapsed(a: dt.datetime, b: dt.datetime) -> dt.timedelta:
    """Real elapsed time between two aware datetimes: compare instants, not wall clocks."""
    return b.astimezone(UTC) - a.astimezone(UTC)


def add_hours(t: dt.datetime, hours: float) -> dt.datetime:
    """'48 hours later' as an instant, shown back in t's zone."""
    return (t.astimezone(UTC) + dt.timedelta(hours=hours)).astimezone(t.tzinfo)


@timed
def run() -> Result:
    start = dt.datetime(2026, 10, 29, 0, 0, tzinfo=NY)  # EDT, UTC-4
    end = dt.datetime(2026, 11, 5, 0, 0, tzinfo=NY)     # EST, UTC-5 (DST ended Nov 1)
    naive_week = end - start
    real_week = elapsed(start, end)
    ticket = dt.datetime(2026, 10, 31, 9, 0, tzinfo=NY)
    naive_sla = ticket + dt.timedelta(hours=48)          # wall-clock addition
    real_sla = add_hours(ticket, 48)
    return Result(
        ID, TITLE, LAYER, KIND,
        reproduced=naive_week != real_week,
        fixed=real_week == dt.timedelta(hours=169) and real_sla.hour == 8,
        evidence=f"{start:%Y-%m-%d} -> {end:%Y-%m-%d} ET: end - start = {naive_week} "
                 f"({naive_week.total_seconds() / 3600:.0f} h), real "
                 f"{real_week.total_seconds() / 3600:.0f} h; 48 h SLA from "
                 f"{ticket:%m-%d %H:%M %Z}: naive {naive_sla:%m-%d %H:%M %Z}, "
                 f"real {real_sla:%m-%d %H:%M %Z}",
    )
