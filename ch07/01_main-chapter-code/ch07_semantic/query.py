"""What the LLM is allowed to say: a MetricQuery. No SQL anywhere in it."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field

Grain = Literal["day", "week", "month", "quarter", "year"]
Preset = Literal[
    "today", "yesterday", "this_week", "last_week", "this_month", "last_month",
    "this_quarter", "last_quarter", "this_year", "last_year",
    "last_n", "previous_n", "custom", "all_time",
]
Op = Literal["=", "!=", "in", "not in", ">", ">=", "<", "<="]


class Filter(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dimension: str
    op: Op
    values: list[str]


class TimeRange(BaseModel):
    """A calendar period in the business time zone, resolved by code, not by the model.

    last_n:     the current `unit` plus the n-1 before it (last_n 7 days = today and 6 more).
    previous_n: the n complete units before the current one (previous_n 3 months).
    custom:     `start` and `end` ISO dates, both inclusive.
    """

    model_config = ConfigDict(extra="forbid")

    preset: Preset
    unit: Grain | None = None
    n: int | None = None
    start: str | None = None
    end: str | None = None


class OrderBy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str
    descending: bool = False


class MetricQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metrics: list[str] = Field(min_length=1)
    dimensions: list[str] = []
    filters: list[Filter] = []
    time_range: TimeRange | None = None
    grain: Grain | None = None
    order_by: list[OrderBy] = []
    limit: int | None = None


class QueryError(ValueError):
    """The MetricQuery is invalid for this semantic model. The message is shown to the model."""


_NAMED = {
    "today": ("day", 0), "yesterday": ("day", 1), "this_week": ("week", 0),
    "last_week": ("week", 1), "this_month": ("month", 0), "last_month": ("month", 1),
    "this_quarter": ("quarter", 0), "last_quarter": ("quarter", 1),
    "this_year": ("year", 0), "last_year": ("year", 1),
}


def period_start(day: date, unit: str) -> date:
    if unit == "day":
        return day
    if unit == "week":                          # weeks start on Monday
        return day - timedelta(days=day.weekday())
    if unit == "month":
        return day.replace(day=1)
    if unit == "quarter":
        return day.replace(month=3 * ((day.month - 1) // 3) + 1, day=1)
    if unit == "year":
        return day.replace(month=1, day=1)
    raise QueryError(f"unknown time unit {unit!r}")


def shift(start: date, unit: str, n: int) -> date:
    """Move a period start by n units (n may be negative)."""
    if unit == "day":
        return start + timedelta(days=n)
    if unit == "week":
        return start + timedelta(weeks=n)
    months = {"month": 1, "quarter": 3, "year": 12}[unit] * n
    index = start.year * 12 + start.month - 1 + months
    return date(index // 12, index % 12 + 1, 1)


def resolve(tr: TimeRange, now: datetime, tz: str) -> tuple[datetime, datetime] | None:
    """[start, end) as timezone-aware datetimes, or None for all time.

    Boundaries are local midnights converted with zoneinfo, so a week that crosses a DST
    change is 167 or 169 hours long, as the business calendar says it should be.
    """
    zone = ZoneInfo(tz)
    today = now.astimezone(zone).date()
    if tr.preset == "all_time":
        return None
    if tr.preset == "custom":
        try:
            first, last = date.fromisoformat(tr.start or ""), date.fromisoformat(tr.end or "")
        except ValueError:
            raise QueryError("time_range custom needs start and end as YYYY-MM-DD") from None
        if last < first:
            raise QueryError(f"time_range end {last} is before start {first}")
        lo, hi = first, last + timedelta(days=1)
    elif tr.preset in _NAMED:
        unit, back = _NAMED[tr.preset]
        lo = shift(period_start(today, unit), unit, -back)
        hi = shift(lo, unit, 1)
    else:
        if tr.unit is None or tr.n is None or tr.n < 1:
            raise QueryError(f"time_range {tr.preset} needs unit and n >= 1")
        current = period_start(today, tr.unit)
        if tr.preset == "last_n":
            lo, hi = shift(current, tr.unit, -(tr.n - 1)), shift(current, tr.unit, 1)
        else:
            lo, hi = shift(current, tr.unit, -tr.n), current
    return (datetime.combine(lo, time(), zone), datetime.combine(hi, time(), zone))
