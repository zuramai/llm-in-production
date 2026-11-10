import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ch07_semantic import QueryError, TimeRange, resolve

TZ = "America/New_York"


def window(now: datetime, **kwargs) -> tuple[str, str]:
    lo, hi = resolve(TimeRange(**kwargs), now, TZ)
    return lo.isoformat(), hi.isoformat()


def test_today_uses_the_business_day_not_utc():
    # 02:00 UTC on Nov 7 is still Nov 6 in New York.
    now = datetime(2026, 11, 7, 2, 0, tzinfo=UTC)
    assert window(now, preset="today") == ("2026-11-06T00:00:00-05:00",
                                           "2026-11-07T00:00:00-05:00")


def test_week_across_dst_change_is_169_hours():
    now = datetime(2026, 11, 7, 12, 0, tzinfo=UTC)
    lo, hi = resolve(TimeRange(preset="last_week"), now, TZ)
    # Subtract in UTC: Python subtracts two datetimes sharing one tzinfo as wall-clock times
    # and would report exactly 7 days.
    assert hi.astimezone(UTC) - lo.astimezone(UTC) == timedelta(hours=169)


def test_previous_n_months_crosses_year_boundary():
    now = datetime(2027, 1, 15, 12, 0, tzinfo=UTC)
    assert window(now, preset="previous_n", unit="month", n=3) == (
        "2026-10-01T00:00:00-04:00", "2027-01-01T00:00:00-05:00")


def test_last_n_includes_the_current_period():
    now = datetime(2026, 11, 7, 12, 0, tzinfo=UTC)
    assert window(now, preset="last_n", unit="day", n=7) == (
        "2026-11-01T00:00:00-04:00", "2026-11-08T00:00:00-05:00")


def test_quarter_and_year_presets():
    now = datetime(2026, 11, 7, 12, 0, tzinfo=UTC)
    assert window(now, preset="last_quarter")[0].startswith("2026-07-01")
    assert window(now, preset="this_quarter")[1].startswith("2027-01-01")
    assert window(now, preset="last_year") == ("2025-01-01T00:00:00-05:00",
                                               "2026-01-01T00:00:00-05:00")


def test_custom_end_date_is_inclusive():
    now = datetime(2026, 11, 7, 12, 0, tzinfo=UTC)
    assert window(now, preset="custom", start="2026-10-01", end="2026-10-31")[1] == \
        "2026-11-01T00:00:00-04:00"


@pytest.mark.parametrize("kwargs", [
    {"preset": "custom", "start": "2026-10-31", "end": "2026-10-01"},
    {"preset": "custom", "start": "October 1st", "end": "2026-10-31"},
    {"preset": "last_n", "unit": "day"},
    {"preset": "previous_n", "unit": "week", "n": 0},
])
def test_bad_ranges_raise(kwargs):
    with pytest.raises(QueryError):
        resolve(TimeRange(**kwargs), datetime(2026, 11, 7, tzinfo=UTC), TZ)


def test_all_time_has_no_bounds():
    assert resolve(TimeRange(preset="all_time"), datetime(2026, 11, 7, tzinfo=UTC), TZ) is None
