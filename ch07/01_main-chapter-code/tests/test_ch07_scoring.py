import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ch07_semantic.scoring import rows_match, values_equal


def test_dates_compare_by_calendar_day():
    assert values_equal("2026-11-02", "2026-11-02T00:00:00-05:00")
    assert not values_equal("2026-11-02", "2026-11-01T23:00:00-05:00")
    assert values_equal("2026-08-01", "2026-08")             # TO_CHAR(..., 'YYYY-MM')
    assert not values_equal("2026-08-01", "2026-09")


def test_fraction_may_be_a_percentage_but_not_off_by_a_cent():
    assert values_equal(0.07356, 7.36)
    assert not values_equal(0.07356, 7.56)
    assert not values_equal(1, 100)             # counts are not fractions


def test_numbers_and_strings_never_match():
    assert not values_equal(5, "5")


def test_extra_columns_and_order_are_ignored():
    gold = [("web", 10.0), ("ios", 5.0)]
    pred = [{"channel": "ios", "orders": 3, "revenue": 5.0},
            {"channel": "web", "orders": 4, "revenue": 10.0}]
    assert rows_match(gold, pred)


def test_row_count_and_values_must_match():
    assert not rows_match([("web", 10.0)], [{"c": "web", "r": 10.0}, {"c": "ios", "r": 1.0}])
    assert not rows_match([("web", 10.0)], [{"c": "web", "r": 10.5}])
    assert not rows_match([(1,)], None)


def test_duplicate_rows_are_a_multiset():
    assert not rows_match([("a", 1), ("a", 1)], [{"x": "a", "y": 1}, {"x": "b", "y": 1}])
