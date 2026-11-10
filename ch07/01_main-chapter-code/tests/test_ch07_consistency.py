import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ch07_semantic import compile_query
from ch07_semantic.consistency import (
    QUESTIONS,
    TOLERANCE,
    consistency_table,
    distinct_answers,
    row_value,
    same_answer,
    stated_value,
)

REFUND, ACTIVE, NET, AOV = QUESTIONS


def test_each_question_has_the_kind_of_its_governed_metric():
    assert [QUESTIONS[q].kind for q in (REFUND, ACTIVE, NET, AOV)] == \
        ["rate", "count", "money", "money"]
    assert TOLERANCE == {"rate": 0.0001, "money": 0.01, "count": 0.0}


def test_rates_match_to_a_hundredth_of_a_percentage_point():
    assert same_answer(0.0307, 0.030714285714285715, "rate")      # "3.07%" vs the governed value
    assert same_answer(0.0306, 0.0307, "rate")                    # one unit in the last place
    assert not same_answer(0.0305, 0.0307, "rate")
    assert not same_answer(0.0253, 0.0307, "rate")                # the old 0.01 tolerance said yes
    assert not same_answer(0.030133, 0.030714, "rate")            # 3.01% is another definition


def test_money_matches_to_the_cent_and_counts_exactly():
    assert same_answer(120.38, 120.38478790338948, "money")
    assert not same_answer(120.40, 120.38478790338948, "money")
    assert same_answer(4066, 4066.0, "count")
    assert not same_answer(4067, 4066, "count")
    assert not same_answer(4077, 4066, "count")


def test_no_number_is_its_own_answer():
    assert same_answer(None, None, "rate")
    assert not same_answer(None, 0.0, "rate")
    assert distinct_answers([0.0626, 0.06258517, None, 0.0301, None], "rate") == \
        [0.0626, None, 0.0301]


def test_stated_value_is_what_the_reply_says():
    q = QUESTIONS[REFUND]
    assert stated_value("The refund rate this month is 7.63% (406 refunds out of 5,320 orders).",
                        q) == pytest.approx(0.0763)
    assert stated_value("Our refund rate this month is 0.02531275857196732 (about 2.53%): $4,244.03",
                        q) == pytest.approx(0.0253)
    assert stated_value("This month's refund rate is 3.0133146461107216%: 43 of 1,427", q) == \
        pytest.approx(0.030133146461107216)
    assert stated_value("I couldn't compute it.", q) is None
    assert stated_value(None, q) is None


def test_stated_count_is_the_number_in_front_of_the_label_not_the_window():
    q = QUESTIONS[ACTIVE]
    assert stated_value("We have **4,077 active customers**, counted over the last 90 days. "
                        "This is as of 3:03 AM EST.", q) == 4077
    assert stated_value("You have **4,077** active customers (any order status).", q) == 4077
    assert stated_value("In the last 90 days including today, 4,066 active customers ordered.",
                        q) == 4066


def test_stated_money_skips_dates():
    q = QUESTIONS[NET]
    assert stated_value("Net revenue for last week (Mon 2026-11-02 to Mon 2026-11-09) was "
                        "**$123,553.77**.", q) == 123553.77


def test_row_value_reads_percent_columns_as_percentages():
    q = QUESTIONS[REFUND]
    assert row_value([{"refunds_usd": 10493.26, "refund_rate_pct": 6.26}], q) == \
        pytest.approx(0.0626)
    assert row_value({"refund_rate": 0.0626}, q) == 0.0626
    # qwen3:8b: integer division made the column 0 while the reply said 7.63%.
    assert row_value({"total_orders": 5320, "refunds": 406, "refund_rate_percent": 0}, q) == 0.0
    assert row_value([], q) is None
    assert row_value(None, q) is None


def _run(variant, question, answer, row):
    return {"variant": variant, "question": question, "answer": answer, "row": row}


def test_reference_is_compile_query_at_run_time_not_the_first_semantic_answer():
    result = {"model": "qwen3:8b", "run_at": "20261109T080108", "runs": [
        _run("semantic", ACTIVE, "We have 93 active customers.", {"active_customers": 93}),
        _run("semantic", ACTIVE, "We have 4,066 active customers.", {"active_customers": 4066}),
        _run("semantic", ACTIVE, "We have 816 active customers.", {"active_customers": 816}),
        _run("sql", REFUND, "The refund rate this month is 7.63%.", {"refund_rate_percent": 0}),
    ]}
    governed = {"active_customers": 4066, "refund_rate": 0.030714285714285715,
                "net_revenue": 123553.77, "aov": 120.38478790338948}
    seen = []

    def fake_db(sql):
        seen.append(sql)
        metric = next(m for m in governed if f"AS {m}" in sql)
        return [{metric: governed[metric]}]

    table = {(r["variant"], r["question"]): r for r in consistency_table([result], run=fake_db)}

    now = datetime(2026, 11, 9, 8, 1, 8, tzinfo=UTC)
    assert seen == [compile_query(q.query, now=now) for q in QUESTIONS.values()]
    active = table[("semantic", ACTIVE)]
    assert active["governed"] == 4066
    assert (active["runs"], active["distinct"], active["matches governed"]) == (3, 3, 1)
    refund = table[("sql", REFUND)]
    assert refund["answers"] == [0.0763]          # the table uses the stated value...
    assert refund["row ≠ stated"] == 1            # ...and counts where the SQL row disagreed


def test_files_with_different_governed_values_are_not_pooled():
    runs = [_run("sql", NET, "Net revenue was $1.00.", {"net_revenue": 1.0})]
    files = [{"model": "m", "run_at": stamp, "runs": runs}
             for stamp in ("20261109T080108", "20261116T080108")]
    # Governed values in QUESTIONS order (refund rate, active, net revenue, AOV), per file.
    values = iter([0.03, 4066, 1.0, 120.0, 0.03, 4066, 2.0, 120.0])

    def fake_db(sql):
        metric = next(m for m in ("refund_rate", "active_customers", "net_revenue", "aov")
                      if f"AS {m}" in sql)
        return [{metric: next(values)}]

    with pytest.raises(ValueError, match="governed value changed"):
        consistency_table(files, run=fake_db)
