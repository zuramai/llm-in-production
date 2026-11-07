import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ch08_eval import judge, percentile, summarize
from ch08_extract import Extraction
from ch08_invoice import Invoice, LineItem


def truth() -> Invoice:
    return Invoice(supplier="A Ltd", invoice_no="1", date="2026-03-04", currency="GBP",
                   line_items=[LineItem(sku="S1", desc="x", qty=2, unit_price_cents=500)],
                   subtotal_cents=1000, tax_cents=200, total_cents=1200)


def ex(inv: Invoice | None, cost=0.001, lat=1.0) -> Extraction:
    return Extraction(method="text", model="m", invoice=inv, cost_usd=cost, latency_s=lat,
                      error=None if inv else "parse_error: nope")


def test_four_outcome_buckets():
    t = truth()
    right = t.model_copy(deep=True)
    swapped = t.model_copy(update={"date": "2026-04-03"})  # wrong but arithmetically fine
    bad_math = t.model_copy(update={"total_cents": 1300})  # wrong and detectable
    rows = [judge("a", ex(right), t), judge("b", ex(swapped), t), judge("c", ex(bad_math), t),
            judge("d", ex(None), t)]
    s = summarize(rows)
    assert s["all_correct"] == 25.0
    assert s["silent_wrong"] == 1      # the date swap: no rule can see it
    assert s["caught"] == 2            # bad arithmetic + parse failure
    assert s["false_alarm"] == 0
    assert s["parse_fail"] == 25.0
    assert s["date"] == 50.0 and s["total_cents"] == 50.0


def test_false_alarm_counts_a_correct_invoice_the_rules_reject():
    # Truth itself violates a rule (a currency we do not accept): extraction is right, flagged.
    t = truth().model_copy(update={"currency": "XXX"})
    s = summarize([judge("a", ex(t.model_copy(deep=True)), t)])
    assert s["all_correct"] == 100.0 and s["false_alarm"] == 1


def test_cost_per_thousand_and_percentiles():
    t = truth()
    rows = [judge(str(i), ex(t.model_copy(deep=True), cost=0.002, lat=float(i + 1)), t)
            for i in range(10)]
    s = summarize(rows)
    assert s["usd_per_1k"] == pytest.approx(2.0)
    assert s["p50_s"] == 5.0 and s["p95_s"] == 10.0


def test_percentile_edges():
    assert percentile([3.0], 0.95) == 3.0
    assert percentile([1.0, 2.0], 0.5) == 1.0
    assert percentile([], 0.5) != percentile([], 0.5)  # nan
