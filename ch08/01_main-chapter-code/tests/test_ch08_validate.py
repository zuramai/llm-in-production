import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ch08_invoice import Invoice, LineItem, validate_invoice

TODAY = date(2026, 11, 7)


def good() -> Invoice:
    return Invoice(
        supplier="Harbor Lane Packaging Inc.", invoice_no="INV-2026-00417", date="2026-03-14",
        currency="USD",
        line_items=[LineItem(sku="HL-1", desc="Box", qty=10, unit_price_cents=250),
                    LineItem(sku="HL-2", desc="Tape", qty=3, unit_price_cents=1099)],
        subtotal_cents=5797, tax_cents=464, total_cents=6261)


def rules(inv: Invoice) -> set[str]:
    return {v.rule for v in validate_invoice(inv, TODAY)}


def test_consistent_invoice_passes():
    assert validate_invoice(good(), TODAY) == []


def test_cents_confused_with_dollars_breaks_line_sum():
    # The classic: the model returns 2.50 -> 2 (or 250 -> 25000) for a unit price.
    inv = good()
    inv.line_items[0].unit_price_cents = 25000
    assert rules(inv) == {"line_sum"}


def test_one_cent_off_is_caught():
    inv = good()
    inv.subtotal_cents += 1
    inv.total_cents += 1
    assert rules(inv) == {"line_sum"}


def test_total_must_equal_subtotal_plus_tax():
    inv = good()
    inv.total_cents = inv.subtotal_cents  # forgot the tax
    assert rules(inv) == {"total_sum"}


def test_both_arithmetic_rules_reported_together():
    inv = good()
    inv.subtotal_cents = 1
    inv.total_cents = 2
    assert rules(inv) == {"line_sum", "total_sum"}


@pytest.mark.parametrize("bad", ["March 14, 2026", "14.03.2026", "2026-13-01", "2026-02-30", ""])
def test_unparseable_dates_rejected(bad):
    inv = good()
    inv.date = bad
    assert rules(inv) == {"date_parse"}


@pytest.mark.parametrize(("d", "ok"), [
    ("2026-11-08", True),   # tomorrow: allowed (time zones)
    ("2026-11-09", False),  # two days ahead: a swapped DD/MM or a typo
    ("2000-01-01", True),
    ("1999-12-31", False),
])
def test_date_range_boundaries(d, ok):
    inv = good()
    inv.date = d
    assert (validate_invoice(inv, TODAY) == []) is ok


@pytest.mark.parametrize("cur", ["US$", "usd", "dollars", "", "$"])
def test_currency_must_be_iso_code(cur):
    inv = good()
    inv.currency = cur
    assert rules(inv) == {"currency_iso"}


def test_zero_tax_is_valid():
    inv = good()
    inv.tax_cents, inv.total_cents = 0, inv.subtotal_cents
    assert validate_invoice(inv, TODAY) == []


def test_empty_invoice_has_no_lines():
    inv = good()
    inv.line_items = []
    inv.subtotal_cents, inv.tax_cents, inv.total_cents = 0, 0, 0
    assert rules(inv) == {"has_lines"}


def test_negative_and_zero_quantities_rejected():
    inv = good()
    inv.line_items[0].qty = 0
    assert "line_values" in rules(inv)
