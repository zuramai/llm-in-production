import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ch08_normalize import (
    PRINTED_FORMAT,
    PrintedInvoice,
    PrintedLine,
    parse_amount_cents,
    parse_printed_date,
    parse_qty,
    to_invoice,
)
from make_invoices import fmt_amount, fmt_date, fmt_money, generate_specs


@pytest.mark.parametrize("printed,cents", [
    ("$1,234.56", 123456), ("1.234,56 €", 123456), ("£8.55", 855), ("$0.95", 95),
    ("27", 2700), ("1 234,56 €", 123456), ("EUR 12,00", 1200), ("-$5.00", -500),
    ("$12,345,678.90", 1234567890),
])
def test_amounts_in_every_format(printed, cents):
    assert parse_amount_cents(printed) == cents


@pytest.mark.parametrize("bad", ["", "$", "12.345", "1,234", "1,2", "about $5", "$5 + $3",
                                 "1.234.5", "8.555"])
def test_amounts_we_do_not_understand_raise(bad):
    # "12.345" is twelve thousand in Germany and twelve point three in the US: refuse to guess.
    with pytest.raises(ValueError):
        parse_amount_cents(bad)

def test_round_trips_every_generated_amount_and_date():
    # The generator's own formatters are the spec: whatever it prints, we must read back.
    for spec in generate_specs(seed=8, n=50):
        inv = spec.invoice
        for c in [inv.total_cents, inv.tax_cents, *(li.unit_price_cents for li in inv.line_items)]:
            assert parse_amount_cents(fmt_money(c, spec)) == c
            assert parse_amount_cents(fmt_amount(c, spec.money_style)) == c
        country = spec.address[-1]
        assert parse_printed_date(fmt_date(spec.issue, spec.date_style), country) == spec.issue


def test_slash_dates_depend_on_supplier_country():
    assert parse_printed_date("10/01/2026", "USA") == date(2026, 10, 1)
    assert parse_printed_date("10/01/2026", "United Kingdom") == date(2026, 1, 10)
    assert parse_printed_date("10.01.2026", "USA") == date(2026, 1, 10)  # dots: always D.M.Y


@pytest.mark.parametrize("bad", ["13/13/2026", "March 32, 2026", "next Tuesday", "2026/01/10"])
def test_impossible_or_unknown_dates_raise(bad):
    with pytest.raises(ValueError):
        parse_printed_date(bad, "USA")


def test_qty_must_be_a_whole_number():
    assert parse_qty(" 12 ") == 12
    for bad in ("1.5", "12 pcs", "-1", ""):
        with pytest.raises(ValueError):
            parse_qty(bad)


def printed(**kw) -> PrintedInvoice:
    base = {"supplier": "Rheintal Textil GmbH", "supplier_country": "Deutschland",
            "invoice_no": "RE-2026-0042", "invoice_date": "02.04.2026",
            "line_items": [PrintedLine(sku="RT-1000", desc="Wolldecke", qty="2",
                                       unit_price="1.057,00 €")],
            "subtotal": "2.114,00 €", "tax": "401,66 €", "total": "2.515,66 €"}
    return PrintedInvoice(**(base | kw))


def test_to_invoice_converts_a_german_invoice():
    inv = to_invoice(printed())
    assert inv.date == "2026-04-02" and inv.currency == "EUR"
    assert inv.line_items[0].unit_price_cents == 105700
    assert (inv.subtotal_cents, inv.tax_cents, inv.total_cents) == (211400, 40166, 251566)


def test_currency_from_symbol_beats_country_and_conflicts_fall_back_to_country():
    assert to_invoice(printed(subtotal="£2,114.00", total="£2,515.66")).currency == "GBP"
    assert to_invoice(printed(subtotal="2.114,00", total="2.515,66")).currency == "EUR"
    with pytest.raises(ValueError):
        to_invoice(printed(subtotal="2.114,00", total="2.515,66", supplier_country="Mars"))


def test_printed_schema_is_strict_and_all_strings():
    schema = PRINTED_FORMAT["json_schema"]["schema"]
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(PrintedInvoice.model_fields)
    line = schema["properties"]["line_items"]["items"]
    assert {p["type"] for p in line["properties"].values()} == {"string"}
