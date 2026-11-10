"""Copy, then convert: the model transcribes values as printed, Python turns them into cents and
ISO dates.

Small models are decent at *reading* `1.234,56 €` and bad at *converting* it to `123456`. So we
split the job: a `PrintedInvoice` schema of strings the model copies verbatim, and deterministic
parsers that are unit-tested. The parsers refuse anything they don't understand (ValueError),
which sends the invoice to the repair/review path instead of guessing.
"""

from __future__ import annotations

import re
from datetime import date

from ch08_invoice import Invoice, LineItem, strict_schema
from pydantic import BaseModel, ConfigDict, Field

PRINTED_RULES = """You transcribe supplier invoices for ShopFlow's accounts-payable team.
Copy every value EXACTLY as it is printed on the invoice, character by character: keep currency
symbols, thousands separators and decimal separators ("$1,234.56", "1.234,56 €", "£8.55"), and
keep dates in the printed format ("March 14, 2026", "03/14/2026", "14.03.2026").
Do not convert, compute or reformat anything.
- invoice_date is the invoice (issue) date, not the due date, delivery date or payment date.
- supplier is the company that issued the invoice, not the customer (ShopFlow Inc.).
- supplier_country is the country in the supplier's address, as printed.
- Include every line item, from every page, in printed order."""


class PrintedLine(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sku: str = Field(description="SKU / article number as printed")
    desc: str = Field(description="Line description as printed")
    qty: str = Field(description="Quantity as printed")
    unit_price: str = Field(description="Unit price as printed, with symbol and separators")


class PrintedInvoice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    supplier: str = Field(description="Name of the company that issued the invoice")
    supplier_country: str = Field(description="Country in the supplier's address, as printed")
    invoice_no: str = Field(description="Invoice number as printed")
    invoice_date: str = Field(description="Issue date exactly as printed")
    line_items: list[PrintedLine]
    subtotal: str = Field(description="Subtotal as printed")
    tax: str = Field(description="Tax amount as printed")
    total: str = Field(description="Total / amount due as printed")


PRINTED_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "PrintedInvoice", "strict": True,
                    "schema": strict_schema(PrintedInvoice)},
}

MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august", "september",
     "october", "november", "december"], start=1)}
MONTHS |= {"januar": 1, "februar": 2, "märz": 3, "mai": 5, "juni": 6, "juli": 7, "oktober": 10,
           "dezember": 12}
# Countries whose numeric dates are month-first. Everything else we see is day-first.
MONTH_FIRST = {"us", "usa", "united states", "united states of america"}
SYMBOLS = {"$": "USD", "£": "GBP", "€": "EUR"}
COUNTRY_CURRENCY = {"us": "USD", "usa": "USD", "united states": "USD", "united kingdom": "GBP",
                    "uk": "GBP", "great britain": "GBP", "deutschland": "EUR", "germany": "EUR"}
# Either a plain integer ("27") or a grouped number with exactly two decimals ("1.234,56").
# "1,234" or "12.345" without decimals is ambiguous across locales, so we refuse it.
_AMOUNT = re.compile(r"^-?(\d+|\d{1,3}([.,  ']?\d{3})*[.,]\d{2})$")


def parse_amount_cents(printed: str) -> int:
    """'$1,234.56' -> 123456, '1.234,56 €' -> 123456, '£8.55' -> 855, '27' -> 2700.

    The decimal separator is the '.' or ',' followed by exactly two final digits. Anything else
    (three decimals, grouped digits without decimals, letters, two numbers) raises ValueError.
    """
    s = re.sub(r"(USD|EUR|GBP|[$£€\s])", "", printed.strip())
    if not s or not _AMOUNT.match(s):
        raise ValueError(f"not an amount: {printed!r}")
    neg = s.startswith("-")
    s = s.lstrip("-")
    if len(s) > 3 and s[-3] in ".,":
        whole, frac = s[:-3], s[-2:]
    else:
        whole, frac = s, "00"
    whole = re.sub(r"[.,  ']", "", whole)
    cents = int(whole) * 100 + int(frac)
    return -cents if neg else cents


def parse_qty(printed: str) -> int:
    s = printed.strip()
    if not re.fullmatch(r"\d+", s):
        raise ValueError(f"not a whole quantity: {printed!r}")
    return int(s)


def parse_printed_date(printed: str, country: str) -> date:
    """Printed date -> date. Numeric a/b/yyyy is month-first only for US suppliers."""
    s = printed.strip().lower().replace(",", " ")
    if m := re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", s):
        return date(int(m[1]), int(m[2]), int(m[3]))
    if m := re.fullmatch(r"(\d{1,2})\.(\d{1,2})\.(\d{4})", s):
        return date(int(m[3]), int(m[2]), int(m[1]))
    if m := re.fullmatch(r"(\d{1,2})/(\d{1,2})/(\d{4})", s):
        a, b, y = int(m[1]), int(m[2]), int(m[3])
        month_first = country.strip().lower() in MONTH_FIRST
        return date(y, a, b) if month_first else date(y, b, a)
    parts = s.split()
    if len(parts) == 3 and parts[0] in MONTHS:  # March 14 2026
        return date(int(parts[2]), MONTHS[parts[0]], int(parts[1]))
    if len(parts) == 3 and parts[1].rstrip(".") in MONTHS:  # 14 March 2026 / 14. März 2026
        return date(int(parts[2]), MONTHS[parts[1].rstrip(".")], int(parts[0].rstrip(".")))
    raise ValueError(f"unrecognized date: {printed!r}")


def currency_of(p: PrintedInvoice) -> str:
    """ISO code from the symbols printed on the amounts, falling back to the supplier country."""
    found = {code for sym, code in SYMBOLS.items() if sym in p.total + p.subtotal}
    found |= {code for code in ("USD", "EUR", "GBP") if code in p.total + p.subtotal}
    if len(found) == 1:
        return found.pop()
    by_country = COUNTRY_CURRENCY.get(p.supplier_country.strip().lower())
    if by_country:
        return by_country
    raise ValueError(f"cannot tell the currency from {p.total!r} / {p.supplier_country!r}")


def to_invoice(p: PrintedInvoice) -> Invoice:
    """Deterministic conversion. Raises ValueError when a printed value can't be parsed."""
    return Invoice(
        supplier=p.supplier.strip(), invoice_no=p.invoice_no.strip(),
        date=parse_printed_date(p.invoice_date, p.supplier_country).isoformat(),
        currency=currency_of(p),
        line_items=[LineItem(sku=li.sku.strip(), desc=li.desc.strip(), qty=parse_qty(li.qty),
                             unit_price_cents=parse_amount_cents(li.unit_price))
                    for li in p.line_items],
        subtotal_cents=parse_amount_cents(p.subtotal), tax_cents=parse_amount_cents(p.tax),
        total_cents=parse_amount_cents(p.total))


def parse_printed(d: dict) -> Invoice:
    return to_invoice(PrintedInvoice.model_validate(d))
