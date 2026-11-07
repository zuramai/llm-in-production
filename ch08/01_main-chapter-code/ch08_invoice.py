"""The Invoice data model, its arithmetic validation rules, and field-level scoring.

Everything here is pure Python: no network, no database. That is deliberate. The validation rules
are the part of the pipeline we trust, so they must be testable without a model.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date, timedelta

from pydantic import BaseModel, ConfigDict, Field

# ISO 4217 codes we accept. A real AP system would load the full list (about 180 codes); the
# point of the rule is that "US$" or "dollars" is not a currency code.
ISO_CURRENCIES = frozenset(
    ["USD", "EUR", "GBP", "CAD", "AUD", "NZD", "CHF", "SEK", "NOK", "DKK", "PLN", "CZK", "HUF", "JPY", "CNY", "HKD", "SGD", "INR", "IDR", "MYR", "THB", "KRW", "MXN", "BRL", "ZAR", "AED", "TRY"]
)
MIN_PLAUSIBLE_DATE = date(2000, 1, 1)
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class LineItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sku: str = Field(description="Supplier's article/SKU code, copied exactly as printed")
    desc: str = Field(description="Line description as printed")
    qty: int = Field(description="Quantity ordered, a whole number")
    unit_price_cents: int = Field(description="Price of ONE unit in integer cents (1.50 -> 150)")


class Invoice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    supplier: str = Field(description="Legal name of the company that ISSUED the invoice")
    invoice_no: str = Field(description="The invoice number, copied exactly as printed")
    date: str = Field(description="Invoice (issue) date as YYYY-MM-DD. Not the due date.")
    currency: str = Field(description="ISO 4217 code, e.g. USD, EUR, GBP")
    line_items: list[LineItem]
    subtotal_cents: int = Field(description="Sum of all lines before tax, integer cents")
    tax_cents: int = Field(description="Total tax amount, integer cents")
    total_cents: int = Field(description="Amount due including tax, integer cents")


def strict_schema(model: type[BaseModel]) -> dict:
    """JSON Schema for OpenAI-style strict structured outputs.

    Pydantic emits `$defs`/`$ref` for nested models and omits `additionalProperties`. Strict
    mode wants every object closed and every property required, so we inline refs and patch.
    """
    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})

    def fix(node):
        if isinstance(node, dict):
            if "$ref" in node:
                return fix(defs[node["$ref"].split("/")[-1]])
            node = {k: fix(v) for k, v in node.items() if k != "title"}
            if node.get("type") == "object":
                node["additionalProperties"] = False
                node["required"] = list(node.get("properties", {}))
            return node
        if isinstance(node, list):
            return [fix(v) for v in node]
        return node

    return fix(schema)


# ---------------------------------------------------------------------------------------------
# Validation: arithmetic, not trust
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Violation:
    rule: str
    message: str

    def __str__(self) -> str:
        return f"{self.rule}: {self.message}"


def validate_invoice(inv: Invoice, today: date | None = None) -> list[Violation]:
    """Return every rule the invoice breaks. An empty list means 'internally consistent'."""
    today = today or date.today()
    out: list[Violation] = []

    if not inv.supplier.strip() or not inv.invoice_no.strip():
        out.append(Violation("required_fields", "supplier and invoice_no must be non-empty"))

    if not inv.line_items:
        out.append(Violation("has_lines", "an invoice needs at least one line item"))
    for i, li in enumerate(inv.line_items, start=1):
        if li.qty < 1 or li.unit_price_cents < 0:
            out.append(Violation(
                "line_values", f"line {i}: qty={li.qty}, unit_price_cents={li.unit_price_cents}"))

    line_sum = sum(li.qty * li.unit_price_cents for li in inv.line_items)
    if line_sum != inv.subtotal_cents:
        out.append(Violation(
            "line_sum",
            f"sum(qty*unit_price_cents) = {line_sum} but subtotal_cents = {inv.subtotal_cents}"))

    if inv.tax_cents < 0:
        out.append(Violation("tax_negative", f"tax_cents = {inv.tax_cents}"))
    if inv.subtotal_cents + inv.tax_cents != inv.total_cents:
        out.append(Violation(
            "total_sum",
            f"subtotal_cents + tax_cents = {inv.subtotal_cents + inv.tax_cents} "
            f"but total_cents = {inv.total_cents}"))

    d = parse_iso_date(inv.date)
    if d is None:
        out.append(Violation("date_parse", f"{inv.date!r} is not a valid YYYY-MM-DD date"))
    elif not (MIN_PLAUSIBLE_DATE <= d <= today + timedelta(days=1)):
        out.append(Violation("date_range", f"{inv.date} is outside {MIN_PLAUSIBLE_DATE}..today"))

    if inv.currency not in ISO_CURRENCIES:
        out.append(Violation("currency_iso", f"{inv.currency!r} is not an accepted ISO 4217 code"))
    return out


def parse_iso_date(s: str) -> date | None:
    if not _ISO_DATE.match(s):
        return None
    try:
        return date.fromisoformat(s)
    except ValueError:
        return None


def violations_to_text(violations: list[Violation]) -> str:
    return "; ".join(str(v) for v in violations)


# ---------------------------------------------------------------------------------------------
# Scoring: field-level accuracy against ground truth
# ---------------------------------------------------------------------------------------------

HEADER_FIELDS = ("supplier", "invoice_no", "date", "currency",
                 "subtotal_cents", "tax_cents", "total_cents")
SCORE_FIELDS = (*HEADER_FIELDS, "line_items")


def _norm_name(s: str) -> str:
    return re.sub(r"[^0-9a-zà-ÿ]", "", s.casefold())


def _item_key(li: LineItem) -> tuple[str, int, int]:
    return (re.sub(r"\s", "", li.sku).upper(), li.qty, li.unit_price_cents)


def score_invoice(pred: Invoice | None, truth: Invoice) -> dict[str, float]:
    """Per-field correctness (1.0 / 0.0) plus `line_item_recall` and `all`.

    A line item counts as correct when (sku, qty, unit_price_cents) all match, because those
    are the fields that move money. `line_items` is 1.0 only if every line matches, with no
    extras. `all` means the whole invoice is right. A failed extraction scores 0 everywhere.
    """
    zero = {f: 0.0 for f in SCORE_FIELDS} | {"line_item_recall": 0.0, "all": 0.0}
    if pred is None:
        return zero
    s: dict[str, float] = {}
    s["supplier"] = float(_norm_name(pred.supplier) == _norm_name(truth.supplier))
    s["invoice_no"] = float(pred.invoice_no.strip() == truth.invoice_no.strip())
    s["date"] = float(pred.date == truth.date)
    s["currency"] = float(pred.currency == truth.currency)
    for f in ("subtotal_cents", "tax_cents", "total_cents"):
        s[f] = float(getattr(pred, f) == getattr(truth, f))

    want = sorted(_item_key(li) for li in truth.line_items)
    got = sorted(_item_key(li) for li in pred.line_items)
    remaining = list(got)
    hits = 0
    for k in want:
        if k in remaining:
            remaining.remove(k)
            hits += 1
    s["line_item_recall"] = hits / len(want) if want else 1.0
    s["line_items"] = float(want == got)
    s["all"] = float(all(s[f] == 1.0 for f in SCORE_FIELDS))
    return s


def extract_json(text: str) -> dict:
    """Parse the outermost {...} in `text`. Models sometimes wrap JSON in prose or fences."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found in model output")
    return json.loads(text[start : end + 1])
