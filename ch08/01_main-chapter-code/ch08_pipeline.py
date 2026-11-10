"""Route -> extract -> validate -> repair -> escalate -> accept or send to human review.

`process_file` is pure with respect to the database (it only calls models), so evals can run it
in a thread pool. `record` writes the outcome to Postgres.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import psycopg
from ch08_extract import (
    SYSTEM_RULES,
    Extraction,
    call_model,
    has_text_layer,
    image_messages,
    pdf_text,
    repair_messages,
    text_messages,
)
from ch08_invoice import Invoice, Violation, validate_invoice, violations_to_text
from ch08_normalize import PRINTED_FORMAT, PRINTED_RULES, parse_printed
from openai import OpenAI

SETUP_SQL = """
CREATE TABLE IF NOT EXISTS ch08_invoices (
    id            serial PRIMARY KEY,
    source_file   text NOT NULL,
    supplier      text NOT NULL,
    invoice_no    text NOT NULL,
    invoice_date  date NOT NULL,
    currency      char(3) NOT NULL,
    subtotal_cents integer NOT NULL,
    tax_cents     integer NOT NULL,
    total_cents   integer NOT NULL,
    line_items    jsonb NOT NULL,
    method        text NOT NULL,
    created_at    timestamptz NOT NULL DEFAULT now(),
    UNIQUE (supplier, invoice_no)
);
CREATE TABLE IF NOT EXISTS ch08_invoice_review (
    id          serial PRIMARY KEY,
    source_file text NOT NULL,
    method      text,
    rules       text[] NOT NULL,
    reason      text NOT NULL,
    extracted   jsonb,
    status      text NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'approved', 'rejected')),
    created_at  timestamptz NOT NULL DEFAULT now()
);
"""


@dataclass
class Attempt:
    stage: str  # first | repair | escalate
    extraction: Extraction
    violations: list[Violation]


@dataclass
class Outcome:
    file: str
    method: str  # text | vision | unsupported
    status: str  # accepted | review
    invoice: Invoice | None
    attempts: list[Attempt] = field(default_factory=list)
    reasons: list[Violation] = field(default_factory=list)

    @property
    def cost_usd(self) -> float:
        return sum(a.extraction.cost_usd for a in self.attempts)

    @property
    def latency_s(self) -> float:
        return sum(a.extraction.latency_s for a in self.attempts)


def route(path: Path) -> tuple[str, str]:
    """(method, text). PDFs with a text layer go the cheap way; images need vision."""
    if path.suffix.lower() == ".pdf":
        text = pdf_text(path)
        if has_text_layer(text):
            return "text", text
        return "unsupported", ""  # image-only PDF: would need a rasterizer (see the chapter)
    return "vision", ""


def _check(ex: Extraction, today: date | None) -> list[Violation]:
    if ex.invoice is None:
        return [Violation("parse_error", ex.error or "no invoice produced")]
    return validate_invoice(ex.invoice, today)


@dataclass(frozen=True)
class ModelSpec:
    """Which model to call, through which client, with which extra arguments.

    `printed=True` switches to copy-then-convert: the model returns values as printed
    (`PrintedInvoice`) and `ch08_normalize` turns them into cents and ISO dates.
    """

    client: OpenAI
    model: str
    kwargs: dict = field(default_factory=dict)  # e.g. {"reasoning_effort": "none"} for qwen3
    max_side: int | None = None  # downscale page images for this model (vision only)
    printed: bool = False

    def messages(self, method: str, path: Path, text: str) -> list[dict]:
        system = PRINTED_RULES if self.printed else SYSTEM_RULES
        return (text_messages(text, system=system) if method == "text"
                else image_messages(path, system=system, max_side=self.max_side))

    def extract(self, messages: list[dict], method: str) -> Extraction:
        if self.printed:
            return call_model(self.client, self.model, messages, method=method,
                              response_format=PRINTED_FORMAT, parse=parse_printed,
                              **self.kwargs)
        return call_model(self.client, self.model, messages, method=method, **self.kwargs)


def process_file(path: Path, *, text: ModelSpec, vision: ModelSpec | None = None,
                 strong: ModelSpec | None = None, repair: bool = True,
                 escalate_fresh: bool = True, today: date | None = None) -> Outcome:
    """Run one file through the pipeline.

    attempt 1: `text` model for PDFs with a text layer, `vision` model for images.
    If a validation rule fails:
    attempt 2: same model, shown its own answer and the broken rules (`repair`). If still bad:
    attempt 3: `strong` model. With `escalate_fresh` (default) it reads the document from
    scratch; otherwise it gets the repair conversation, weak answer included. If still bad:
    human review.
    """
    method, page_text = route(path)
    if method == "unsupported":
        return Outcome(path.name, method, "review", None,
                       reasons=[Violation("no_text_layer", "image-only PDF; needs vision")])
    first_spec = text if method == "text" else (vision or text)
    plan = [("first", first_spec)]
    if repair:
        plan.append(("repair", first_spec))
    if strong:
        plan.append(("escalate", strong))

    attempts: list[Attempt] = []
    for stage, spec in plan:
        messages = spec.messages(method, path, page_text)
        if attempts and not (stage == "escalate" and escalate_fresh):
            prev = attempts[-1]
            messages = repair_messages(messages, prev.extraction.raw,
                                       violations_to_text(prev.violations))
        ex = spec.extract(messages, method)
        attempts.append(Attempt(stage, ex, _check(ex, today)))
        if not attempts[-1].violations:
            return Outcome(path.name, method, "accepted", ex.invoice, attempts)

    last = attempts[-1]
    return Outcome(path.name, method, "review", last.extraction.invoice, attempts,
                   reasons=last.violations)


# ---------------------------------------------------------------------------------------------
# Postgres
# ---------------------------------------------------------------------------------------------


def setup_tables(conn: psycopg.Connection, reset: bool = False) -> None:
    """Idempotent. `reset=True` empties our own two tables (nothing else is touched)."""
    conn.execute(SETUP_SQL)
    if reset:
        conn.execute("TRUNCATE ch08_invoices, ch08_invoice_review RESTART IDENTITY")


def record(conn: psycopg.Connection, outcome: Outcome) -> str:
    """Persist an outcome. Returns 'accepted' or 'review'. Duplicates go to review."""
    inv = outcome.invoice
    if outcome.status == "accepted" and inv is not None:
        try:
            with conn.transaction():
                conn.execute(
                    """INSERT INTO ch08_invoices (source_file, supplier, invoice_no, invoice_date,
                           currency, subtotal_cents, tax_cents, total_cents, line_items, method)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                    (outcome.file, inv.supplier, inv.invoice_no, inv.date, inv.currency,
                     inv.subtotal_cents, inv.tax_cents, inv.total_cents,
                     json.dumps([li.model_dump() for li in inv.line_items]), outcome.method))
            return "accepted"
        except psycopg.errors.UniqueViolation:
            outcome.reasons = [Violation(
                "duplicate_invoice", f"{inv.supplier} / {inv.invoice_no} is already booked")]
    reasons = outcome.reasons or [Violation("unknown", "rejected without a reason")]
    conn.execute(
        "INSERT INTO ch08_invoice_review (source_file, method, rules, reason, extracted) "
        "VALUES (%s,%s,%s,%s,%s)",
        (outcome.file, outcome.method, [r.rule for r in reasons], violations_to_text(reasons),
         json.dumps(inv.model_dump()) if inv else None))
    return "review"
