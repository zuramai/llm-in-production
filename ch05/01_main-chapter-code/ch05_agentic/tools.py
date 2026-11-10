"""The two tools every variant shares: search the documents, query the database.

`search_docs` is Chapter 4's hybrid retrieval + reranker. `run_sql` is Chapter 6's guard + the
read-only role. Evidence carries a citation tag (`[doc:path#N]` or `[sql:N]`) so answers can be
checked against what was actually retrieved.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict

from llmprod import get_settings

from .llm import LLM, Usage
from .prev import prev

ALLOWED_TABLES = frozenset({
    "orders", "order_items", "products", "categories", "customers", "payments", "refunds",
    "shipments", "inventory", "support_tickets",
})
MAX_ROWS_SHOWN = 20
SQL_MAX_TOKENS = 1500

# A condensed version of Chapter 6's semantic_context.yaml, with the tables this chapter needs.
SCHEMA = """PostgreSQL database of ShopFlow, a US online shop. Business time zone: {tz}.
Conventions:
- Money columns are INTEGER CENTS (4599 = $45.99). Return dollars as cents/100.0 named *_usd.
- Timestamps are timestamptz. Return them in business time, (ts AT TIME ZONE '{tz}') AS
  <name>_local, plus the days elapsed until today:
  (now() AT TIME ZONE '{tz}')::date - (ts AT TIME ZONE '{tz}')::date AS days_since_<name>.
- customers.email is PII and not readable. Never select it.
Tables (id is the primary key of each):
- customers(id, full_name, country ISO-2, city, segment 'consumer'|'business'|'vip', created_at)
- orders(id, customer_id, status, channel, subtotal_cents, discount_cents, shipping_cents,
  total_cents = subtotal - discount + shipping, placed_at)
  status: pending (no successful payment yet) | paid (not shipped yet) | shipped (in transit) |
  delivered | cancelled (cancelled after payment, refunded) | refunded (returned after delivery)
- order_items(order_id, product_id, quantity, unit_price_cents)
- products(id, sku, name, category_id, price_cents, active)
- categories(id, name)  e.g. Electronics, Beauty, Grocery, Apparel, Home & Kitchen
- payments(order_id, provider, status 'succeeded'|'failed', amount_cents, created_at)
- refunds(order_id, amount_cents, reason, created_at)
- shipments(order_id, carrier, status 'in_transit'|'delivered', shipped_at, delivered_at)
- support_tickets(id, customer_id, order_id, channel, subject, body, created_at,
  label_category, label_priority)
- inventory(product_id, warehouse, on_hand)"""


@dataclass
class DocEvidence:
    citation: str  # [doc:path#N]
    doc_path: str
    heading: str
    content: str
    query: str

    def render(self) -> str:
        head = f" ({self.heading})" if self.heading else ""
        return f"{self.citation}{head}\n{self.content}"


@dataclass
class SqlEvidence:
    citation: str  # [sql:N]
    question: str
    sql: str | None
    rows: list[dict]
    error: str | None = None

    def render(self) -> str:
        if self.error:
            return f"{self.citation} for: {self.question}\nSQL: {self.sql}\nERROR: {self.error}"
        shown = json.dumps(self.rows[:MAX_ROWS_SHOWN], default=str)
        more = f" (+{len(self.rows) - MAX_ROWS_SHOWN} more rows)" if len(
            self.rows) > MAX_ROWS_SHOWN else ""
        return f"{self.citation} for: {self.question}\nSQL: {self.sql}\nrows: {shown}{more}"


def evidence_dict(e: DocEvidence | SqlEvidence) -> dict:
    return {"kind": "doc" if isinstance(e, DocEvidence) else "sql", **asdict(e)}


def evidence_from_dict(d: dict) -> DocEvidence | SqlEvidence:
    d = dict(d)
    kind = d.pop("kind")
    return DocEvidence(**d) if kind == "doc" else SqlEvidence(**d)


# ------------------------------------------------------------------------------------- docs
class DocSearch:
    """Chapter 4's best configuration: heading chunks, hybrid search, cross-encoder rerank."""

    def __init__(self, k: int = 4):
        self.retriever = prev.Retriever(prev.open_db())
        self.k = k

    def search(self, query: str, k: int | None = None) -> list[DocEvidence]:
        hits = self.retriever.retrieve(query, "heading", k or self.k)
        return [DocEvidence(h.citation, h.doc_path, h.heading_path, h.content, query)
                for h in hits]


# -------------------------------------------------------------------------------------- sql
def run_sql(sql: str) -> list[dict]:
    """Guard first (raises prev.UnsafeSQLError), then run as the read-only role
    (raises prev.QueryError)."""
    return prev.run_query(prev.validate_sql(sql, set(ALLOWED_TABLES)))


SQL_RULES = """You write ONE PostgreSQL SELECT query that fetches the facts needed to answer a \
question from ShopFlow's support team. Read-only: never write data. Select every column that \
could matter (status, money columns, dates and days_since_*). Join what you need.

{schema}

Today is {today}."""


class SqlPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sql: str


def ask_sql(question: str, llm: LLM, usage: Usage, citation: str, today: str,
            max_attempts: int = 2) -> SqlEvidence:
    """Text-to-SQL for one data question, with Chapter 6's self-correction: guard and Postgres
    errors go back to the model, up to `max_attempts` plans."""
    tz = get_settings().business_timezone
    messages = [
        {"role": "system", "content": SQL_RULES.format(schema=SCHEMA.format(tz=tz), today=today)},
        {"role": "user", "content": f"Data needed: {question}\n"
                                    "Reply with a JSON object with key sql."},
    ]
    sql, error = None, None
    for _ in range(max_attempts):
        try:
            # Wide SELECTs are long: 800 tokens cut one off mid-string in the eval.
            plan = llm.structured(messages, SqlPlan, usage, "sql_plan", max_tokens=SQL_MAX_TOKENS)
        except ValueError as e:  # still not valid JSON after one retry: evidence, not a crash
            return SqlEvidence(citation, question, sql, [], f"planner reply invalid: {e}"[:300])
        sql = plan.sql
        try:
            rows = run_sql(sql)
            return SqlEvidence(citation, question, sql, rows)
        except (prev.UnsafeSQLError, prev.QueryError) as e:
            error = str(e)
            messages += [{"role": "assistant", "content": plan.model_dump_json()},
                         {"role": "user", "content": f"The query failed: {error}\nFix it."}]
    return SqlEvidence(citation, question, sql, [], error)


# -------------------------------------------------------------------------------- citations
_DOC = re.compile(r"\[doc:([^\]#\s]+)#(\d+)\]")
_SQL = re.compile(r"\[sql:(\d+)\]")


def cited(text: str) -> list[str]:
    """Citation tags in an answer, normalised, in order of first appearance."""
    tags = [f"[doc:{m.group(1)}#{m.group(2)}]" for m in _DOC.finditer(text)]
    tags += [f"[sql:{m.group(1)}]" for m in _SQL.finditer(text)]
    order = {t: text.find(t) for t in tags}
    return sorted(dict.fromkeys(tags), key=lambda t: order[t])


def check_citations(text: str, evidence: list[DocEvidence | SqlEvidence]) -> dict:
    """Which citations point at evidence the model actually saw? Invented tags are a red flag."""
    known = {e.citation for e in evidence}
    tags = cited(text)
    return {"citations": tags, "invalid": [t for t in tags if t not in known],
            "cites_doc": any(t.startswith("[doc:") for t in tags if t in known),
            "cites_sql": any(t.startswith("[sql:") for t in tags if t in known)}


_WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


def now_text(now: datetime | None = None) -> str:
    """'Saturday 2026-11-07 08:30 (America/New_York)'. Not strftime('%A'): that follows the
    machine's locale, and this machine's is Indonesian."""
    tz = get_settings().business_timezone
    now = (now or datetime.now(ZoneInfo(tz))).astimezone(ZoneInfo(tz))
    return f"{_WEEKDAYS[now.weekday()]} {now:%Y-%m-%d %H:%M} ({tz})"


def timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, (time.perf_counter() - t0) * 1000
