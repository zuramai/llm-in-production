"""Question -> MetricPlan (LLM, structured output) -> compile -> run (read-only) -> phrase.

The model chooses from the catalog in metrics.yaml. It never writes SQL: the JSON schema it
must follow lists every metric and dimension name as an enum.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import openai
from pydantic import BaseModel, ConfigDict, ValidationError

from llmprod import chat_client, get_settings

from .compile import compile_query
from .db import DatabaseError, run_query
from .model import SemanticModel, default_model
from .query import MetricQuery, QueryError

PLAN_MAX_TOKENS = 4000
ANSWER_MAX_TOKENS = 1000
MAX_ROWS_IN_PROMPT = 50
OPS = ["=", "!=", "in", "not in", ">", ">=", "<", "<="]
PRESETS = ["today", "yesterday", "this_week", "last_week", "this_month", "last_month",
           "this_quarter", "last_quarter", "this_year", "last_year", "last_n", "previous_n",
           "custom", "all_time"]


class MetricPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assumptions: list[str]
    query: MetricQuery | None
    needs_clarification: str | None
    refusal: str | None


class SemanticAnswer(BaseModel):
    answer: str
    query: dict | None
    sql: str | None
    rows: list[dict]
    attempts: int
    model: str
    latency_ms: int
    usage: dict


def _nullable(schema: dict) -> dict:
    return {"anyOf": [schema, {"type": "null"}]}


def _obj(properties: dict) -> dict:
    return {"type": "object", "additionalProperties": False,
            "required": list(properties), "properties": properties}


def plan_schema(model: SemanticModel) -> dict:
    """Strict JSON schema for MetricPlan, with the catalog baked in as enums."""
    metric = {"type": "string", "enum": list(model.metrics)}
    dimension = {"type": "string", "enum": list(model.dimensions)}
    grain = {"type": "string", "enum": list(model.grains)}
    query = _obj({
        "metrics": {"type": "array", "items": metric},
        "dimensions": {"type": "array", "items": dimension},
        "filters": {"type": "array", "items": _obj({
            "dimension": dimension,
            "op": {"type": "string", "enum": OPS},
            "values": {"type": "array", "items": {"type": "string"}},
        })},
        "time_range": _nullable(_obj({
            "preset": {"type": "string", "enum": PRESETS},
            "unit": _nullable(grain),
            "n": _nullable({"type": "integer"}),
            "start": _nullable({"type": "string"}),
            "end": _nullable({"type": "string"}),
        })),
        "grain": _nullable(grain),
        "order_by": {"type": "array", "items": _obj({
            "field": {"type": "string",
                      "enum": [*model.metrics, *model.dimensions, *model.grains]},
            "descending": {"type": "boolean"},
        })},
        "limit": _nullable({"type": "integer"}),
    })
    return _obj({
        "assumptions": {"type": "array", "items": {"type": "string"}},
        "query": _nullable(query),
        "needs_clarification": _nullable({"type": "string"}),
        "refusal": _nullable({"type": "string"}),
    })


def render_catalog(model: SemanticModel) -> str:
    """The metric catalog as compact prompt text (about 1,500 tokens for ShopFlow)."""
    lines = ["## Metrics"]
    for name, m in model.metrics.items():
        entities = sorted({model.dimensions[d].entity for d in model.dimensions_for(name)})
        unit = f" [{m.unit}]" if m.unit else ""
        syn = f" Also called: {', '.join(m.synonyms)}." if m.synonyms else ""
        by = ", ".join(f"{e}.*" for e in entities) or "none"
        lines.append(f"- {name}{unit}: {m.description}{syn} Dimensions: {by}")
    lines += ["", "## Dimensions"]
    for name, d in model.dimensions.items():
        detail = " | ".join(d.values) if d.values else (d.description or d.type)
        lines.append(f"- {name}: {detail}")
    return "\n".join(lines)


PLANNER_RULES = """You answer questions from ShopFlow's ops and finance team by choosing \
governed metrics from a semantic layer. You never write SQL: you return a MetricQuery that \
code compiles.

Rules:
- Use only the metrics and dimensions listed below, with their definitions. Map synonyms.
- Only add filters, dimensions or time ranges the question asks for.
- time_range: today, yesterday, this_week, last_week, this_month, last_month, this_quarter,
  last_quarter, this_year, last_year; last_n (unit, n) = the current unit plus the n-1 before
  it; previous_n (unit, n) = the n complete units before the current one; custom = start and
  end dates (YYYY-MM-DD, inclusive); all_time; null = all time (or the metric's own default).
  Calendar periods are in {tz}; weeks start on Monday. Code computes the dates.
- grain buckets results by day, week, month, quarter or year ("per day", "weekly", "trend").
- order_by + limit for "top N" / "highest" questions.
- Read-only: never write, modify or delete data. If asked to, set `refusal`.
- Customer emails and other PII are not available. If asked for them, set `refusal`.
- If no metric measures what is asked, set `refusal` and say what is missing. Never answer a
  different question with a near-miss metric.
- If the question is ambiguous, pick the closest governed metric and record the choice in
  `assumptions`. Use `needs_clarification` only when no reasonable default exists.
- Exactly one of `query`, `needs_clarification`, `refusal` is non-null.

{catalog}

## Examples
Q: Revenue by channel last month
{{"metrics": ["gross_revenue"], "dimensions": ["order.channel"], "filters": [], \
"time_range": {{"preset": "last_month", "unit": null, "n": null, "start": null, "end": null}}, \
"grain": null, "order_by": [], "limit": null}}
Q: Daily refunds this week
{{"metrics": ["refund_count"], "dimensions": [], "filters": [], "time_range": {{"preset": \
"this_week", "unit": null, "n": null, "start": null, "end": null}}, "grain": "day", \
"order_by": [], "limit": null}}
Q: Top 3 products by units sold this year
{{"metrics": ["units_sold"], "dimensions": ["product.name"], "filters": [], "time_range": \
{{"preset": "this_year", "unit": null, "n": null, "start": null, "end": null}}, "grain": null, \
"order_by": [{{"field": "units_sold", "descending": true}}], "limit": 3}}
Q: Orders from VIP customers in Canada over the last 2 complete quarters
{{"metrics": ["transactions"], "dimensions": [], "filters": [{{"dimension": "customer.segment", \
"op": "=", "values": ["vip"]}}, {{"dimension": "customer.country", "op": "=", "values": \
["CA"]}}], "time_range": {{"preset": "previous_n", "unit": "quarter", "n": 2, "start": null, \
"end": null}}, "grain": null, "order_by": [], "limit": null}}
(These show the `query` field only; the reply wraps it in a MetricPlan.)"""

ANSWER_RULES = """You answer questions from ShopFlow's ops team in Slack, using ONLY the query \
result below.

- State numbers exactly as given; never invent, estimate or round away numbers.
- Metric values arrive already formatted as strings (dollars, percentages, counts). Copy them
  exactly. Never write a number that is not in the rows.
- Name each metric in the business's words and say briefly how it is defined (the definitions
  are given). Mention any assumptions.
- If the result is empty, say that no data matched.
- End time-sensitive answers with "as of {as_of}".
- Reply in 1-3 sentences, or a short list for multi-row results. No SQL, no preamble."""


def _merge_usage(total: dict, usage) -> None:
    if usage is None:
        return
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        total[key] = total.get(key, 0) + (getattr(usage, key, 0) or 0)
    cost = getattr(usage, "cost", None) or (usage.model_extra or {}).get("cost")
    if cost:
        total["cost_usd"] = total.get("cost_usd", 0.0) + float(cost)


def as_of(now: datetime) -> str:
    """'8:26 AM EST'. Built by hand: strftime('%p') is empty under some server locales."""
    return f"{now:%-I:%M} {'AM' if now.hour < 12 else 'PM'} {now.tzname()}"


def format_rows(rows: list[dict], sem: SemanticModel) -> list[dict]:
    """Format metric values in code, so the phrasing model copies strings, not math.

    Left to the model, a ratio of 0.030075... came back as "3.007518796992481%".
    """
    def fmt(name: str, value):
        metric = sem.metrics.get(name)
        if metric is None or not isinstance(value, (int, float)) or isinstance(value, bool):
            return value
        if metric.unit == "usd":
            return f"${value:,.2f}"
        if metric.unit == "ratio":
            return f"{value:.2%}"
        return f"{value:,.0f}" if float(value).is_integer() else f"{value:,.2f}"

    return [{k: fmt(k, v) for k, v in row.items()} for row in rows]


def answer(
    question: str,
    *,
    now: datetime | None = None,
    model: str | None = None,
    semantic_model: SemanticModel | None = None,
    max_attempts: int = 3,
    client: openai.OpenAI | None = None,
    reasoning_effort: str | None = None,
) -> SemanticAnswer:
    started = time.perf_counter()
    settings = get_settings()
    sem = semantic_model or default_model()
    tz = sem.timezone
    model = model or settings.llm_model
    client = client or chat_client()
    now = (now or datetime.now(UTC)).astimezone(ZoneInfo(tz))
    usage: dict = {}
    options = {"reasoning_effort": reasoning_effort} if reasoning_effort else {}
    fmt = {"type": "json_schema", "json_schema": {"name": "MetricPlan", "strict": True,
                                                   "schema": plan_schema(sem)}}

    def done(text: str, query: MetricQuery | None = None, sql: str | None = None,
             rows: list[dict] | None = None, attempts: int = 0) -> SemanticAnswer:
        return SemanticAnswer(
            answer=text, query=query.model_dump() if query else None, sql=sql, rows=rows or [],
            attempts=attempts, model=model, usage=usage,
            latency_ms=round((time.perf_counter() - started) * 1000))

    messages = [
        {"role": "system", "content": PLANNER_RULES.format(tz=tz, catalog=render_catalog(sem))},
        {"role": "user", "content": f"Current time: {now.isoformat()} ({tz})\n"
                                    f"Question: {question}\n"
                                    "Reply with a MetricPlan JSON object."},
    ]
    error, sql, plan = "", None, None
    for attempt in range(1, max_attempts + 1):
        resp = client.chat.completions.create(model=model, messages=messages,
                                              response_format=fmt,
                                              max_tokens=PLAN_MAX_TOKENS, **options)
        _merge_usage(usage, resp.usage)
        content = resp.choices[0].message.content or ""
        messages.append({"role": "assistant", "content": content})
        try:
            plan = MetricPlan.model_validate_json(content)
        except ValidationError as e:
            error = f"Your reply was not a valid MetricPlan: {e.errors()[:3]}"
            messages.append({"role": "user", "content": error})
            continue
        if plan.refusal:
            return done(plan.refusal, attempts=attempt)
        if plan.needs_clarification:
            return done(plan.needs_clarification, attempts=attempt)
        if plan.query is None:
            error = "Set exactly one of query, needs_clarification or refusal."
            messages.append({"role": "user", "content": error})
            continue
        try:
            sql = compile_query(plan.query, model=sem, now=now)
            rows = run_query(sql)
            break
        except (QueryError, DatabaseError) as e:
            error = str(e)
            messages.append({"role": "user", "content": f"That MetricQuery is invalid: {error}\n"
                                                        "Fix it and reply with a new MetricPlan."})
    else:
        return done(f"Sorry, I couldn't answer that: {error}",
                    query=plan.query if plan else None, sql=sql, attempts=max_attempts)

    q = plan.query
    phrase = client.chat.completions.create(
        model=model,
        max_tokens=ANSWER_MAX_TOKENS,
        **options,
        messages=[
            {"role": "system", "content": ANSWER_RULES.format(as_of=as_of(now))},
            {"role": "user", "content": json.dumps({
                "question": question,
                "metrics": {n: {"unit": sem.metrics[n].unit or "count",
                                "definition": sem.metrics[n].description} for n in q.metrics},
                "assumptions": plan.assumptions,
                "row_count": len(rows),
                "rows": format_rows(rows[:MAX_ROWS_IN_PROMPT], sem),
            }, default=str)},
        ],
    )
    _merge_usage(usage, phrase.usage)
    return done((phrase.choices[0].message.content or "").strip(), query=q, sql=sql, rows=rows,
                attempts=attempt)
