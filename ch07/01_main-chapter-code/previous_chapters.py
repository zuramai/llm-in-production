"""Code from earlier chapters that this chapter reuses, copied so ch07 can be read on its own.

- Chapter 1 (ch01/01_main-chapter-code/ch01.ipynb, "Cost napkin math"): PRICES, estimate_cost.
- Chapter 6 (ch06/01_main-chapter-code/sqlbot/): the v1 text-to-SQL bot (semantic context,
  structured SqlPlan, AST guard, read-only role, self-correction, phrasing). Chapter 7 uses it
  as the baseline the semantic layer is compared against. Logic is unchanged; the three
  modules (guard.py, db.py, agent.py) are merged into this file.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from zoneinfo import ZoneInfo

import openai
import psycopg
import sqlglot
import yaml
from psycopg.rows import dict_row
from pydantic import BaseModel, ConfigDict, ValidationError
from sqlglot import exp
from sqlglot.errors import ParseError

from llmprod import chat_client, connect, get_settings

# ---- Chapter 1 ---------------------------------------------------------------------------
# $ per 1M tokens (input, output) on OpenRouter, as of 2026-11-07.
PRICES = {
    "openai/gpt-6-luna": (0.10, 0.50),
    "anthropic/claude-haiku-5.5": (0.10, 0.50),
    "anthropic/claude-sonnet-5.5": (2.00, 10.00),
    "qwen/qwen3.8-flash": (0.15, 0.47),
}


def estimate_cost(prompt_tokens: int, completion_tokens: int,
                  usd_per_m_in: float, usd_per_m_out: float) -> float:
    return (prompt_tokens * usd_per_m_in + completion_tokens * usd_per_m_out) / 1_000_000


# ---- Chapter 6: guard.py -----------------------------------------------------------------
BLOCKED_FUNCTIONS = frozenset({
    "pg_sleep", "pg_read_file", "pg_read_binary_file", "pg_ls_dir", "pg_stat_file",
    "lo_import", "lo_export", "dblink", "dblink_exec", "copy", "pg_terminate_backend",
    "pg_cancel_backend", "set_config", "pg_reload_conf",
})
WRITE_NODES = (
    exp.Insert, exp.Update, exp.Delete, exp.Merge, exp.Drop, exp.Create, exp.Alter,
    exp.TruncateTable, exp.Command, exp.Copy, exp.Grant,
)


class UnsafeSQLError(ValueError):
    """The SQL breaks a guard rule. The message is safe to show to the model."""


def _function_name(node: exp.Func) -> str:
    if isinstance(node, exp.Anonymous):
        return node.name.lower()
    return node.sql_name().lower()


def validate_sql(sql: str, allowed_tables: set[str], max_limit: int = 1000) -> str:
    """Return normalized, LIMIT-capped Postgres SQL, or raise UnsafeSQLError."""
    try:
        statements = [s for s in sqlglot.parse(sql, dialect="postgres") if s is not None]
    except ParseError as e:
        raise UnsafeSQLError(f"rule=parse: SQL does not parse as Postgres: {e}") from None
    if len(statements) != 1:
        raise UnsafeSQLError(
            f"rule=single_statement: expected exactly 1 statement, got {len(statements)}")
    tree = statements[0]

    if not isinstance(tree, (exp.Select, exp.SetOperation)):
        raise UnsafeSQLError(
            f"rule=select_only: only SELECT queries are allowed, got {tree.key.upper()}")
    for node in tree.walk():
        if isinstance(node, WRITE_NODES):
            raise UnsafeSQLError(
                f"rule=select_only: {node.key.upper()} is not allowed anywhere in the query")
        if isinstance(node, exp.Select) and node.args.get("into"):
            raise UnsafeSQLError("rule=no_select_into: SELECT ... INTO creates a table")
        if isinstance(node, exp.Func) and _function_name(node) in BLOCKED_FUNCTIONS:
            raise UnsafeSQLError(
                f"rule=blocked_function: function {_function_name(node)}() is not allowed")

    cte_names = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
    allowed = {t.lower() for t in allowed_tables} | cte_names
    for table in tree.find_all(exp.Table):
        name = table.name.lower()
        schema = table.db.lower()
        if not name or schema not in ("", "public") or name not in allowed:
            shown = f"{schema}.{name}" if schema else (name or table.sql())
            raise UnsafeSQLError(
                f"rule=allowed_tables: table {shown} is not allowed; "
                f"use only: {', '.join(sorted(allowed_tables))}")

    limit = tree.args.get("limit")
    if limit is None:
        tree = tree.limit(max_limit)
    else:
        value = limit.expression
        if not (isinstance(value, exp.Literal) and value.is_int):
            raise UnsafeSQLError("rule=limit: LIMIT must be an integer literal")
        if int(value.this) > max_limit:
            tree = tree.limit(max_limit)
    return tree.sql(dialect="postgres")


# ---- Chapter 6: db.py --------------------------------------------------------------------
class QueryError(Exception):
    """Postgres rejected or failed the query. The message is fed back to the model."""


def _jsonable(value):
    from datetime import date
    from decimal import Decimal

    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def run_query(sql: str) -> list[dict]:
    try:
        with connect(readonly=True) as conn:
            conn.execute("SET statement_timeout = '5s'")
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(sql)
                rows = cur.fetchmany(1000)
    except psycopg.Error as e:
        raise QueryError(str(e).strip()) from None
    return [{k: _jsonable(v) for k, v in row.items()} for row in rows]


# ---- Chapter 6: agent.py -----------------------------------------------------------------
CONTEXT_PATH = Path(__file__).with_name("ch06_semantic_context.yaml")
MAX_ROWS_IN_PROMPT = 50
PLAN_MAX_TOKENS = 4000
ANSWER_MAX_TOKENS = 1000


class SqlPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assumptions: list[str]
    sql: str | None
    needs_clarification: str | None
    refusal: str | None


class SqlAnswer(BaseModel):
    answer: str
    sql: str | None
    rows: list[dict]
    attempts: int
    model: str
    latency_ms: int
    usage: dict


def _strict_schema(model: type[BaseModel]) -> dict:
    schema = model.model_json_schema()
    schema["additionalProperties"] = False
    schema["required"] = list(schema["properties"])
    return schema


PLAN_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "SqlPlan", "strict": True, "schema": _strict_schema(SqlPlan)},
}


@cache
def load_context(tz: str) -> tuple[str, frozenset[str]]:
    """Render ch06_semantic_context.yaml into prompt text, with the allowed table names."""
    ctx = yaml.safe_load(CONTEXT_PATH.read_text().replace("{tz}", tz))
    lines = ["## Conventions", *(f"- {c}" for c in ctx["conventions"]), "", "## Tables"]
    for table, spec in ctx["tables"].items():
        lines.append(f"### {table}: {spec['description']}")
        lines += [f"- {col}: {desc}" for col, desc in spec["columns"].items()]
    lines += ["", "## Joins", *(f"- {j}" for j in ctx["joins"]), "", "## Glossary"]
    lines += [f"- {term}: {definition}" for term, definition in ctx["glossary"].items()]
    lines += ["", "## Examples"]
    for ex in ctx["examples"]:
        lines += [f"Q: {ex['question']}", "```sql", ex["sql"].strip(), "```"]
    return "\n".join(lines), frozenset(ctx["tables"])


PLANNER_RULES = """You translate questions from ShopFlow's ops and finance team into one \
PostgreSQL SELECT query.

Rules:
- Use only the tables and columns documented below. Follow the glossary definitions exactly.
- Read-only: never write, modify or delete data. If asked to, set `refusal`.
- Customer emails and other PII are not available. If asked for them, set `refusal`.
- If the data needed is not in these tables, set `refusal` and say what is missing.
- If the question is ambiguous, pick the glossary definition or the most common business
  meaning, write the SQL, and record the choice in `assumptions`. Use `needs_clarification`
  only when no reasonable default exists.
- Name output columns clearly; money columns as *_usd (already divided by 100.0).
- Exactly one of `sql`, `needs_clarification`, `refusal` is non-null.

{context}"""

ANSWER_RULES = """You answer questions from ShopFlow's ops team in Slack, using ONLY the query \
result below.

- State numbers exactly as given; never invent, estimate or round away numbers.
- Columns ending in _usd are dollars: format as $1,234.56. Columns ending in _cents are cents:
  divide by 100 and format as dollars.
- Use the business's words (e.g. "transactions (orders)") and mention any assumptions briefly.
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


_json_object_fallback: set[str] = set()


def _plan(client: openai.OpenAI, model: str, messages: list[dict], usage: dict,
          options: dict) -> SqlPlan:
    fmt = {"type": "json_object"} if model in _json_object_fallback else PLAN_FORMAT
    try:
        resp = client.chat.completions.create(model=model, messages=messages,
                                              response_format=fmt, max_tokens=PLAN_MAX_TOKENS,
                                              **options)
    except openai.BadRequestError as e:
        if fmt is PLAN_FORMAT and ("response_format" in str(e) or "json_schema" in str(e)):
            _json_object_fallback.add(model)
            return _plan(client, model, messages, usage, options)
        raise
    _merge_usage(usage, resp.usage)
    content = resp.choices[0].message.content or ""
    messages.append({"role": "assistant", "content": content})
    return SqlPlan.model_validate_json(content)


def sql_answer(
    question: str,
    *,
    now: datetime | None = None,
    model: str | None = None,
    max_attempts: int = 3,
    client: openai.OpenAI | None = None,
    reasoning_effort: str | None = None,
) -> SqlAnswer:
    """Chapter 6's `sqlbot.answer`: the v1 text-to-SQL pipeline."""
    started = time.perf_counter()
    settings = get_settings()
    tz = settings.business_timezone
    model = model or settings.llm_model
    client = client or chat_client()
    now = (now or datetime.now(UTC)).astimezone(ZoneInfo(tz))
    context, tables = load_context(tz)
    usage: dict = {}
    options = {"reasoning_effort": reasoning_effort} if reasoning_effort else {}

    def done(text: str, sql: str | None = None, rows: list[dict] | None = None,
             attempts: int = 0) -> SqlAnswer:
        return SqlAnswer(answer=text, sql=sql, rows=rows or [], attempts=attempts, model=model,
                         latency_ms=round((time.perf_counter() - started) * 1000), usage=usage)

    messages = [
        {"role": "system", "content": PLANNER_RULES.format(context=context)},
        {"role": "user", "content": f"Current time: {now.isoformat()} ({tz})\n"
                                    f"Question: {question}\n"
                                    "Reply with a JSON object with keys assumptions, sql, "
                                    "needs_clarification, refusal."},
    ]
    sql, error, plan = None, "", None
    for attempt in range(1, max_attempts + 1):
        try:
            plan = _plan(client, model, messages, usage, options)
        except ValidationError as e:
            error = f"Your reply was not a valid SqlPlan JSON object: {e.errors()[:3]}"
            messages.append({"role": "user", "content": error})
            continue
        if plan.refusal:
            return done(plan.refusal, attempts=attempt)
        if plan.needs_clarification:
            return done(plan.needs_clarification, attempts=attempt)
        if not plan.sql:
            error = "Set exactly one of sql, needs_clarification or refusal."
            messages.append({"role": "user", "content": error})
            continue
        try:
            sql = validate_sql(plan.sql, set(tables))
            rows = run_query(sql)
            break
        except (UnsafeSQLError, QueryError) as e:
            error = str(e)
            sql = plan.sql
            messages.append({"role": "user", "content": f"The query failed: {error}\n"
                                                        "Fix it and reply with a new SqlPlan."})
    else:
        return done(f"Sorry, I couldn't answer that: {error}", sql=sql, attempts=max_attempts)

    # Chapter 6 used now.strftime("%-I:%M %p %Z"); %p is empty under this host's broken
    # locale, so Chapter 7 formats AM/PM by hand (see ch07_semantic.agent.as_of).
    as_of = f"{now:%-I:%M} {'AM' if now.hour < 12 else 'PM'} {now.tzname()}"
    phrase = client.chat.completions.create(
        model=model,
        max_tokens=ANSWER_MAX_TOKENS,
        **options,
        messages=[
            {"role": "system", "content": ANSWER_RULES.format(as_of=as_of)},
            {"role": "user", "content": json.dumps({
                "question": question,
                "assumptions": plan.assumptions,
                "row_count": len(rows),
                "rows": rows[:MAX_ROWS_IN_PROMPT],
            }, default=str)},
        ],
    )
    _merge_usage(usage, phrase.usage)
    # content is None when the model returns no text (refusal, length cut, provider hiccup).
    choice = phrase.choices[0]
    text = (choice.message.content or "").strip()
    if not text:
        text = f"Sorry, I got the data but couldn't phrase an answer ({choice.finish_reason})."
    return done(text, sql=sql, rows=rows, attempts=attempt)
