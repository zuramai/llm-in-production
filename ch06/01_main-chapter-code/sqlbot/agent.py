"""Text-to-SQL pipeline: plan (structured output) -> guard -> run (read-only) -> phrase."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from zoneinfo import ZoneInfo

import openai
import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from llmprod import chat_client, get_settings

from .db import QueryError, run_query
from .guard import UnsafeSQLError, validate_sql

CONTEXT_PATH = Path(__file__).with_name("semantic_context.yaml")
MAX_ROWS_IN_PROMPT = 50
# Always cap output. Reasoning tokens count against max_tokens, so leave room. (Some gateways
# also reserve credit for the full default max_tokens; see Chapter 22.)
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
    """Render semantic_context.yaml into prompt text; return it with the allowed table names."""
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


def answer(
    question: str,
    *,
    now: datetime | None = None,
    model: str | None = None,
    max_attempts: int = 3,
    client: openai.OpenAI | None = None,
    reasoning_effort: str | None = None,
) -> SqlAnswer:
    started = time.perf_counter()
    settings = get_settings()
    tz = settings.business_timezone
    model = model or settings.llm_model
    client = client or chat_client()
    now = (now or datetime.now(UTC)).astimezone(ZoneInfo(tz))
    context, tables = load_context(tz)
    usage: dict = {}
    # e.g. "none" turns off qwen3 thinking on Ollama and lowers latency on reasoning models.
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

    as_of = now.strftime("%-I:%M %p %Z")
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
    return done(phrase.choices[0].message.content.strip(), sql=sql, rows=rows, attempts=attempt)
