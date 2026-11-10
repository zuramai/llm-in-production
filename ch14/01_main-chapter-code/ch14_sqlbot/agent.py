"""The Chapter 6 text-to-SQL pipeline, instrumented with Langfuse (Python SDK v5).

Same logic as ch06/01_main-chapter-code/sqlbot/agent.py: plan (structured output) -> guard ->
run (read-only) -> phrase. What's new is the telemetry around each step:

    trace "sqlbot"   user_id, session_id, tags, release, version
    └─ agent  sqlbot.answer          input: question, output: answer + sql
       ├─ generation plan (×1-3)     model, prompt version, usage, cost, gen_ai.* attributes
       ├─ guardrail  sql_guard       pass / rule=... (level WARNING when it blocks)
       ├─ tool       postgres.query  row count, errors
       └─ generation phrase          model, prompt version, usage, cost
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from zoneinfo import ZoneInfo

import ch14_obs as obs
import openai
import yaml
from langfuse import propagate_attributes
from opentelemetry import trace as otel_trace
from pydantic import BaseModel, ConfigDict, ValidationError

from llmprod import chat_client, get_settings

from . import prompts
from .db import QueryError, run_query
from .guard import UnsafeSQLError, validate_sql

CONTEXT_PATH = Path(__file__).with_name("semantic_context.yaml")
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
    trace_id: str | None = None
    prompt_versions: dict[str, int | None] = {}


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


def _merge_usage(total: dict, usage) -> None:
    if usage is None:
        return
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        total[key] = total.get(key, 0) + (getattr(usage, key, 0) or 0)
    cost = getattr(usage, "cost", None) or (usage.model_extra or {}).get("cost")
    if cost:
        total["cost_usd"] = total.get("cost_usd", 0.0) + float(cost)


def _llm_call(lf, client: openai.OpenAI, *, name: str, model: str, messages: list[dict],
              prompt, usage: dict, max_tokens: int, **kwargs):
    """One chat completion, recorded as a Langfuse generation with GenAI semconv attributes."""
    provider = obs.provider_of(str(client.base_url))
    params = {"max_tokens": max_tokens} | {k: v for k, v in kwargs.items()
                                           if k in ("reasoning_effort", "temperature")}
    with lf.start_as_current_observation(
        as_type="generation", name=name, model=model, input=messages,
        model_parameters=params, prompt=None if prompt.is_fallback else prompt,
    ) as gen:
        span = otel_trace.get_current_span()
        span.set_attributes(obs.genai_attributes(operation="chat", provider=provider,
                                                 request_model=model, max_tokens=max_tokens))
        try:
            resp = client.chat.completions.create(model=model, messages=messages,
                                                  max_tokens=max_tokens, **kwargs)
        except Exception as e:
            gen.update(level="ERROR", status_message=f"{type(e).__name__}: {e}"[:500])
            span.set_attribute("error.type", type(e).__name__)
            raise
        span.set_attributes(obs.genai_attributes(operation="chat", provider=provider,
                                                 request_model=model, max_tokens=max_tokens,
                                                 response=resp))
        gen.update(output=resp.choices[0].message.content,
                   usage_details=obs.usage_details(resp.usage),
                   cost_details=obs.cost_details(resp.usage, provider))
    _merge_usage(usage, resp.usage)
    return resp


_json_object_fallback: set[str] = set()


def _plan(lf, client, model, messages, usage, options, prompt) -> SqlPlan:
    fmt = {"type": "json_object"} if model in _json_object_fallback else PLAN_FORMAT
    try:
        resp = _llm_call(lf, client, name="plan", model=model, messages=messages, prompt=prompt,
                         usage=usage, max_tokens=PLAN_MAX_TOKENS, response_format=fmt, **options)
    except openai.BadRequestError as e:
        if fmt is PLAN_FORMAT and ("response_format" in str(e) or "json_schema" in str(e)):
            _json_object_fallback.add(model)
            return _plan(lf, client, model, messages, usage, options, prompt)
        raise
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
    user_id: str | None = None,
    session_id: str | None = None,
    prompt_label: str = "production",
    tags: list[str] | None = None,
    langfuse=None,
) -> SqlAnswer:
    lf = langfuse or obs.client()
    settings = get_settings()
    model = model or settings.llm_model
    planner = prompts.get_prompt(lf, prompts.PLANNER, prompt_label)
    answerer = prompts.get_prompt(lf, prompts.ANSWERER, "production")
    with lf.start_as_current_observation(as_type="agent", name="sqlbot.answer",
                                         input={"question": question}) as root, \
            propagate_attributes(user_id=user_id, session_id=session_id, trace_name="sqlbot",
                                 version=obs.APP_VERSION,
                                 tags=["sqlbot", f"planner:{prompt_label}", *(tags or [])],
                                 metadata={"model": model}):
        result = _answer(lf, question, now=now, model=model, max_attempts=max_attempts,
                         client=client or chat_client(), reasoning_effort=reasoning_effort,
                         planner=planner, answerer=answerer, tz=settings.business_timezone)
        root.update(output={"answer": result.answer, "sql": result.sql,
                            "row_count": len(result.rows), "attempts": result.attempts},
                    metadata={"latency_ms": result.latency_ms,
                              "prompt_versions": result.prompt_versions})
        result.trace_id = lf.get_current_trace_id()
    return result


def _answer(lf, question: str, *, now, model, max_attempts, client, reasoning_effort,
            planner, answerer, tz) -> SqlAnswer:
    started = time.perf_counter()
    now = (now or datetime.now(UTC)).astimezone(ZoneInfo(tz))
    context, tables = load_context(tz)
    usage: dict = {}
    options = {"reasoning_effort": reasoning_effort} if reasoning_effort else {}
    versions = {prompts.PLANNER: None if planner.is_fallback else planner.version,
                prompts.ANSWERER: None if answerer.is_fallback else answerer.version}

    def done(text: str, sql: str | None = None, rows: list[dict] | None = None,
             attempts: int = 0) -> SqlAnswer:
        return SqlAnswer(answer=text, sql=sql, rows=rows or [], attempts=attempts, model=model,
                         latency_ms=round((time.perf_counter() - started) * 1000), usage=usage,
                         prompt_versions=versions)

    messages = [
        {"role": "system", "content": planner.compile(context=context)},
        {"role": "user", "content": f"Current time: {now.isoformat()} ({tz})\n"
                                    f"Question: {question}\n"
                                    "Reply with a JSON object with keys assumptions, sql, "
                                    "needs_clarification, refusal."},
    ]
    sql, error, plan, rows = None, "", None, []
    for attempt in range(1, max_attempts + 1):
        try:
            plan = _plan(lf, client, model, messages, usage, options, planner)
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
            with lf.start_as_current_observation(as_type="guardrail", name="sql_guard",
                                                 input=plan.sql) as g:
                try:
                    sql = validate_sql(plan.sql, set(tables))
                except UnsafeSQLError as e:
                    g.update(level="WARNING", status_message=str(e), output={"blocked": True})
                    raise
                g.update(output={"blocked": False, "sql": sql})
            with lf.start_as_current_observation(as_type="tool", name="postgres.query",
                                                 input=sql) as t:
                try:
                    rows = run_query(sql)
                except QueryError as e:
                    t.update(level="ERROR", status_message=str(e)[:500])
                    raise
                # Row *count* only: result rows can hold customer data (Chapter 6, logging).
                t.update(output={"row_count": len(rows)})
            break
        except (UnsafeSQLError, QueryError) as e:
            error = str(e)
            sql = plan.sql
            messages.append({"role": "user", "content": f"The query failed: {error}\n"
                                                        "Fix it and reply with a new SqlPlan."})
    else:
        return done(f"Sorry, I couldn't answer that: {error}", sql=sql, attempts=max_attempts)

    # Not %p: it's empty under some locales (this host's LC_TIME=id_ID gave "5:19  EST").
    as_of = f"{now:%-I:%M} {'AM' if now.hour < 12 else 'PM'} {now:%Z}"
    phrase = _llm_call(lf, client, name="phrase", model=model, prompt=answerer, usage=usage,
                       max_tokens=ANSWER_MAX_TOKENS, **options, messages=[
                           {"role": "system", "content": answerer.compile(as_of=as_of)},
                           {"role": "user", "content": json.dumps({
                               "question": question,
                               "assumptions": plan.assumptions,
                               "row_count": len(rows),
                               "rows": rows[:MAX_ROWS_IN_PROMPT],
                           }, default=str)},
                       ])
    # content is None when the model returns no text (refusal, length cut, provider hiccup).
    choice = phrase.choices[0]
    text = (choice.message.content or "").strip()
    if not text:
        text = f"Sorry, I got the data but couldn't phrase an answer ({choice.finish_reason})."
    return done(text, sql=sql, rows=rows, attempts=attempt)
