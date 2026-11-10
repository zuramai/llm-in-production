"""The incident-analysis agent: a tool loop that investigates, then a strict-schema report call.

Read-only by construction: none of the four tools changes anything. The report only *proposes*
next actions; a human executes them.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import openai

from llmprod import chat_client

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from previous_chapters import extract_json, usage_cost

from .report import IncidentReport, response_format
from .tools import TOOL_SPECS, dispatch

OLLAMA_URL = "http://localhost:11434/v1"


def client_for(model: str) -> openai.OpenAI:
    """'vendor/model' goes to LLM_BASE_URL (OpenRouter); a bare 'name:tag' goes to local Ollama."""
    if "/" in model:
        return chat_client(timeout=90, max_retries=2)
    return openai.OpenAI(base_url=OLLAMA_URL, api_key="ollama", timeout=600, max_retries=0)


def reasoning_for(model: str) -> str:
    """gpt-oss can't switch reasoning off (low/medium/high only); the others take 'none'."""
    return "low" if model.startswith("gpt-oss") else "none"


SYSTEM = """You are an on-call incident analysis agent for ShopFlow's `checkout` service.
You investigate with read-only tools, then report. You NEVER change anything: you propose
actions and a human executes them.

Current time (UTC): {now}

Telemetry contract (job="checkout"):
- http_requests_total{{route,method,status}}   counter
- http_request_duration_seconds{{route}}       histogram (_bucket)
- db_pool_connections_in_use, db_pool_size     gauges
- db_pool_wait_seconds                         histogram (_bucket)
- payment_provider_request_duration_seconds{{provider}} histogram; provider is stripe|paypal
- payment_provider_errors_total{{provider,reason}} counter; reason is timeout|declined|error
Loki: {{service="checkout", level="error"}} with JSON fields msg, route, status, duration_ms,
deploy_version, error, provider. Use `| json` to filter, e.g. `| msg="unhandled exception"`.

Method: confirm the symptom, then test hypotheses one at a time with targeted queries. Compare
signals across routes, providers and deploy versions. Check recent deploys. Consult the runbooks
for severity and known causes. Prefer `sum by (...)` and `rate(...[1m])` to keep results small.
Stop investigating as soon as one cause is supported by evidence and the others are ruled out."""

FINAL = """Investigation over. Write the incident report as JSON matching the schema.
- hypotheses: list every cause you considered, each with 1-4 evidence strings quoting numbers
  or log patterns you actually saw. Give calibrated confidences between 0 and 1.
- timeline: UTC times from the data (deploys, first errors, peak).
- next_actions: proposals only, written for a human (e.g. 'Roll back to <version>'). Do not
  say you performed any action."""


@dataclass
class RunResult:
    report: IncidentReport | None
    model: str
    steps: int = 0
    cost_usd: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tool_chars: int = 0
    max_prompt_tokens: int = 0
    latency_s: float = 0.0
    error: str | None = None
    trace: list[dict] = field(default_factory=list)


def run_agent(alert: str, model: str, *, max_steps: int = 8, summarize: bool = True,
              now: datetime | None = None) -> RunResult:
    client = client_for(model)
    effort = reasoning_for(model)
    now = now or datetime.now(UTC)
    messages = [{"role": "system", "content": SYSTEM.format(now=now.strftime("%Y-%m-%d %H:%M:%S"))},
                {"role": "user", "content": f"ALERT: {alert}\nInvestigate."}]
    res = RunResult(report=None, model=model)
    t0 = time.perf_counter()

    def track(resp) -> None:
        u = resp.usage
        res.prompt_tokens += u.prompt_tokens
        res.completion_tokens += u.completion_tokens
        res.max_prompt_tokens = max(res.max_prompt_tokens, u.prompt_tokens)
        res.cost_usd += usage_cost(u, model)

    try:
        for step in range(max_steps):
            resp = client.chat.completions.create(
                model=model, messages=messages, tools=TOOL_SPECS, max_tokens=1200,
                reasoning_effort=effort)
            track(resp)
            msg = resp.choices[0].message
            if not msg.tool_calls:
                if msg.content:
                    messages.append({"role": "assistant", "content": msg.content})
                break
            messages.append(msg.model_dump(exclude_none=True))
            for call in msg.tool_calls:
                try:
                    args = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                out = dispatch(call.function.name, args, summarize=summarize)
                res.tool_chars += len(out)
                res.trace.append({"step": step + 1, "tool": call.function.name, "args": args,
                                  "chars": len(out), "output": out[:400]})
                messages.append({"role": "tool", "tool_call_id": call.id, "content": out})
            res.steps = step + 1
        messages.append({"role": "user", "content": FINAL})
        resp = client.chat.completions.create(
            model=model, messages=messages, max_tokens=2500, response_format=response_format(),
            reasoning_effort=effort)
        track(resp)
        res.report = IncidentReport.model_validate_json(
            extract_json(resp.choices[0].message.content or ""))
    except Exception as exc:  # noqa: BLE001 report the failure in the eval instead of crashing it
        res.error = f"{type(exc).__name__}: {str(exc)[:300]}"
    res.latency_s = time.perf_counter() - t0
    return res
