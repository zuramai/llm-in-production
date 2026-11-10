"""B2: same input, temperature 0, different answers. Judgments drift; extracted facts less so."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from ch22_common import LUNA, QWEN_FLASH, Result, chat, timed

ID, TITLE, LAYER, KIND = "B2", "Nondeterminism at temperature 0", "Structured output", "hosted"

TICKETS = [
    "I was charged twice for order #24190 and now my rent payment bounced. Please fix it.",
    ("My bank shows two pending charges of $45.99 for order #24190. Not sure if one will drop "
     "off by itself?"),
    ("Someone changed the email on my account last night and I didn't do it. Order #24190 is "
     "still showing though, so maybe it's fine?"),
]
RUBRIC = ("Priority rubric: urgent = safety risk, account takeover or financial hardship; "
          "high = money charged wrongly; medium = delays; low = questions.")

JUDGE_SCHEMA = {"type": "json_schema", "json_schema": {"name": "p", "strict": True, "schema": {
    "type": "object", "additionalProperties": False,
    "properties": {"priority": {"type": "string", "enum": ["low", "medium", "high", "urgent"]}},
    "required": ["priority"]}}}
FACTS_SCHEMA = {"type": "json_schema", "json_schema": {"name": "f", "strict": True, "schema": {
    "type": "object", "additionalProperties": False,
    "properties": {
        "charged_incorrectly": {"type": "boolean"},
        "customer_mentions_safety_risk": {"type": "boolean"},
        "customer_mentions_account_takeover": {"type": "boolean"},
        "customer_says_a_payment_of_theirs_failed_because_of_us": {"type": "boolean"},
        "days_waiting": {"type": ["integer", "null"]},
    },
    "required": ["charged_incorrectly", "customer_mentions_safety_risk",
                 "customer_mentions_account_takeover",
                 "customer_says_a_payment_of_theirs_failed_because_of_us", "days_waiting"]}}}


def priority_from_facts(f: dict) -> str:
    """The rubric as code: the model reports facts, we make the call."""
    if (f["customer_mentions_safety_risk"] or f["customer_mentions_account_takeover"]
            or f["customer_says_a_payment_of_theirs_failed_because_of_us"]):
        return "urgent"
    if f["charged_incorrectly"]:
        return "high"
    if (f.get("days_waiting") or 0) >= 7:
        return "medium"
    return "low"


class DecisionStore:
    """Decide once per (input, model, prompt version), then replay the stored decision.

    A user who re-opens a ticket, or an auditor who replays it, sees the label we acted on,
    not a fresh sample. In production this is a table keyed by the same hash.
    """

    def __init__(self, prompt_version: str) -> None:
        self.prompt_version = prompt_version
        self.rows: dict[str, str] = {}
        self.model_calls = 0

    def key(self, text: str, model: str) -> str:
        raw = f"{self.prompt_version}|{model}|{' '.join(text.split())}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def decide(self, text: str, model: str, fn) -> str:
        k = self.key(text, model)
        if k not in self.rows:
            self.model_calls += 1
            self.rows[k] = fn(model, text)
        return self.rows[k]


def _kw(model: str) -> dict:
    # Luna ignores temperature (see A4); qwen3.8-flash honors temperature and seed.
    if model == QWEN_FLASH:
        return {"temperature": 0, "seed": 7, "reasoning_effort": "none"}
    return {"reasoning_effort": "none"}


def _judge(model: str, ticket: str) -> str:
    r = chat(model, [{"role": "system", "content": RUBRIC}, {"role": "user", "content": ticket}],
             max_tokens=60, response_format=JUDGE_SCHEMA, **_kw(model))
    return json.loads(r.choices[0].message.content)["priority"]


def _facts_rule(model: str, ticket: str) -> str:
    r = chat(model, [{"role": "system", "content": "Extract facts from the ticket."},
                     {"role": "user", "content": ticket}],
             max_tokens=120, response_format=FACTS_SCHEMA, **_kw(model))
    return priority_from_facts(json.loads(r.choices[0].message.content))


def _summary(model: str, ticket: str) -> str:
    r = chat(model, [{"role": "user", "content": "Summarize in one sentence: " + ticket}],
             max_tokens=80, **_kw(model))
    return (r.choices[0].message.content or "").strip()


@timed
def run(n: int = 10) -> Result:
    out = {}
    with ThreadPoolExecutor(8) as pool:
        for model in (LUNA, QWEN_FLASH):
            per_ticket = []
            for ticket in TICKETS:
                judged = Counter(pool.map(_judge, [model] * n, [ticket] * n))
                ruled = Counter(pool.map(_facts_rule, [model] * n, [ticket] * n))
                per_ticket.append({"judged": dict(judged), "facts_rule": dict(ruled)})
            summaries = set(pool.map(_summary, [model] * n, [TICKETS[1]] * n))
            out[model] = {"tickets": per_ticket, "distinct_summaries": len(summaries)}

    def stability(key: str) -> float:
        """Share of runs that gave the majority label, worst ticket and model."""
        return min(max(t[key].values()) / n for v in out.values() for t in v["tickets"])

    def cross_model(key: str) -> int:
        """Tickets where both models' majority labels agree."""
        majors = [[max(t[key], key=t[key].get) for t in v["tickets"]] for v in out.values()]
        return sum(a == b for a, b in zip(*majors, strict=True))

    stats = {f"{f.__name__}_{key}": f(key) for key in ("judged", "facts_rule")
             for f in (stability, cross_model)}
    store = DecisionStore("triage-v3")
    replays = Counter(store.decide(TICKETS[1] + " " * (i % 2), QWEN_FLASH, _summary)
                      for i in range(n))
    stats |= {"store_distinct": len(replays), "store_model_calls": store.model_calls}
    return Result(
        ID, TITLE, LAYER, KIND,
        reproduced=stats["stability_judged"] < 1 or out[QWEN_FLASH]["distinct_summaries"] > 1,
        fixed=stats["store_distinct"] == 1 and stats["store_model_calls"] == 1,
        evidence="; ".join(
            f"{m.split('/')[1]}: judged " + " | ".join(str(t["judged"]) for t in v["tickets"])
            + " ; facts+rule " + " | ".join(str(t["facts_rule"]) for t in v["tickets"])
            + f" ; {v['distinct_summaries']}/{n} distinct summaries"
            for m, v in out.items()) + f" (n={n} per ticket); {stats}",
        data=out | stats,
    )
