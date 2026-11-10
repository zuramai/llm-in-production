"""The plan -> compile -> retry loop, driven by a scripted fake client (no model calls)."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ch07_semantic.agent import answer

EMPTY = {"dimensions": [], "filters": [], "time_range": None, "grain": None, "order_by": [],
         "limit": None}


def plan(query=None, refusal=None):
    return json.dumps({"assumptions": [], "query": query, "needs_clarification": None,
                       "refusal": refusal})


class ScriptedClient:
    """Returns the scripted replies in order and records every request."""

    def __init__(self, *replies: str):
        self.replies, self.requests = list(replies), []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        # Snapshot: the agent keeps appending to the same messages list.
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        message = SimpleNamespace(content=self.replies.pop(0))
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)


def test_refusal_returns_without_sql():
    client = ScriptedClient(plan(refusal="Customer emails are PII."))
    res = answer("List customer emails", client=client, model="fake")
    assert (res.answer, res.sql, res.attempts) == ("Customer emails are PII.", None, 1)
    assert client.requests[0]["response_format"]["json_schema"]["strict"] is True
    assert "max_tokens" in client.requests[0]


def test_invalid_queries_are_fed_back_and_attempts_are_capped():
    bad = plan({**EMPTY, "metrics": ["transactions"],
                "filters": [{"dimension": "customer.segment", "op": "=", "values": ["gold"]}]})
    client = ScriptedClient(bad, bad, bad)
    res = answer("Orders from gold customers", client=client, model="fake")
    assert res.attempts == 3 and res.answer.startswith("Sorry, I couldn't answer that")
    feedback = client.requests[-1]["messages"][-1]["content"]
    assert "allowed: consumer, business, vip" in feedback
    assert len(client.requests) == 3                 # no phrasing call after a failure


def test_malformed_json_is_retried():
    client = ScriptedClient("not json", plan(refusal="No metric for that."))
    res = answer("Conversion rate?", client=client, model="fake")
    assert res.attempts == 2
    assert "not a valid MetricPlan" in client.requests[1]["messages"][-1]["content"]


@pytest.mark.docker
def test_fixed_plan_runs_and_phrase_gets_formatted_rows():
    bad = plan({**EMPTY, "metrics": ["gross_revenue"], "dimensions": ["category.name"]})
    good = plan({**EMPTY, "metrics": ["product_revenue"], "dimensions": ["category.name"]})
    client = ScriptedClient(bad, good, "phrased")
    res = answer("Revenue by category", client=client, model="fake")
    assert res.attempts == 2 and res.answer == "phrased" and len(res.rows) == 12
    payload = json.loads(client.requests[-1]["messages"][-1]["content"])
    assert all(row["product_revenue"].startswith("$") for row in payload["rows"])
