"""Unit tests for the fixes in Chapter 22's repros. No network, no model, no database."""

import datetime as dt
import sys
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace as NS
from zoneinfo import ZoneInfo

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from repros import a1_finish_length as a1
from repros import a2_reasoning_budget as a2
from repros import a3_content_none as a3
from repros import a4_unsupported_params as a4
from repros import a5_unicode_tokens as a5
from repros import a6_context_overflow as a6
from repros import b1_json_object_prose as b1
from repros import b2_nondeterminism as b2
from repros import b3_content_filter as b3
from repros import d1_tool_arg_hallucination as d1
from repros import d3_agent_runaway as d3
from repros import e1_judge_position as e1
from repros import e2_confidence as e2
from repros import e3_eval_gate_noise as e3
from repros import f1_429_storm as f1
from repros import g1_timezone_today as g1
from repros import g2_cents_dollars as g2
from repros import g3_strftime_locale as g3
from repros import g4_dst_arithmetic as g4
from repros import g5_enum_spelling as g5
from repros import h1_pii_logs as h1
from repros import h2_prompt_injection as h2
from repros import i2_tool_schema_tax as i2
from repros import j1_model_churn as j1
from repros import j2_prometheus_missing as j2


def choice(content=None, finish="stop", tool_calls=None, refusal=None):
    return NS(finish_reason=finish,
              message=NS(content=content, tool_calls=tool_calls, refusal=refusal))


# A: API & tokens ------------------------------------------------------------------------------

def test_parse_structured_refuses_truncated_json():
    with pytest.raises(a1.TruncatedOutput, match="length"):
        a1.parse_structured(choice('{"category":"billing","summ', "length"))
    with pytest.raises(a1.TruncatedOutput, match="empty"):
        a1.parse_structured(choice(None, "stop"))
    assert a1.parse_structured(choice('{"a": 1}')) == {"a": 1}


def test_visible_answer_reports_reasoning_burn():
    resp = NS(choices=[choice(None, "length")],
              usage=NS(completion_tokens=60, completion_tokens_details=NS(reasoning_tokens=60)))
    with pytest.raises(RuntimeError, match="60/60"):
        a2.visible_answer(resp)


def test_message_text_never_none():
    tc = NS(function=NS(name="get_order_status"))
    assert a3.message_text(choice(None, "tool_calls", [tc])) == "[tool call: get_order_status]"
    assert a3.message_text(choice(None, "stop", refusal="no")) == "[refused] no"
    assert a3.message_text(choice(None, "length")) == "[no content: finish_reason=length]"
    assert a3.message_text(choice("  hi \n")) == "hi"


LUNA_META = {"supported_parameters": ["max_tokens", "reasoning_effort", "seed"],
             "reasoning": {"supported_efforts": ["high", "medium", "low", "none"]}}
SONNET_META = {"supported_parameters": ["max_tokens", "reasoning_effort"],
               "reasoning": {"mandatory": True, "supported_efforts": ["high", "medium", "low"]}}


def test_fit_params_drops_and_maps():
    sent, notes = a4.fit_params(LUNA_META, {"temperature": 0, "seed": 1,
                                            "reasoning_effort": "none"})
    assert sent == {"seed": 1, "reasoning_effort": "none"}
    assert notes == ["dropped temperature"]
    sent, notes = a4.fit_params(SONNET_META, {"reasoning_effort": "none", "max_tokens": 5})
    assert sent == {"reasoning_effort": "low", "max_tokens": 5}


def test_normalize_text_folds_quotes_forms_and_space():
    assert a5.normalize_text("wasn\u2019t  found") == a5.normalize_text("Wasn't found")
    assert a5.normalize_text("Cafe\u0301") == a5.normalize_text("Café")
    assert a5.cache_key("refund policy?") == a5.cache_key(" refund\u00a0policy? ")
    assert a5.cache_key("refund policy?") != a5.cache_key("refund policy!")


def test_context_preflight_uses_the_smaller_window():
    assert a6.effective_context({"context_length": 8192,
                                 "top_provider": {"context_length": 4096}}) == 4096
    assert a6.effective_context({"context_length": 8192, "top_provider": {}}) == 8192
    with pytest.raises(a6.ContextOverflow):
        a6.preflight("x" * 12_300, 4096, max_tokens=10)  # ~4,101 + 10 > 4,096
    assert a6.preflight("x" * 3_000, 4096, max_tokens=10) == 1001


# B: structured output -------------------------------------------------------------------------

def test_extract_json_handles_fences_and_prose():
    assert b1.extract_json('```json\n{"a": {"b": 1}}\n```') == {"a": {"b": 1}}
    assert b1.extract_json('Sure! Here it is: {"a": 2}. Anything else?') == {"a": 2}
    with pytest.raises(ValueError):
        b1.extract_json("no json here")


def test_priority_rule_and_decision_store():
    facts = {"charged_incorrectly": True, "customer_mentions_safety_risk": False,
             "customer_mentions_account_takeover": False,
             "customer_says_a_payment_of_theirs_failed_because_of_us": False,
             "days_waiting": None}
    assert b2.priority_from_facts(facts) == "high"
    assert b2.priority_from_facts(facts | {"customer_mentions_account_takeover": True}) == "urgent"
    assert b2.priority_from_facts(facts | {"charged_incorrectly": False, "days_waiting": 9}) \
        == "medium"
    calls = iter(["high", "urgent", "low"])
    store = b2.DecisionStore("v1")
    got = {store.decide(t, "m", lambda m, x: next(calls)) for t in ("a  b", "a b", " a b ")}
    assert got == {"high"} and store.model_calls == 1
    assert b2.DecisionStore("v2").key("a b", "m") != store.key("a b", "m")


def test_content_filter_is_terminal():
    assert b3.next_action(choice(None, "content_filter")) == "human_review"
    assert b3.next_action(choice("{", "length")) == "retry"
    assert b3.next_action(choice("{}", "stop")) == "accept"
    assert b3.next_action(choice(None, "stop")) == "human_review"


# D: agents & tools ----------------------------------------------------------------------------

@pytest.mark.parametrize(("args", "text", "bad"), [
    ({"order_id": 0}, "Where is my order?", ["order_id"]),
    ({"order_id": None}, "Where is my order?", ["order_id"]),
    ({"order_id": 79}, "the earbuds cost $79.99", ["order_id"]),
    ({"order_id": 99}, "the earbuds cost $79.99", ["order_id"]),
    ({"order_id": 4512}, "Where is order #4512.", []),
    ({"order_id": 4512, "amount_cents": 1}, "refund 4512 please", []),
])
def test_ungrounded_ids(args, text, bad):
    assert d1.ungrounded_ids(args, text) == bad


def test_guard_trips_on_repeated_tool_calls_and_steps():
    g = d3.Guard(max_steps=10, max_per_tool=2)
    g.check_call("search_orders")
    g.check_call("search_orders")
    with pytest.raises(d3.BudgetExceeded, match="3x"):
        g.check_call("search_orders")
    g = d3.Guard(max_steps=2)
    g.charge(0.0)
    with pytest.raises(d3.BudgetExceeded, match="step cap"):
        g.charge(0.0)
    with pytest.raises(d3.BudgetExceeded, match=r"\$"):
        d3.Guard(max_usd=0.01).charge(0.02)


# E: evals -------------------------------------------------------------------------------------

def test_swap_verdict_counts_only_consistent_wins():
    assert e1.swap_verdict("A", "B") == "x"      # x first and x second: x wins
    assert e1.swap_verdict("B", "A") == "y"
    assert e1.swap_verdict("A", "A") == "tie"    # always slot A: position, not content


def test_correct_and_agreement():
    assert e2.correct("12,345,678.", "12345678")
    assert e2.correct("thursday", "Thursday")
    assert not e2.correct("1234567", "12345678")
    assert e2.agreement(["1,000", "1000", "999"]) == pytest.approx(2 / 3)


def test_gates():
    passes = {f"q{i}": 1 for i in range(18)} | {"j1": 0}
    assert e3.bar_gate(passes, 1, 0.90) and not e3.bar_gate(passes, 1, 0.95)
    base = {"a": 3, "b": 3, "c": 0}
    assert e3.flip_gate(base, {"a": 1, "b": 3, "c": 0}, 3)       # flaky, not flipped
    assert not e3.flip_gate(base, {"a": 0, "b": 3, "c": 3}, 3)   # a flipped 3/3 -> 0/3


# F: infra -------------------------------------------------------------------------------------

def test_bucket_burst_then_wait():
    b = f1.Bucket(rate=10, burst=2)
    assert b.take() == 0 and b.take() == 0
    wait = b.take()
    assert 0 < wait <= 0.1


# G: data & time -------------------------------------------------------------------------------

def test_business_day_bounds_on_dst_end():
    start, end = g1.business_day_bounds(dt.date(2026, 11, 1), "America/New_York")
    assert (end - start) == dt.timedelta(days=1)            # wall clock
    assert g4.elapsed(start, end) == dt.timedelta(hours=25)  # real: the fall-back day
    assert start.utcoffset() == dt.timedelta(hours=-4)
    assert end.utcoffset() == dt.timedelta(hours=-5)


@pytest.mark.parametrize(("cents", "text"), [
    (0, "$0.00"), (5, "$0.05"), (4599, "$45.99"), (1126220, "$11,262.20"),
    (-150, "-$1.50"), (10**12 + 1, "$10,000,000,000.01"),
])
def test_format_cents(cents, text):
    assert g2.format_cents(cents) == text


def test_as_of_is_locale_free():
    assert g3.as_of(dt.datetime(2026, 11, 7, 0, 5)) == "as of 12:05 AM"
    assert g3.as_of(dt.datetime(2026, 11, 7, 12, 0)) == "as of 12:00 PM"
    assert g3.as_of(dt.datetime(2026, 11, 7, 17, 19), "EST") == "as of 5:19 PM EST"


def test_add_hours_across_dst():
    t = dt.datetime(2026, 10, 31, 9, 0, tzinfo=ZoneInfo("America/New_York"))
    assert g4.add_hours(t, 48).hour == 8
    assert g4.add_hours(t, 48).utcoffset() == dt.timedelta(hours=-5)


@pytest.mark.parametrize("sql", [
    "SELECT count(*) FROM orders WHERE status <> 'canceled'",
    "SELECT count(*) FROM orders WHERE status = 'Cancelled'",
    "SELECT count(*) FROM orders o WHERE o.status IN ('paid', 'canceled')",
])
def test_enum_check_rejects_unknown_values(sql):
    with pytest.raises(g5.UnknownEnumValue):
        g5.check_enum_literals(sql, {"status": {"paid", "cancelled"}})


def test_enum_check_ignores_valid_and_other_columns():
    g5.check_enum_literals("SELECT 1 FROM orders WHERE status IN ('paid', 'cancelled') "
                           "AND country = 'Narnia'", {"status": {"paid", "cancelled"}})


def test_decimal_money_adds_up():
    assert sum([Decimal("0.10")] * 3) == Decimal("0.30")


# H: security ----------------------------------------------------------------------------------

def test_redact_keeps_order_numbers():
    out = h1.redact(h1.PROMPT)
    for pii in h1.PII:
        assert pii not in out
    assert "order 4512" in out and "[EMAIL]" in out and "[CARD]" in out


@pytest.mark.parametrize(("text", "blocked"), [
    ("Claim at https://shopflow-refunds.example/claim now", ["shopflow-refunds.example"]),
    ("Go to **shopflow-refunds.example/claim**", ["shopflow-refunds.example"]),
    ("Start at https://shopflow.example/returns.", []),
    ("Visit help.shopflow.example or e.g. 5.99 dollars", []),
    ("See https://shopflow.example.evil.io/x", ["shopflow.example.evil.io"]),
])
def test_link_guard(text, blocked):
    clean, got = h2.link_guard(text)
    assert got == blocked
    for host in blocked:
        assert host not in clean


# I/J: cost & ops ------------------------------------------------------------------------------

def test_tool_routing():
    names = {t["function"]["name"] for t in i2.tools_for("Where is my package?")}
    assert "get_order_status" in names and "issue_refund" not in names
    names = {t["function"]["name"] for t in i2.tools_for("Can I return skincare?")}
    assert names == {"search_faq", "escalate_to_human"}


def test_model_health():
    today = dt.date(2026, 11, 8)
    pinned = {"m/a": "m/a-1", "m/b": "m/b-1", "m/c": "m/c-1"}
    catalog = {"m/a": {"canonical_slug": "m/a-1"},
               "m/b": {"canonical_slug": "m/b-2"},
               "m/c": {"canonical_slug": "m/c-1", "expiration_date": "2026-12-01"}}
    problems = j1.model_health(catalog | {}, pinned | {"m/gone": "x"}, today)
    joined = " | ".join(problems)
    assert "m/b: now serves m/b-2" in joined
    assert "m/c: retires 2026-12-01 (23 days)" in joined
    assert "m/gone: not in the catalog" in joined
    assert not any(p.startswith("m/a") for p in problems)


def test_prometheus_preinit_exports_zero():
    from prometheus_client import CollectorRegistry

    reg = CollectorRegistry()
    j2.make_counter(reg, preinit=True)
    assert 'llm_requests_total{status="5xx"} 0.0' in j2.series(reg)
    reg = CollectorRegistry()
    j2.make_counter(reg, preinit=False)
    assert j2.series(reg) == []
