import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ch15_gateway.calls import Call, parse_cost, plain_text_messages
from ch15_gateway.cascade import answer_messages, cascade, parse_verdict
from ch15_gateway.judge import parse_grade

PASS = '{"supported": true, "answers": true, "reason": "ok"}'
FAIL = '{"supported": false, "answers": true, "reason": "invented a fee"}'


class FakeChat:
    """Scripted models: answers per model name, verifier replies in order, records every call."""

    def __init__(self, verdicts: list[str]):
        self.verdicts = list(verdicts)
        self.log: list[tuple[str, object]] = []

    def __call__(self, model, messages, max_tokens, response_format):
        self.log.append((model, response_format))
        if response_format is not None:          # verifier call
            text = self.verdicts.pop(0)
            return Call("verifier", text, 0.0001, 0.1, 500, 20)
        cost = {"cheap": 0.0002, "mid": 0.001, "strong": 0.01}[model]
        return Call(model, f"answer from {model}", cost, 1.0, 1000, 50)


def run(chat, tiers=("cheap", "strong")):
    return cascade(chat, "q?", "system", "faq", tiers, verifier="cheap")


def test_accepted_draft_is_not_escalated():
    chat = FakeChat([PASS])
    r = run(chat)
    assert (r.answered_by, r.escalated, r.answer) == ("cheap", False, "answer from cheap")
    assert [m for m, _ in chat.log] == ["cheap", "cheap"]          # draft + verify, no strong call
    assert r.cost_usd == pytest.approx(0.0003)


def test_rejected_draft_escalates_and_last_tier_is_not_verified():
    chat = FakeChat([FAIL])
    r = run(chat)
    assert (r.answered_by, r.escalated) == ("strong", True)
    assert len(r.calls) == 3 and chat.verdicts == []               # exactly one verifier call
    assert r.verdict is not None and r.verdict.reason == "invented a fee"
    assert r.cost_usd == pytest.approx(0.0002 + 0.0001 + 0.01)
    assert r.latency_s == pytest.approx(2.1)


def test_three_tiers_stop_at_first_accepted():
    chat = FakeChat([FAIL, PASS])
    r = run(chat, tiers=("cheap", "mid", "strong"))
    assert r.answered_by == "mid" and r.escalated
    assert "strong" not in [m for m, _ in chat.log]


def test_single_tier_never_calls_the_verifier():
    chat = FakeChat([])
    r = run(chat, tiers=("strong",))
    assert r.answered_by == "strong" and not r.escalated and len(r.calls) == 1


def test_no_tiers_is_an_error():
    with pytest.raises(ValueError):
        run(FakeChat([]), tiers=())


@pytest.mark.parametrize(
    ("text", "ok"),
    [
        (PASS, True),
        (FAIL, False),
        ('Sure! Here is my verdict:\n{"supported": true, "answers": true, "reason": "x"}', True),
        ('{"supported": "true", "answers": true, "reason": "x"}', False),   # string, not boolean
        ('{"supported": true}', False),                                     # missing key
        ("I think it's fine", False),                                       # no JSON at all
        ("", False),
    ],
)
def test_parse_verdict_fails_closed(text, ok):
    assert parse_verdict(text).ok is ok


def test_unparseable_verifier_output_escalates():
    r = run(FakeChat(["<html>502 Bad Gateway</html>"]))
    assert r.escalated and r.answered_by == "strong"


def test_parse_grade_fails_closed():
    assert parse_grade('{"pass": true, "reason": "same facts"}') == (True, "same facts")
    assert parse_grade('{"pass": "yes", "reason": "x"}')[0] is False
    assert parse_grade("PASS")[0] is False


def test_parse_cost_prefers_gateway_header():
    assert parse_cost({"x-litellm-response-cost": "0.0025"}, {"cost": 9.0}) == 0.0025
    assert parse_cost({"x-litellm-response-cost": "None"}, {"cost": 0.004}) == 0.004
    assert parse_cost({}, None) == 0.0
    assert parse_cost(None, {"cost": 0.004}) == 0.004


def test_gateway_cache_hit_costs_nothing_even_with_stale_usage_cost():
    # A hit carries no cost header. Whatever `usage` the cached body holds describes the
    # original call, not this one, so it must never be counted again.
    assert parse_cost({"x-litellm-cache-key": "596f5c"}, {"cost": 0.0013}) == 0.0


def test_static_prefix_carries_the_cache_breakpoint_and_question_comes_last():
    msgs = answer_messages("SYSTEM PROMPT", "How long do refunds take?")
    assert msgs[0]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    assert msgs[-1] == {"role": "user", "content": "How long do refunds take?"}


def test_plain_text_messages_flattens_blocks_without_mutating_input():
    msgs = answer_messages("SYSTEM PROMPT", "q?")
    flat = plain_text_messages(msgs)
    assert flat[0] == {"role": "system", "content": "SYSTEM PROMPT"}
    assert isinstance(msgs[0]["content"], list)
