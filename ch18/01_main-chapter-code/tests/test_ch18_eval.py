import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ch18_data import FT_RULES, previous, to_example
from ch18_eval import (
    fewshot_rules,
    local_cost_per_1k,
    request_options,
    segment_scores,
    tag_for,
)


def rec(i, hard=False):
    return {"id": i, "subject": "s", "body": "b", "channel": "chat", "category": "refund",
            "priority": "high", "order_id": None, "hard": hard, "sentiment": "neutral",
            "summary": "x"}


def test_served_prompt_is_the_trained_prompt():
    """The adapter must see at inference exactly the system prompt it was trained with."""
    p = previous()
    opts = request_options("vllm", "q17-lora", "ft")
    kw = p.build_request("json_schema", subject="s", body="b", channel="chat", model="m",
                         rules=opts["rules"], extra=opts["extra"])
    trained = to_example(rec(1))["messages"]
    assert kw["messages"][0] == trained[0]
    assert kw["messages"][1] == trained[1]
    # and the unconstrained variant (prompt_only appends the same format hint) matches too
    kw2 = p.build_request("prompt_only", subject="s", body="b", channel="chat", model="m",
                          rules=FT_RULES)
    assert kw2["messages"][0] == trained[0]


def test_vllm_requests_turn_thinking_off_and_decode_greedily():
    opts = request_options("vllm", "Qwen/Qwen3-1.7B", "ch02")
    assert opts["extra"]["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert opts["extra"]["temperature"] == 0
    assert "rules" not in opts  # Chapter 2's full rulebook is build_request's default


def test_gpt_oss_gets_a_reasoning_budget():
    opts = request_options("ollama", "gpt-oss:20b", "ch02")
    assert opts["extra"] == {"reasoning_effort": "low"} and opts["max_tokens"] >= 1000


def test_unknown_prompt_and_endpoint_are_rejected():
    with pytest.raises(ValueError):
        request_options("vllm", "m", "few-shot")
    with pytest.raises(ValueError):
        request_options("sglang", "m", "ft")


def test_local_cost_per_1k():
    # $0.36/h at 10 tickets/s: 1,000 tickets take 100 s = $0.01
    assert local_cost_per_1k(10, 0.36) == pytest.approx(0.01)
    with pytest.raises(ValueError):
        local_cost_per_1k(0, 0.36)


def test_segment_scores_separate_seen_unseen_and_hard():
    p = previous()
    # ticket 1 is word-for-word in train (differing only in case/whitespace); 2 is new text
    records = [dict(rec(1), subject="Hello", body="Same text"), dict(rec(2), body="new text"),
               rec(3, hard=True)]
    train = [dict(rec(50), subject=" hello", body="SAME TEXT "), dict(rec(51), body="other")]

    def res(i, cat):
        triage = None if cat is None else {"category": cat, "priority": "high",
                                           "sentiment": "neutral", "order_id": None,
                                           "summary": "x"}
        return p.Result(i, "json_schema", "m", triage, 1, triage is not None, True, None,
                        latency_ms=100 * i)

    s = segment_scores([res(1, "refund"), res(2, "shipping"), res(3, None)], records, train)
    assert {k: v["n"] for k, v in s.items()} == {"all": 3, "seen": 1, "unseen": 1, "hard": 1,
                                                "headline": 2}
    assert s["seen"]["category_acc_%"] == 100.0
    assert s["unseen"]["category_acc_%"] == 0.0
    assert s["hard"]["category_acc_%"] == 0.0  # a failed parse counts as wrong
    assert s["headline"]["category_acc_%"] == 0.0  # the memorisable ticket is excluded
    assert s["all"]["final_valid_%"] == pytest.approx(66.7)


def test_segments_on_the_committed_split():
    from ch18_data import DATA, read_jsonl
    from ch18_eval import segments

    # "seen" = the 37 test tickets of the 8 identical-text templates (one row of each is in train)
    seg = segments(read_jsonl(DATA / "test.jsonl"), read_jsonl(DATA / "train.jsonl"))
    assert {k: len(v) for k, v in seg.items()} == {"all": 210, "seen": 37, "unseen": 113,
                                                  "hard": 60, "headline": 173}


def test_fewshot_never_uses_a_row_whose_text_is_held_out():
    train = [dict(rec(1), template_id=0, subject="leak", body="same"),
             dict(rec(2), template_id=0, subject="ok", body="fresh")]
    text = fewshot_rules(train, [dict(rec(9), subject="LEAK ", body=" same")])
    assert "subject: ok" in text and "subject: leak" not in text
    assert fewshot_rules(train[:1], [dict(rec(9), subject="leak", body="same")]).count(
        "<ticket>") == 0


def test_fewshot_shows_one_example_per_template_and_only_routing_labels():
    train = [dict(rec(i), template_id=i % 3, subject=f"subj-{i}") for i in range(10, 16)]
    train.append(dict(rec(99, hard=True), template_id=None, subject="hard-one"))
    text = fewshot_rules(train)
    assert text.startswith(previous().RULES)
    assert text.count("<ticket>") == 3
    assert "subj-10" in text and "subj-13" not in text  # the first ticket of each template
    assert "hard-one" not in text
    assert '"summary"' not in text.split("Examples", 1)[1]


def test_tag_is_filesystem_safe():
    assert tag_for("ollama", "gpt-oss:20b", "ch02", 1) == "ollama__gpt-oss_20b__ch02__c1"
    assert "/" not in tag_for("vllm", "Qwen/Qwen3-1.7B", "ch02", 32)
