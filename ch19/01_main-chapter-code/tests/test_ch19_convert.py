import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ch19_ft import convert

RECORD = {
    "id": 3, "subject": "Item arrived broken", "channel": "chat",
    "body": "My Max Cutting Board (order #370) arrived cracked. <b>Photos</b> attached.",
    "category": "product_defect", "priority": "high", "order_id": 370, "hard": False,
    "sentiment": "negative", "summary": "The cutting board arrived cracked.",
}


class _WordEncoding:
    """Stand-in for tiktoken (no download in unit tests): one token per word."""

    def encode(self, text):
        return text.split()


@pytest.fixture(autouse=True)
def offline_tokens(monkeypatch):
    monkeypatch.setattr(convert, "_encoding", lambda: _WordEncoding())


def lines(rows):
    return [json.dumps(r) for r in rows]


def record(**changes):
    return {**RECORD, **changes}


@pytest.mark.parametrize("provider", convert.FORMATS)
def test_every_format_round_trips_to_the_same_three_strings(provider):
    system, turns = convert.PARSERS[provider](convert.CONVERTERS[provider](RECORD))
    assert system.startswith("You triage customer-support tickets for ShopFlow")
    assert [role for role, _ in turns] == ["user", "assistant"]
    assert "&lt;b&gt;Photos&lt;/b&gt;" in turns[0][1]  # the ticket body is escaped, as in ch18
    assert json.loads(turns[1][1]) == {"category": "product_defect", "priority": "high",
                                       "sentiment": "negative", "order_id": 370,
                                       "summary": "The cutting board arrived cracked."}
    assert (system, turns) == convert.parse_openai(convert.to_openai(RECORD))


@pytest.mark.parametrize("provider", convert.FORMATS)
def test_a_clean_dataset_passes(provider):
    rows = [convert.CONVERTERS[provider](record(id=i, subject=f"Item {i} broke"))
            for i in range(12)]
    rep = convert.validate_rows(provider, lines(rows))
    assert rep.ok and rep.n == 12 and not rep.warnings
    assert len(rep.tokens) == 12


def test_vertex_rejects_the_openai_role_name_with_a_hint():
    row = convert.to_vertex(RECORD)
    row["contents"][1]["role"] = "assistant"
    with pytest.raises(convert.FormatError, match="Vertex calls it 'model'") as e:
        convert.parse_vertex(row)
    assert e.value.code == "bad_role"


def test_bedrock_needs_a_schema_version_string():
    row = convert.to_bedrock(RECORD)
    row["schemaVersion"] = 2024
    with pytest.raises(convert.FormatError) as e:
        convert.parse_bedrock(row)
    assert e.value.code == "bad_schema"
    del row["schemaVersion"]
    with pytest.raises(convert.FormatError) as e:
        convert.parse_bedrock(row)
    assert e.value.code == "missing_key"


@pytest.mark.parametrize("word", ["Assistant:", "[EOS]", "System:"])
def test_nova_reserved_words_fail_bedrock_but_not_the_other_formats(word):
    rec = record(body=f"Copy-pasted from a chat log. {word} we will refund you. Order #370.")
    assert convert.validate_rows("openai", lines([convert.to_openai(rec)] * 10)).ok
    assert convert.validate_rows("vertex", lines([convert.to_vertex(rec)] * 10)).ok
    rep = convert.validate_rows("bedrock", lines([convert.to_bedrock(rec)] * 10))
    assert {p.code for p in rep.errors} == {"reserved_text"} and len(rep.errors) == 10


def test_bedrock_content_must_be_text_blocks_not_a_string():
    row = convert.to_bedrock(RECORD)
    row["messages"][0]["content"] = "plain string, OpenAI style"
    with pytest.raises(convert.FormatError) as e:
        convert.parse_bedrock(row)
    assert e.value.code == "bad_content"


def test_unknown_keys_are_rejected_not_ignored():
    row = convert.to_openai(RECORD)
    row["metadata"] = {"id": 3}
    with pytest.raises(convert.FormatError) as e:
        convert.parse_openai(row)
    assert e.value.code == "unknown_key"


@pytest.mark.parametrize(("mutate", "code"), [
    (lambda m: m.append(m.pop(0)), "role_order"),                        # system not first
    (lambda m: m.pop(), "no_assistant"),                                   # ends on the user
    (lambda m: m.insert(2, {"role": "user", "content": "hi"}), "role_order"),  # user, user
    (lambda m: m[1].update(content="   "), "empty_text"),
    (lambda m: m[1].update(role="tool"), "bad_role"),
    (lambda m: m[1].update(weight=1), "bad_weight"),                       # weight on a user turn
    (lambda m: m[2].update(content='{"category": "billing"}'), "bad_answer"),
    (lambda m: m[2].update(content="```json\n" + m[2]["content"] + "\n```"), "bad_answer"),
])
def test_openai_structural_errors_get_a_code_and_a_line(mutate, code):
    good = convert.to_openai(RECORD)
    bad = convert.to_openai(record(id=4))
    mutate(bad["messages"])
    rows = lines([good] * 5 + [bad] + [good] * 5)
    rep = convert.validate_rows("openai", ["", *rows])  # a blank first line still counts
    assert [(p.line, p.code) for p in rep.errors] == [(7, code)]
    assert rep.n == 11


def test_bad_json_is_reported_and_the_rest_still_checked():
    rows = lines([convert.to_openai(RECORD)] * 10)
    rows.insert(3, '{"messages": [')
    rep = convert.validate_rows("openai", rows)
    assert [(p.line, p.code) for p in rep.errors] == [(4, "bad_json")]
    assert len(rep.tokens) == 10


def test_duplicates_and_conflicts_are_warnings():
    a = convert.to_openai(RECORD)
    other_answer = convert.to_openai(record(summary="Cracked board, customer upset."))
    rep = convert.validate_rows("openai", lines([a] * 9 + [a, other_answer]))
    codes = [(p.line, p.code) for p in rep.warnings]
    assert codes[:8] == [(i, "duplicate") for i in range(2, 10)]
    assert codes[-2:] == [(10, "duplicate"), (11, "conflict")]
    assert rep.ok


def test_dataset_size_limits_per_provider_and_split():
    row = convert.to_bedrock(RECORD)
    assert [p.code for p in convert.validate_rows("bedrock", lines([row] * 7)).errors] == [
        "too_few"]
    assert convert.validate_rows("bedrock", lines([row] * 8)).ok
    vrow = convert.to_vertex(RECORD)
    limit = convert.LIMITS["vertex"].max_val_examples
    over = convert.validate_rows("vertex", lines([vrow] * (limit + 1)), split="val")
    assert [p.code for p in over.errors] == ["too_many"]
    assert convert.validate_rows("vertex", lines([vrow] * (limit + 1)), split="train").ok


def test_too_long_uses_the_provider_token_limit(monkeypatch):
    monkeypatch.setitem(convert.LIMITS, "openai",
                        convert.Limits(min_examples=1, max_examples=None,
                                       max_tokens_per_example=40, max_file_mb=1))
    n = convert.count_tokens(convert.parse_openai(convert.to_openai(RECORD)))
    assert n > 40
    rep = convert.validate_rows("openai", lines([convert.to_openai(RECORD)]))
    assert [p.code for p in rep.errors] == ["too_long"]


def test_token_count_adds_message_overhead():
    canonical = ("a b", [("user", "c"), ("assistant", "d e f")])
    assert convert.count_tokens(canonical) == (3 + 2) + (3 + 1) + (3 + 3) + 3


def test_dedupe_keeps_the_first_copy_only():
    rows = [record(id=1), record(id=2), record(id=3, summary="Different.")]
    kept, dropped = convert.dedupe(rows)
    assert [r["id"] for r in kept] == [1, 3] and dropped == 1


def test_leakage_ignores_case_and_surrounding_space():
    train = [{"subject": "Where is my order?", "body": "Order #1 is late."}]
    test = [{"id": 7, "subject": " where is my order?", "body": "ORDER #1 IS LATE."},
            {"id": 8, "subject": "Where is my order?", "body": "Order #2 is late."}]
    assert convert.leakage(train, test) == [7]


def test_convert_all_writes_valid_files_for_every_provider(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    recs = [record(id=i, summary=f"Summary {i}.") for i in range(12)]
    (data / "train.jsonl").write_text("\n".join(json.dumps(r) for r in recs + recs[:2]) + "\n")
    (data / "val.jsonl").write_text("\n".join(json.dumps(r) for r in recs[:10]) + "\n")
    reports = convert.convert_all(tmp_path / "out", data)
    for provider in convert.FORMATS:
        train, val = reports[provider]
        assert train.ok and val.ok
        assert (train.n, val.n) == (12, 10)  # the two duplicates were dropped
        assert (tmp_path / "out" / provider / "train.jsonl").exists()


def test_segments_split_templated_tickets_by_exact_text_and_headline_excludes_identical():
    from ch19_ft.evaluate import segments

    train = [{"subject": "Store opening hours", "body": "When are you open?"}]
    test = [{"id": 1, "subject": "store opening hours ", "body": "When are you open?", "hard": False},
            {"id": 2, "subject": "Late order", "body": "Order #9 is late.", "hard": False},
            {"id": 3, "subject": "Store opening hours", "body": "When are you open?", "hard": True}]
    seg = segments(test, train)
    assert seg["identical"] == {1}
    assert seg["unseen"] == {2}
    assert seg["hard"] == {3}  # hand-written tickets are never "identical", even on a text match
    assert seg["headline"] == {2, 3} and seg["all"] == {1, 2, 3}
