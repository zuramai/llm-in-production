"""Tools, citations and structured-output helpers. No network: the SQL guard rejects before any
database connection is opened."""

import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ch05_agentic import evalset, llm, tools, variants
from ch05_agentic.prev import prev


def test_previous_chapters_loads_under_unique_name():
    assert prev.__name__ == "ch05_previous_chapters"
    assert "previous_chapters" not in sys.modules or sys.modules["previous_chapters"] is not prev


@pytest.mark.parametrize("sql", [
    "DELETE FROM orders",
    "WITH d AS (DELETE FROM orders RETURNING id) SELECT * FROM d",
    "SELECT 1; DROP TABLE orders",
    "SELECT * FROM pg_catalog.pg_roles",
    "SELECT * FROM ch05_doc_chunks",  # our own table isn't on the allow-list for the model
])
def test_run_sql_guard_rejects_before_touching_the_db(sql):
    with pytest.raises(prev.UnsafeSQLError):
        tools.run_sql(sql)


def test_tool_loop_returns_guard_error_to_model_as_evidence():
    evidence = []
    out = variants._call_tool("run_sql", '{"sql": "UPDATE orders SET status = \'paid\'"}',
                              docs=None, evidence=evidence)
    assert "rule=select_only" in out
    assert evidence[0].citation == "[sql:1]" and evidence[0].error


def test_tool_loop_survives_bad_arguments_and_unknown_tools():
    assert variants._call_tool("run_sql", "{not json", None, []).startswith("ERROR")
    assert variants._call_tool("delete_everything", "{}", None, []).startswith("ERROR")


def test_citations_extracted_in_order_and_deduplicated():
    text = "Yes [sql:1]. Policy [doc:policies/returns-policy.md#2] and again [sql:1]."
    assert tools.cited(text) == ["[sql:1]", "[doc:policies/returns-policy.md#2]"]


def test_check_citations_flags_unknown_tags():
    ev = [tools.SqlEvidence("[sql:1]", "q", "SELECT 1", [{"x": 1}])]
    res = tools.check_citations("A [sql:1] B [sql:2] C [doc:faq.md#0]", ev)
    assert res["invalid"] == ["[sql:2]", "[doc:faq.md#0]"]
    assert res["cites_sql"] and not res["cites_doc"]


def test_sql_evidence_render_truncates_rows():
    rows = [{"i": i} for i in range(tools.MAX_ROWS_SHOWN + 5)]
    text = tools.SqlEvidence("[sql:1]", "q", "SELECT i", rows).render()
    assert "(+5 more rows)" in text and '{"i": 25}' not in text


def test_evidence_roundtrip_through_plain_dicts():
    e = tools.DocEvidence("[doc:faq.md#1]", "faq.md", "FAQ", "text", "query")
    assert tools.evidence_from_dict(tools.evidence_dict(e)) == e


class Nested(BaseModel):
    a: str | None
    items: list[dict]


def test_strict_schema_requires_every_property():
    schema = llm.strict_schema(Nested)
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"a", "items"}


def test_parse_json_strips_prose_and_fences():
    class R(BaseModel):
        sql: str

    assert llm.parse_json('Sure!\n```json\n{"sql": "SELECT 1"}\n```', R).sql == "SELECT 1"
    with pytest.raises(ValueError):
        llm.parse_json("no json here", R)


def test_dates_are_english_whatever_the_locale():
    d = datetime(2026, 11, 19, 10, 0, tzinfo=ZoneInfo("America/New_York"))
    assert evalset._d(d.date()) == "2026-11-19"
    assert tools.now_text(d).startswith("Thursday 2026-11-19 10:00")


def test_question_file_has_25_unique_questions_with_gold():
    _, qs = evalset.load()
    assert len(qs) == 25
    assert len({q.id for q in qs}) == 25
    assert all(q.gold and q.docs for q in qs)


def test_reranker_falls_back_to_cpu_when_shared_gpu_is_full(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (87 << 20, 24 << 30))
    assert prev.rerank_device() == "cpu"
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (8 << 30, 24 << 30))
    assert prev.rerank_device() == "cuda"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert prev.rerank_device() == "cpu"


def test_sql_planner_garbage_becomes_evidence_not_a_crash():
    """A planner reply cut off at max_tokens used to raise and abort the whole graph run."""

    class Truncating:
        def structured(self, messages, schema, usage, step, **kw):
            raise ValueError('no JSON object in reply: \'{"sql":"SELECT o.id, o.subtotal_cen\'')

    ev = tools.ask_sql("order 1: everything", Truncating(), llm.Usage(), "[sql:1]", "today")
    assert ev.citation == "[sql:1]" and ev.rows == []
    assert ev.error.startswith("planner reply invalid")
    assert "ERROR" in ev.render()
