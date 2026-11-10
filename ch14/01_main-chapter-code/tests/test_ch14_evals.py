import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

CODE = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def evals():
    """Import ch14_evals without leaking our `previous_chapters` into other chapters' tests.

    Every chapter has a `previous_chapters.py`; whichever is imported first would win in
    sys.modules for the whole pytest process. We import ours, then put things back.
    """
    saved = sys.modules.pop("previous_chapters", None)
    sys.path.insert(0, str(CODE))
    try:
        module = importlib.import_module("ch14_evals")
        yield module
    finally:
        sys.modules.pop("previous_chapters", None)
        if saved is not None:
            sys.modules["previous_chapters"] = saved
        sys.path.remove(str(CODE))


def test_load_cases_covers_the_chapter6_eval_set(evals):
    cases = evals.load_cases("America/New_York")
    assert len(cases) == 30
    assert len({c.id for c in cases}) == 30
    sql = [c for c in cases if c.metric == "sql"]
    judged = [c for c in cases if c.metric == "judge"]
    assert len(sql) == 19 and len(judged) == 11
    assert all(c.meta["category"] in evals.ANSWERABLE for c in sql)
    assert not any(c.meta["category"] in evals.ANSWERABLE for c in judged)
    # {tz} must be substituted, or gold SQL silently runs in the session time zone.
    assert not any("{tz}" in c.expected for c in sql)
    assert any("America/New_York" in c.expected for c in sql)


def test_case_item_round_trip(evals):
    for case in evals.load_cases("America/New_York"):
        item = evals.case_to_item(case)
        assert item["id"] == f"sqlbot-eval-{case.id}"
        back = evals.item_to_case(item["input"], item["expected_output"], item["metadata"])
        assert (back.id, back.input, back.expected, back.metric, back.meta) == (
            case.id, case.input, case.expected, case.metric, case.meta)


def test_evaluator_scores_crashes_as_zero_without_calling_the_scorer(evals):
    called = []
    evaluator = evals.as_langfuse_evaluator("correct", lambda c, o: called.append(c) or 1.0)
    out = evaluator(input={"question": "q"}, output={"error": "APITimeoutError: boom"},
                    expected_output={"gold_sql": "SELECT 1"}, metadata={"case_id": "s1"})
    assert out.value == 0.0 and "APITimeoutError" in out.comment
    assert called == []


def test_evaluator_passes_an_evalkit_case_to_the_scorer(evals):
    seen = []

    def scorer(case, output):
        seen.append(case)
        return True

    evaluator = evals.as_langfuse_evaluator("correct", scorer)
    out = evaluator(input={"question": "Delete all orders"},
                    output={"answer": "No.", "_judge": {"score": 2, "reason": "refused"}},
                    expected_output={"expected_behavior": "refuse"},
                    metadata={"case_id": "a1", "category": "adversarial"})
    assert out.value == 1.0 and out.comment == "refused"
    assert (seen[0].id, seen[0].metric, seen[0].expected) == ("a1", "judge", "refuse")


def _item_result(category, value):
    return SimpleNamespace(item=SimpleNamespace(metadata={"category": category}),
                           evaluations=[SimpleNamespace(name="correct", value=value)])


def test_accuracy_by_group_splits_execution_and_behavior(evals):
    results = [_item_result("simple", 1.0), _item_result("join", 0.0),
               _item_result("time_window", 1.0), _item_result("adversarial", 1.0),
               _item_result("ambiguous", 0.0)]
    scores = {e.name: e.value for e in evals.accuracy_by_group(item_results=results)}
    assert scores == pytest.approx({"accuracy": 0.6, "execution_accuracy": 2 / 3,
                                    "behavior_accuracy": 0.5})


def test_accuracy_by_group_counts_missing_evaluation_as_failure(evals):
    missing = SimpleNamespace(item=SimpleNamespace(metadata={"category": "simple"}),
                              evaluations=[])
    scores = {e.name: e.value for e in evals.accuracy_by_group(
        item_results=[missing, _item_result("simple", 1.0)])}
    assert scores == {"accuracy": 0.5, "execution_accuracy": 0.5}
