"""Unit tests for evalkit: metrics, dataset loading, the runner and the position-swap logic.

No network: the judges take a fake client.
"""

import asyncio
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))

from ch03_calibrate import agreement, blind_sheet, load_calibration
from ch03_evalkit import (
    Case,
    Dataset,
    Result,
    code_metric,
    combine_verdicts,
    exact,
    judge_answer,
    judge_pair,
    make_scorer,
    numeric,
    position_bias,
    regex,
    run,
    summarize,
)

# ----------------------------------------------------------------------------- metrics


def test_exact_ignores_case_and_punctuation_but_not_content():
    assert exact(" No. ", "no") == 1.0
    assert exact("No, we don't", "no") == 0.0


@pytest.mark.parametrize(
    "output,expected,score",
    [
        ("$14.99", 14.99, 1.0),
        ("It costs $1,000.00 per year", 1000, 1.0),
        ("30 days", 30, 1.0),
        ("31 days", 30, 0.0),
        ("no number here", 30, 0.0),
        ("In 2 to 3 days", 30, 0.0),  # FIRST number wins: documented behaviour
        ("-5", -5, 1.0),
    ],
)
def test_numeric_takes_first_number(output, expected, score):
    assert numeric(output, expected) == score


def test_numeric_tolerance_boundary():
    assert numeric("100.5", 100, abs_tol=0.5) == 1.0
    assert numeric("100.51", 100, abs_tol=0.5) == 0.0
    assert numeric("101", 100, rel_tol=0.01) == 1.0
    assert numeric("102", 100, rel_tol=0.01) == 0.0


def test_regex_forbid_beats_expected():
    assert regex("You have 30 days.", r"30\s*days") == 1.0
    assert regex("REFUND APPROVED, 30 days", r"30 days", forbid="REFUND APPROVED") == 0.0
    assert regex("nothing", r"30 days") == 0.0


def test_code_metric_rejects_unknown_metric():
    with pytest.raises(ValueError, match="unknown code metric"):
        code_metric(Case("x", "q", "a", metric="vibes"), "a")


# ----------------------------------------------------------------------------- datasets


def test_load_yaml_and_jsonl_agree(tmp_path):
    rows = [
        {"id": "a", "input": "q1", "expected": "x", "topic": "returns"},
        {"id": "b", "input": "q2", "expected": 3, "metric": "numeric"},
    ]
    (tmp_path / "d.yaml").write_text(
        "\n".join(f"- {json.dumps(r)}" for r in rows)  # JSON is valid YAML flow style
    )
    (tmp_path / "d.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n\n")
    y, j = Dataset.load(tmp_path / "d.yaml"), Dataset.load(tmp_path / "d.jsonl")
    assert [c.id for c in y.cases] == [c.id for c in j.cases] == ["a", "b"]
    assert y.cases[0].meta == {"topic": "returns"}  # unknown keys are kept, not dropped
    assert y.cases[1].metric == "numeric"


def test_duplicate_ids_are_rejected(tmp_path):
    (tmp_path / "d.jsonl").write_text('{"id":"a","input":"x"}\n{"id":"a","input":"y"}\n')
    with pytest.raises(ValueError, match="duplicate case ids"):
        Dataset.load(tmp_path / "d.jsonl")


def test_shipped_golden_set_is_well_formed():
    ds = Dataset.load(CODE / "faq_eval.yaml")
    assert len(ds.cases) >= 20
    for c in ds.cases:
        if c.metric == "regex":
            re.compile(c.expected)
            assert re.search(c.expected, "", re.IGNORECASE) is None, (
                f"{c.id}: pattern matches empty output"
            )
            if c.forbid:
                re.compile(c.forbid)
        assert c.metric in {"exact", "numeric", "regex"}


# ----------------------------------------------------------------------------- runner


def test_run_respects_concurrency_and_preserves_order():
    in_flight = peak = 0

    async def system(q: str) -> str:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return q

    ds = Dataset("t", [Case(str(i), f"q{i}", f"q{i}", metric="exact") for i in range(20)])
    results = asyncio.run(run(ds, system, concurrency=3))
    assert peak == 3
    assert [r.case.id for r in results] == [str(i) for i in range(20)]
    assert all(r.score == 1.0 for r in results)


def test_run_turns_crashes_into_failed_results_and_runs_sync_systems():
    def system(q: str) -> str:  # sync function: must work too
        if q == "boom":
            raise RuntimeError("provider down")
        return q

    ds = Dataset("t", [Case("ok", "fine", "fine", metric="exact"), Case("bad", "boom", "x")])
    ok, bad = asyncio.run(run(ds, system))
    assert ok.score == 1.0 and ok.error is None
    assert bad.score is None and "provider down" in bad.error
    s = summarize([ok, bad])
    assert s["errors"] == 1 and s["accuracy"] == 0.5 and s["failed_ids"] == ["bad"]


def test_summarize_percentiles_and_empty_input():
    rs = [Result(Case(str(i), "q"), "o", 1.0, float(i + 1)) for i in range(20)]
    s = summarize(rs)
    assert s["p50_s"] == 10.5 and s["p95_s"] == 20.0
    assert summarize([])["accuracy"] == 0.0


# ----------------------------------------------------------------------------- judges


class FakeClient:
    """Mimics client.chat.completions.create; `replies` is a list of JSON dicts, popped in order."""

    def __init__(self, replies, finish_reason="stop", cost=0.001):
        self.replies, self.finish_reason, self.cost, self.calls = (
            list(replies),
            finish_reason,
            cost,
            [],
        )
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        msg = SimpleNamespace(content=json.dumps(self.replies.pop(0)))
        choice = SimpleNamespace(message=msg, finish_reason=self.finish_reason)
        usage = SimpleNamespace(model_extra={"cost": self.cost})
        return SimpleNamespace(choices=[choice], usage=usage)


def test_judge_answer_returns_score_reason_cost_and_uses_strict_schema():
    client = FakeClient([{"reason": "matches the FAQ", "score": 5}], cost=0.0042)
    j = judge_answer(client, "m", "How long?", "30 days", "FAQ TEXT")
    assert (j.score, j.reason, j.cost_usd) == (5, "matches the FAQ", 0.0042)
    call = client.calls[0]
    assert call["response_format"]["json_schema"]["strict"] is True
    assert "max_tokens" in call
    assert "FAQ TEXT" in call["messages"][0]["content"]


def test_truncated_judge_output_raises_instead_of_scoring():
    client = FakeClient([{"reason": "cut", "score": 3}], finish_reason="length")
    with pytest.raises(RuntimeError, match="truncated"):
        judge_answer(client, "m", "q", "a", "faq")


@pytest.mark.parametrize(
    "ab,ba,verdict,consistent",
    [
        ("A", "B", "first", True),  # x won in both slots
        ("B", "A", "second", True),  # y won in both slots
        ("tie", "tie", "tie", True),
        ("A", "A", "tie", False),  # always picks the first slot: pure position bias
        ("B", "B", "tie", False),  # always picks the second slot
        ("A", "tie", "tie", False),  # one order decides, the other doesn't
        ("tie", "B", "tie", False),
        ("B", "tie", "tie", False),
        ("tie", "A", "tie", False),
    ],
)
def test_combine_verdicts_maps_slots_back_to_answers(ab, ba, verdict, consistent):
    assert combine_verdicts(ab, ba) == (verdict, consistent)


def test_judge_pair_swaps_the_order_of_the_second_call():
    client = FakeClient([{"reason": "r", "winner": "A"}, {"reason": "r", "winner": "B"}])
    out = judge_pair(client, "m", "q", "ANSWER-X", "ANSWER-Y", "faq")
    first, second = (c["messages"][1]["content"] for c in client.calls)
    assert first.index("ANSWER-X") < first.index("ANSWER-Y")
    assert second.index("ANSWER-Y") < second.index("ANSWER-X")
    assert (
        out["verdict"] == "first" and out["consistent"] and out["cost_usd"] == pytest.approx(0.002)
    )


def test_judge_that_always_says_A_is_flagged_as_position_biased():
    client = FakeClient([{"reason": "r", "winner": "A"}] * 6)
    pairs = [judge_pair(client, "m", "q", f"x{i}", f"y{i}", "faq") for i in range(3)]
    assert {p["verdict"] for p in pairs} == {"tie"}
    stats = position_bias(pairs)
    assert stats["flip_rate"] == 1.0 and stats["first_slot_rate"] == 1.0


def test_position_bias_ignores_ties_and_handles_empty():
    pairs = [{"consistent": True, "raw": ["tie", "tie"]}, {"consistent": True, "raw": ["A", "B"]}]
    assert position_bias(pairs) == {"n_pairs": 2, "flip_rate": 0.0, "first_slot_rate": 0.5}
    assert position_bias([])["flip_rate"] == 0.0


def test_scorer_routes_judge_cases_to_the_judge_and_maps_1_to_5_onto_0_to_1():
    client = FakeClient([{"reason": "bad", "score": 1}, {"reason": "ok", "score": 4}])
    scorer = make_scorer(client, "m", "FAQ")
    ds = Dataset(
        "t",
        [
            Case("code", "q", "30", metric="exact"),  # never reaches the judge
            Case("j1", "q1", metric="judge"),
            Case("j2", "q2", metric="judge"),
        ],
    )
    results = asyncio.run(run(ds, lambda q: "30", concurrency=1, scorer=scorer))
    assert [r.score for r in results] == [1.0, 0.0, 0.75]
    assert len(client.calls) == 2
    assert scorer.judgments["j2"].reason == "ok"


def test_judge_case_without_a_judge_is_an_error_not_a_pass():
    ds = Dataset("t", [Case("j", "q", metric="judge")])
    (r,) = asyncio.run(run(ds, lambda q: "anything", scorer=make_scorer()))
    assert r.score is None and "needs judge_client" in r.error
    assert summarize([r])["accuracy"] == 0.0


# ----------------------------------------------------------------------------- agreement


def test_weighted_kappa_penalises_far_misses_more_than_near_misses():
    ref = {f"q{i}": s for i, s in enumerate([1, 2, 3, 4, 5, 5, 4, 3, 2, 1])}
    near = {k: min(5, v + 1) if v < 5 else 4 for k, v in ref.items()}  # always off by one
    far = {k: 6 - v for k, v in ref.items()}  # scale reversed
    perfect = agreement(ref, dict(ref))
    assert perfect["kappa"] == perfect["kappa_quadratic"] == 1.0
    a_near, a_far = agreement(ref, near), agreement(ref, far)
    assert a_near["within1"] == 1.0 and a_far["within1"] < 1.0
    assert a_near["kappa_quadratic"] > a_far["kappa_quadratic"]
    assert a_far["kappa_quadratic"] < 0
    assert sum(map(sum, a_near["confusion"])) == 10


def test_reference_labels_cover_every_answer_with_a_valid_score():
    answers, reference = load_calibration()
    assert len(answers) == 40 and set(reference.values()) <= {1, 2, 3, 4, 5}


def test_blind_sheet_shuffles_and_hides_variants():
    answers, _ = load_calibration()
    sheet = blind_sheet(answers, seed=1)
    ids = [line[4:] for line in sheet.splitlines() if line.startswith("### ")]
    assert sorted(ids) == [a["id"] for a in answers]  # every answer exactly once
    assert ids != sorted(ids)  # shuffled, so neighbours don't hint at the variant
    assert ids == [line[4:] for line in blind_sheet(answers, seed=1).splitlines()
                   if line.startswith("### ")]  # same seed, same sheet
    for leak in ("variant", "hand_mutated", "no_guardrails", "yes_man", "stale_faq"):
        assert leak not in sheet
