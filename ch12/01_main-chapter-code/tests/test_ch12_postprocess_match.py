import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ch12_review.diff import DiffLine, FileDiff, Hunk
from ch12_review.evaluate import line_offsets, score_pr, score_run_file
from ch12_review.match import match_comments
from ch12_review.postprocess import dedupe, filter_comments, map_to_diff
from ch12_review.reviewer import parse_result
from ch12_review.schemas import RESPONSE_FORMAT, ReviewComment


def c(path="a.py", line=10, category="correctness", conf=0.8, sev="medium", body="x is wrong"):
    return ReviewComment(path=path, line=line, severity=sev, category=category, body=body,
                         confidence=conf)


# ------------------------------------------------------------------ dedupe
def test_dedupe_keeps_the_most_confident_of_a_nearby_same_category_pair():
    kept = dedupe([c(line=10, conf=0.6), c(line=11, conf=0.9, body="other words")])
    assert len(kept) == 1 and kept[0].confidence == 0.9


def test_dedupe_line_window_is_inclusive_at_two():
    assert len(dedupe([c(line=10), c(line=12)])) == 1
    assert len(dedupe([c(line=10), c(line=13)])) == 2


def test_dedupe_never_merges_across_files():
    assert len(dedupe([c(path="a.py"), c(path="b.py")])) == 2


def test_dedupe_different_category_and_different_text_are_both_kept():
    a = c(category="security", body="query built from user input")
    b = c(category="performance", body="loop issues one query per row")
    assert len(dedupe([a, b])) == 2


def test_dedupe_same_text_different_category_is_one_finding():
    a = c(category="security", body="the status value is interpolated into the SQL string")
    b = c(category="correctness", body="the status value is interpolated into the SQL string")
    assert len(dedupe([a, b])) == 1


# ------------------------------------------------------------------ filter
def test_filter_threshold_is_inclusive():
    kept = filter_comments([c(conf=0.6), c(path="b.py", conf=0.59)], min_confidence=0.6)
    assert [k.path for k in kept] == ["a.py"]


def test_filter_severity_floor_and_cap_keeps_the_best():
    cs = [c(path=f"f{i}.py", conf=0.5 + i / 20, sev="high" if i else "low") for i in range(6)]
    kept = filter_comments(cs, min_confidence=0.0, min_severity="medium", max_comments=3)
    assert [k.path for k in kept] == ["f3.py", "f4.py", "f5.py"]  # f0 is low; best 3 of the rest


def test_confidence_outside_0_1_is_clamped_by_map_to_diff():
    fd = _filediff("a.py", [10])
    out = map_to_diff([c(conf=1.7), c(conf=-3, line=10, body="y")], [fd])
    assert [o.confidence for o in out] == [1.0, 0.0]


# ------------------------------------------------------------------ map_to_diff
def _filediff(path, new_lines):
    lines = [DiffLine("+", "x", None, n, i + 1) for i, n in enumerate(new_lines)]
    return FileDiff(path=path, old_path=path,
                    hunks=[Hunk(1, 0, new_lines[0], len(new_lines), "", 0, lines)])


def test_map_to_diff_drops_unknown_files_and_far_lines_and_snaps_near_ones():
    fd = _filediff("a.py", [10, 11, 12])
    out = map_to_diff([c(path="ghost.py", line=10), c(line=40), c(line=13)], [fd])
    assert [(o.path, o.line) for o in out] == [("a.py", 12)]


# ------------------------------------------------------------------ matching
BUGS = [
    {"id": "sqli", "file": "a.py", "line": 27, "categories": ["security"]},
    {"id": "race", "file": "b.py", "line": 24, "categories": ["concurrency", "correctness"]},
]


def test_match_window_boundary():
    assert "sqli" in match_comments([c(line=30, category="security")], BUGS, window=3).found
    assert "sqli" not in match_comments([c(line=31, category="security")], BUGS, window=3).found


def test_multi_line_bug_matches_anywhere_in_its_region_plus_window():
    race = [{"id": "race", "file": "a.py", "line": 24, "start": 19, "end": 24,
             "categories": ["concurrency"]}]
    assert "race" in match_comments([c(line=19, category="concurrency")], race, window=3).found
    assert "race" in match_comments([c(line=16, category="concurrency")], race, window=3).found
    assert "race" not in match_comments([c(line=15, category="concurrency")], race, window=3).found
    assert "race" in match_comments([c(line=27, category="concurrency")], race, window=3).found
    assert "race" not in match_comments([c(line=28, category="concurrency")], race, window=3).found


def test_pairing_prefers_the_bug_whose_region_contains_the_comment():
    two = [{"id": "far", "file": "a.py", "line": 12, "categories": ["correctness"]},
           {"id": "span", "file": "a.py", "line": 20, "start": 10, "end": 20,
            "categories": ["correctness"]}]
    m = match_comments([c(line=13)], two)
    assert list(m.found) == ["span"]  # inside span's region (distance 0) beats 1 line from "far"


def test_match_requires_category_unless_lenient():
    wrong = [c(line=27, category="style")]
    assert match_comments(wrong, BUGS).found == {}
    assert "sqli" in match_comments(wrong, BUGS, require_category=False).found


def test_match_requires_the_same_file():
    assert match_comments([c(path="z.py", line=27, category="security")], BUGS).found == {}


def test_one_comment_cannot_claim_two_bugs_and_one_bug_is_found_once():
    close = [{"id": "b1", "file": "a.py", "line": 10, "categories": ["correctness"]},
             {"id": "b2", "file": "a.py", "line": 12, "categories": ["correctness"]}]
    m = match_comments([c(line=11)], close)
    assert len(m.found) == 1 and len(m.missed) == 1
    m2 = match_comments([c(line=10), c(line=10, body="dup")], close[:1])
    assert len(m2.found) == 1 and len(m2.unmatched) == 1


def test_closest_comment_wins_and_the_other_is_a_false_positive():
    m = match_comments([c(line=12, conf=0.99), c(line=10, conf=0.5)],
                       [{"id": "b1", "file": "a.py", "line": 10, "categories": ["correctness"]}])
    assert m.found["b1"].line == 10
    assert [u.line for u in m.unmatched] == [12]
    assert m.precision == 0.5 and m.recall == 1.0


def test_empty_inputs_do_not_divide_by_zero():
    m = match_comments([], BUGS)
    assert m.recall == 0.0 and m.precision == 0.0 and m.missed == ["sqli", "race"]
    assert match_comments([], []).recall == 0.0


def test_line_offsets_are_signed_distances_to_the_bug_region():
    race = {"id": "race", "file": "b.py", "line": 24, "start": 19, "end": 24,
            "categories": ["concurrency"]}
    comments = [c(path="a.py", line=21, category="security"),   # 6 above sqli (line 27)
                c(path="b.py", line=22, category="concurrency"),  # inside the race region
                c(path="b.py", line=40, category="concurrency")]  # 16 below: too far to pair
    assert line_offsets(comments, [BUGS[0], race], window=10) == {"sqli": -6, "race": 0}
    assert line_offsets([c(path="a.py", line=29, category="security")], BUGS) == {"sqli": 2}


# ------------------------------------------------------------------ scoring over thresholds
def test_threshold_trades_recall_for_fewer_false_positives():
    good = c(path="a.py", line=27, category="security", conf=0.9)
    shaky = c(path="b.py", line=24, category="concurrency", conf=0.5)
    noise = c(path="a.py", line=99, conf=0.55)
    comments = [good, shaky, noise]
    lo = score_pr(comments, BUGS, min_confidence=0.4)
    hi = score_pr(comments, BUGS, min_confidence=0.6)
    assert (lo["recall"], lo["false_positives"]) == (1.0, 1)
    assert (hi["recall"], hi["false_positives"]) == (0.5, 0)


def test_score_run_file_averages_runs_and_counts_clean_pr_false_positives():
    run_a = {"buggy": {"comments": [c(path="a.py", line=27, category="security").model_dump()]},
             "clean": {"comments": [c(conf=0.9).model_dump()]}}
    run_b = {"buggy": {"comments": []}, "clean": {"comments": []}}
    gt = {"bugs": BUGS}
    s = score_run_file([run_a, run_b], gt, min_confidence=0.5)
    assert s["recall"] == pytest.approx(0.25)  # run a: 1/2, run b: 0/2
    assert s["fp_per_clean_pr"] == 0.5
    assert s["precision"] == 1.0
    assert s["found_counts"] == {"sqli": 1, "race": 0}


# ------------------------------------------------------------------ model output parsing
def test_parse_result_accepts_prose_around_json_and_rejects_garbage():
    body = '{"comments": [{"path": "a.py", "line": 3, "severity": "low", "category": "style", ' \
           '"body": "b", "confidence": 0.5}]}'
    assert parse_result(f"Sure! Here you go:\n{body}\nHope it helps").comments[0].line == 3
    with pytest.raises(ValidationError):
        parse_result("no json here at all")
    with pytest.raises(ValidationError):
        parse_result('{"comments": [{"path": "a.py"}]}')


def test_strict_schema_requires_every_property_even_in_nested_defs():
    schema = RESPONSE_FORMAT["json_schema"]["schema"]
    assert schema["additionalProperties"] is False
    item = schema["$defs"]["ReviewComment"]
    assert item["additionalProperties"] is False
    assert set(item["required"]) == set(item["properties"])
