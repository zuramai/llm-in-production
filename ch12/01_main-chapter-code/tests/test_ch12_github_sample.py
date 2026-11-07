"""Payload building, the post_comments dry run, and the sample repo's ground truth."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))

from ch12_review.diff import parse_diff
from ch12_review.github import build_review_payload
from ch12_review.reviewer import git_diff
from ch12_review.schemas import ReviewComment


def rc(path, line, body="problem"):
    return ReviewComment(path=path, line=line, severity="high", category="security", body=body,
                         confidence=0.9)


@pytest.fixture(scope="module")
def sample(tmp_path_factory):
    """Build the sample repo exactly as the chapter does, into a temp dir."""
    root = tmp_path_factory.mktemp("sample")
    repo, gt = root / "repo", root / "gt.json"
    subprocess.run(["bash", str(CODE / "make_sample_pr.sh"), str(repo), str(gt)], check=True,
                   capture_output=True)
    return repo, json.loads(gt.read_text())


def test_ground_truth_has_exactly_eight_bugs_and_enough_innocuous_changes(sample):
    _, gt = sample
    assert len(gt["bugs"]) == 8
    assert {b["id"] for b in gt["bugs"]} == {
        "sql_injection", "off_by_one", "missing_await", "secret_in_code", "n_plus_one",
        "unchecked_none", "wrong_timezone", "race_condition"}
    assert len(gt["innocuous"]) >= 5


SNIPPETS = {"sql_injection": "{status}", "off_by_one": "page * PAGE_SIZE",
            "missing_await": "notify.send_receipt", "secret_in_code": 'STRIPE_API_KEY = "',
            "n_plus_one": "WHERE customer_id = %s", "unchecked_none": "order['status']",
            "wrong_timezone": "utcnow", "race_condition": "SET on_hand = %s"}


def test_every_ground_truth_line_points_at_the_bug_and_is_commentable(sample):
    repo, gt = sample
    files = {f.path: f for f in parse_diff(git_diff(str(repo), gt["base"], gt["buggy_head"]))}
    for bug in gt["bugs"]:
        head_text = subprocess.run(
            ["git", "-C", str(repo), "show", f"{gt['buggy_head']}:{bug['file']}"],
            check=True, capture_output=True, text=True).stdout.split("\n")
        assert SNIPPETS[bug["id"]] in head_text[bug["line"] - 1], bug
        assert "@@bug" not in "\n".join(head_text), "markers must be stripped before commit"
        assert bug["line"] in files[bug["file"]].added_lines(), bug
        assert bug["start"] <= bug["line"] <= bug["end"], bug
        assert {bug["start"], bug["end"]} <= files[bug["file"]].commentable_lines(), bug


def test_clean_pr_touches_no_bug_lines_and_has_no_bug_code(sample):
    repo, gt = sample
    diff = git_diff(str(repo), gt["base"], gt["clean_head"])
    for needle in SNIPPETS.values():
        assert needle not in "\n".join(ln for ln in diff.split("\n") if ln.startswith("+"))


def test_payload_moves_unresolvable_lines_into_the_summary(sample):
    repo, gt = sample
    files = parse_diff(git_diff(str(repo), gt["base"], gt["buggy_head"]))
    ok_line = next(b["line"] for b in gt["bugs"] if b["id"] == "sql_injection")
    payload = build_review_payload(
        [rc("service/orders.py", ok_line), rc("service/orders.py", 9999, "far away"),
         rc("nope.py", 1, "ghost")], files, commit_id="abc123")
    assert payload["event"] == "COMMENT" and payload["commit_id"] == "abc123"
    assert payload["comments"] == [{"path": "service/orders.py", "line": ok_line, "side": "RIGHT",
                                    "body": payload["comments"][0]["body"]}]
    assert "far away" in payload["body"] and "ghost" in payload["body"]
    assert "not a gate" in payload["body"]


def test_payload_caps_inline_comments(sample):
    repo, gt = sample
    files = parse_diff(git_diff(str(repo), gt["base"], gt["buggy_head"]))
    line = next(b["line"] for b in gt["bugs"] if b["id"] == "sql_injection")
    payload = build_review_payload([rc("service/orders.py", line, f"c{i}") for i in range(5)],
                                   files, max_inline=2)
    assert len(payload["comments"]) == 2 and "c4" in payload["body"]


def test_empty_review_payload():
    payload = build_review_payload([], [])
    assert payload["comments"] == [] and "No issues" in payload["body"]
    assert "commit_id" not in payload


def test_post_comments_dry_run_prints_payload_and_sends_nothing(sample, tmp_path):
    repo, gt = sample
    bug = next(b for b in gt["bugs"] if b["id"] == "off_by_one")
    comments = tmp_path / "comments.json"
    comments.write_text(json.dumps([rc(bug["file"], bug["line"], "page is 1-based").model_dump()]))
    out = subprocess.run(
        [sys.executable, str(CODE / "post_comments.py"), "--comments", str(comments),
         "--repo", str(repo), "--base", gt["base"], "--head", gt["buggy_head"], "--dry-run"],
        check=True, capture_output=True, text=True, env={"PATH": "/usr/bin:/bin"}).stdout
    first, _, rest = out.partition("\n")
    assert first.startswith("POST https://api.github.com/repos/OWNER/REPO/pulls/")
    payload = json.loads(rest)
    assert payload["comments"][0]["line"] == bug["line"]
    assert payload["comments"][0]["side"] == "RIGHT"
