"""Eval metrics over saved review runs (no network)."""

from __future__ import annotations

from .match import match_comments
from .postprocess import filter_comments
from .schemas import ReviewComment


def score_pr(comments: list[ReviewComment], bugs: list[dict], *, min_confidence: float,
             window: int = 3, require_category: bool = True) -> dict:
    """Recall and false positives for one review of one PR at one confidence threshold."""
    kept = filter_comments(comments, min_confidence=min_confidence)
    m = match_comments(kept, bugs, window=window, require_category=require_category)
    return {"posted": len(kept), "found": sorted(m.found), "missed": m.missed,
            "false_positives": len(m.unmatched), "recall": m.recall}


def line_offsets(comments: list[ReviewComment], bugs: list[dict], window: int = 10) -> dict:
    """For each bug a comment talks about, how far its line is from the bug (signed, 0 = on it).

    Uses a wide window so that a comment describing the right bug on the wrong line still
    pairs up; the offset then shows how badly the model counted.
    """
    m = match_comments(comments, bugs, window=window, require_category=True)
    by_id = {b["id"]: b for b in bugs}
    out = {}
    for bug_id, c in m.found.items():
        b = by_id[bug_id]
        lo, hi = b.get("start", b["line"]), b.get("end", b["line"])
        out[bug_id] = 0 if lo <= c.line <= hi else (c.line - lo if c.line < lo else c.line - hi)
    return out


def score_run_file(runs: list[dict], gt: dict, *, min_confidence: float, window: int = 3,
                   require_category: bool = True) -> dict:
    """Aggregate saved runs (each has a buggy-PR and a clean-PR review) into mean metrics.

    Returns mean recall on the buggy PR, mean false positives on the buggy PR and on the clean
    PR, and precision = true positives / posted comments on the buggy PR.
    """
    n = len(runs)
    recall = fp_buggy = fp_clean = tp = posted = 0.0
    found_counts: dict[str, int] = {b["id"]: 0 for b in gt["bugs"]}
    for run in runs:
        buggy = [ReviewComment.model_validate(c) for c in run["buggy"]["comments"]]
        clean = [ReviewComment.model_validate(c) for c in run["clean"]["comments"]]
        sb = score_pr(buggy, gt["bugs"], min_confidence=min_confidence, window=window,
                      require_category=require_category)
        sc = score_pr(clean, [], min_confidence=min_confidence)
        recall += sb["recall"]
        fp_buggy += sb["false_positives"]
        fp_clean += sc["false_positives"]
        tp += len(sb["found"])
        posted += sb["posted"]
        for b in sb["found"]:
            found_counts[b] += 1
    return {
        "recall": recall / n,
        "precision": tp / posted if posted else 0.0,
        "fp_per_buggy_pr": fp_buggy / n,
        "fp_per_clean_pr": fp_clean / n,
        "posted_per_buggy_pr": posted / n,
        "found_counts": found_counts,
        "runs": n,
    }
