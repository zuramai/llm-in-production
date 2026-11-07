"""Score a list of review comments against the planted bugs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .schemas import ReviewComment


def load_ground_truth(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


@dataclass
class MatchResult:
    found: dict[str, ReviewComment] = field(default_factory=dict)  # bug id -> matching comment
    missed: list[str] = field(default_factory=list)
    unmatched: list[ReviewComment] = field(default_factory=list)  # false positives
    n_bugs: int = 0
    n_comments: int = 0

    @property
    def recall(self) -> float:
        return len(self.found) / self.n_bugs if self.n_bugs else 0.0

    @property
    def precision(self) -> float:
        return len(self.found) / self.n_comments if self.n_comments else 0.0


def match_comments(comments: list[ReviewComment], bugs: list[dict], window: int = 3,
                   require_category: bool = True) -> MatchResult:
    """Match comments to planted bugs by file, line window and category.

    A bug has a primary `line` and optionally a `start`..`end` region (a race condition is a
    read at one line and a write at another, and a good reviewer may point at either). A comment
    matches when its line is within `window` lines of that region.

    One comment can explain at most one bug, and one bug is found at most once. When several
    comments could match, the pairing with the smallest line distance wins (greedy). Comments
    that match nothing are false positives for this PR.
    """
    candidates = []
    for ci, c in enumerate(comments):
        for b in bugs:
            lo, hi = b.get("start", b["line"]), b.get("end", b["line"])
            if c.path != b["file"] or not lo - window <= c.line <= hi + window:
                continue
            if require_category and c.category not in b["categories"]:
                continue
            dist = 0 if lo <= c.line <= hi else min(abs(c.line - lo), abs(c.line - hi))
            candidates.append((dist, -c.confidence, ci, b["id"]))
    candidates.sort()
    used_comments: set[int] = set()
    found: dict[str, ReviewComment] = {}
    for _dist, _conf, ci, bug_id in candidates:
        if ci in used_comments or bug_id in found:
            continue
        found[bug_id] = comments[ci]
        used_comments.add(ci)
    return MatchResult(
        found=found,
        missed=[b["id"] for b in bugs if b["id"] not in found],
        unmatched=[c for i, c in enumerate(comments) if i not in used_comments],
        n_bugs=len(bugs),
        n_comments=len(comments),
    )
