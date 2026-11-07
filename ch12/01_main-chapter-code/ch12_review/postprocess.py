"""Clean up raw model comments: snap lines into the diff, dedupe, filter, rank."""

from __future__ import annotations

import re

from .diff import FileDiff, snap_line
from .schemas import SEVERITY_RANK, ReviewComment

_WORD = re.compile(r"[a-z0-9_]+")


def clamp_confidence(c: ReviewComment) -> ReviewComment:
    return c.model_copy(update={"confidence": min(1.0, max(0.0, c.confidence))})


def map_to_diff(comments: list[ReviewComment], files: list[FileDiff]) -> list[ReviewComment]:
    """Keep only comments on files in the diff, with lines moved onto commentable lines.

    Comments the model aimed at a file that isn't in the PR, or at a line more than two lines
    from any changed line, are dropped: GitHub would reject them anyway.
    """
    by_path = {f.path: f for f in files}
    out = []
    for c in comments:
        fd = by_path.get(c.path)
        if fd is None:
            continue
        line = snap_line(fd, c.line)
        if line is not None:
            out.append(clamp_confidence(c).model_copy(update={"line": line}))
    return out


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def _jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    return len(ta & tb) / len(ta | tb) if ta | tb else 0.0


def is_duplicate(a: ReviewComment, b: ReviewComment, line_window: int = 2,
                 text_similarity: float = 0.5) -> bool:
    """Same file and nearby lines, and either the same category or near-identical text."""
    if a.path != b.path or abs(a.line - b.line) > line_window:
        return False
    return a.category == b.category or _jaccard(a.body, b.body) >= text_similarity


def dedupe(comments: list[ReviewComment], **kw) -> list[ReviewComment]:
    """Collapse duplicates, keeping the most confident (then most severe) of each group.

    Overlapping hunks or the same file reviewed twice produce the same finding more than once.
    """
    ranked = sorted(comments, key=lambda c: (-c.confidence, -SEVERITY_RANK[c.severity]))
    kept: list[ReviewComment] = []
    for c in ranked:
        if not any(is_duplicate(c, k, **kw) for k in kept):
            kept.append(c)
    return sorted(kept, key=lambda c: (c.path, c.line))


def filter_comments(comments: list[ReviewComment], min_confidence: float = 0.6,
                    min_severity: str = "low", max_comments: int | None = None
                    ) -> list[ReviewComment]:
    """Drop low-confidence / low-severity comments and cap the total, best first.

    The cap keeps the *most* confident comments: a bot that posts 25 comments gets muted.
    """
    floor = SEVERITY_RANK[min_severity]
    kept = [c for c in comments
            if c.confidence >= min_confidence and SEVERITY_RANK[c.severity] >= floor]
    kept.sort(key=lambda c: (-c.confidence, -SEVERITY_RANK[c.severity], c.path, c.line))
    if max_comments is not None:
        kept = kept[:max_comments]
    return sorted(kept, key=lambda c: (c.path, c.line))
