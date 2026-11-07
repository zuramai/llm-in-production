"""Parse a unified diff and map every line back to the right place.

Three different "line numbers" are in play in a pull request review, and mixing them up is the
classic bug of review bots:

* the line number in the NEW version of the file (what a human sees on the right side),
* the line number in the OLD version of the file (left side, for deleted lines),
* the GitHub *position*: how many lines below the file's first ``@@`` header a line sits.

We parse the diff once and keep all three on every line.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$")


@dataclass
class DiffLine:
    kind: str  # " " context, "+" added, "-" removed
    text: str
    old_no: int | None
    new_no: int | None
    position: int  # GitHub "position": 1 = the line just below the file's first @@


@dataclass
class Hunk:
    old_start: int
    old_len: int
    new_start: int
    new_len: int
    header: str
    position: int  # position of the @@ line itself (0 for a file's first hunk)
    lines: list[DiffLine] = field(default_factory=list)


@dataclass
class FileDiff:
    path: str  # path on the new side (old path for deleted files)
    old_path: str
    is_new: bool = False
    is_deleted: bool = False
    is_binary: bool = False
    hunks: list[Hunk] = field(default_factory=list)

    def added_lines(self) -> set[int]:
        return {ln.new_no for h in self.hunks for ln in h.lines if ln.kind == "+"}

    def commentable_lines(self) -> set[int]:
        """New-side line numbers GitHub accepts for an inline comment (side=RIGHT)."""
        return {ln.new_no for h in self.hunks for ln in h.lines if ln.new_no is not None}


def parse_diff(text: str) -> list[FileDiff]:
    """Parse ``git diff`` output into per-file hunks with exact line mapping."""
    files: list[FileDiff] = []
    cur: FileDiff | None = None
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("diff --git "):
            # "diff --git a/x b/y"; the ---/+++ lines refine this, but binary files have none.
            m = re.match(r"^diff --git a/(.*) b/(.*)$", line)
            old, new = (m.group(1), m.group(2)) if m else ("", "")
            cur = FileDiff(path=new, old_path=old)
            files.append(cur)
            i += 1
            continue
        if cur is None:
            i += 1
            continue
        if line.startswith("new file mode"):
            cur.is_new = True
        elif line.startswith("deleted file mode"):
            cur.is_deleted = True
        elif line.startswith("rename from "):
            cur.old_path = line[len("rename from "):]
        elif line.startswith("rename to "):
            cur.path = line[len("rename to "):]
        elif line.startswith(("Binary files", "GIT binary patch")):
            cur.is_binary = True
        elif line.startswith("--- ") and not cur.hunks:
            if line[4:] == "/dev/null":
                cur.is_new = True
        elif line.startswith("+++ ") and not cur.hunks:
            if line[4:] == "/dev/null":
                cur.is_deleted = True
                cur.path = cur.old_path
        elif (m := HUNK_RE.match(line)) and not cur.is_binary:
            i = _parse_hunk(cur, m, lines, i)
            continue
        i += 1
    return files


def _parse_hunk(fd: FileDiff, m: re.Match, lines: list[str], i: int) -> int:
    old_start, old_len = int(m.group(1)), int(m.group(2) or 1)
    new_start, new_len = int(m.group(3)), int(m.group(4) or 1)
    # position counts every line after the first @@, including later @@ lines
    position = 0 if not fd.hunks else fd.hunks[-1].lines[-1].position + 1
    hunk = Hunk(old_start, old_len, new_start, new_len, m.group(5).strip(), position)
    fd.hunks.append(hunk)
    old_no, new_no = old_start, new_start
    old_left, new_left = old_len, new_len
    i += 1
    # Use the hunk's own line counts to find its end. A removed line "-- comment" is printed
    # as "--- comment" and would be mistaken for a file header if we trusted prefixes.
    while i < len(lines) and (old_left > 0 or new_left > 0):
        line = lines[i]
        position += 1
        kind, text = (line[:1] or " "), line[1:]
        if kind == "\\":  # "\ No newline at end of file" occupies a position but no line number
            hunk.lines.append(DiffLine("\\", text, None, None, position))
        elif kind == "+":
            hunk.lines.append(DiffLine("+", text, None, new_no, position))
            new_no += 1
            new_left -= 1
        elif kind == "-":
            hunk.lines.append(DiffLine("-", text, old_no, None, position))
            old_no += 1
            old_left -= 1
        else:
            hunk.lines.append(DiffLine(" ", text, old_no, new_no, position))
            old_no += 1
            new_no += 1
            old_left -= 1
            new_left -= 1
        i += 1
    # trailing "\ No newline" marker(s) belong to this hunk too
    while i < len(lines) and lines[i].startswith("\\"):
        position += 1
        hunk.lines.append(DiffLine("\\", lines[i][1:], None, None, position))
        i += 1
    return i


def snap_line(fd: FileDiff, line: int, max_distance: int = 2) -> int | None:
    """Return a commentable line near `line`, or None.

    Models are off by one or two lines surprisingly often, and GitHub rejects a comment on a
    line outside the diff ("Line could not be resolved"), which fails the whole review.
    Prefer the nearest *added* line, then the nearest context line.
    """
    commentable = fd.commentable_lines()
    if line in commentable:
        return line
    added = fd.added_lines()
    for pool in (added, commentable):
        near = [n for n in pool if abs(n - line) <= max_distance]
        if near:
            return min(near, key=lambda n: (abs(n - line), n))
    return None


def render_hunk(hunk: Hunk) -> str:
    """Show a hunk to the model with NEW-file line numbers on every visible line.

    We hand the model the numbers instead of asking it to count: ``  42 + code``. Removed lines
    have no new-file number, so they are shown with a blank gutter.
    """
    out = [f"@@ new lines {hunk.new_start}-{hunk.new_start + max(hunk.new_len, 1) - 1} @@"
           f" {hunk.header}".rstrip()]
    for ln in hunk.lines:
        if ln.kind == "\\":
            continue
        gutter = f"{ln.new_no:>4}" if ln.new_no is not None else "    "
        out.append(f"{gutter} {ln.kind} {ln.text}")
    return "\n".join(out)


def render_file(fd: FileDiff, hunks: list[Hunk] | None = None) -> str:
    status = "new file" if fd.is_new else "modified"
    head = f"### {fd.path} ({status})"
    body = "\n".join(render_hunk(h) for h in (hunks if hunks is not None else fd.hunks))
    return f"{head}\n{body}"


@dataclass
class ReviewUnit:
    """One model call's worth of diff."""

    paths: list[str]
    text: str


def make_units(files: list[FileDiff], granularity: str) -> list[ReviewUnit]:
    """Split a parsed diff into the pieces we send to the model.

    hunk: one call per hunk (with the diff's context lines), file: one per file, diff: one call.
    Binary and deleted files are skipped: there is nothing to comment on.
    """
    reviewable = [f for f in files if f.hunks and not f.is_binary and not f.is_deleted]
    if granularity == "hunk":
        return [ReviewUnit([f.path], render_file(f, [h])) for f in reviewable for h in f.hunks]
    if granularity == "file":
        return [ReviewUnit([f.path], render_file(f)) for f in reviewable]
    if granularity == "diff":
        if not reviewable:
            return []
        return [ReviewUnit([f.path for f in reviewable],
                           "\n\n".join(render_file(f) for f in reviewable))]
    raise ValueError(f"unknown granularity {granularity!r}; use hunk, file or diff")
