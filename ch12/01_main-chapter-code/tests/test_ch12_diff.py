import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ch12_review.diff import make_units, parse_diff, render_hunk, snap_line

# A hand-written diff that contains every awkward case in one place:
#   * a removed line whose text starts with "-- " (printed as "--- sql comment")
#   * two hunks in one file (GitHub positions keep counting across the second @@)
#   * "\ No newline at end of file"
#   * a new file, a deleted file, a binary file, a rename, a hunk header without counts
DIFF = """\
diff --git a/app.py b/app.py
index 111..222 100644
--- a/app.py
+++ b/app.py
@@ -1,6 +1,7 @@
 import os
 
-x = 1
+x = 2
+y = 3
 def f():
--- sql comment
+    return 1
 
@@ -20,3 +21,4 @@ def g():
 a
 b
+c
 d
\\ No newline at end of file
diff --git a/new.py b/new.py
new file mode 100644
index 0000000..333
--- /dev/null
+++ b/new.py
@@ -0,0 +1,2 @@
+a
+b
diff --git a/gone.py b/gone.py
deleted file mode 100644
index 444..0000000
--- a/gone.py
+++ /dev/null
@@ -1,1 +0,0 @@
-z
diff --git a/img.png b/img.png
index 555..666 100644
Binary files a/img.png and b/img.png differ
diff --git a/old.py b/renamed.py
similarity index 90%
rename from old.py
rename to renamed.py
--- a/old.py
+++ b/renamed.py
@@ -1 +1 @@
-q = 1
+q = 2
"""


@pytest.fixture(scope="module")
def files():
    return {f.path: f for f in parse_diff(DIFF)}


def test_all_files_found(files):
    assert set(files) == {"app.py", "new.py", "gone.py", "img.png", "renamed.py"}
    assert files["new.py"].is_new
    assert files["gone.py"].is_deleted
    assert files["img.png"].is_binary and not files["img.png"].hunks
    assert files["renamed.py"].old_path == "old.py"


def test_removed_line_that_looks_like_a_file_header_stays_in_its_hunk(files):
    """'--- sql comment' is a deleted line '-- sql comment', not a '---' file header."""
    hunk = files["app.py"].hunks[0]
    removed = [ln for ln in hunk.lines if ln.kind == "-"]
    assert [ln.text for ln in removed] == ["x = 1", "-- sql comment"]
    assert len(files["app.py"].hunks) == 2  # and the second hunk was still found


def test_new_and_old_line_numbers(files):
    lines = {(ln.kind, ln.text): ln for h in files["app.py"].hunks for ln in h.lines}
    assert lines[("+", "x = 2")].new_no == 3 and lines[("+", "x = 2")].old_no is None
    assert lines[("-", "x = 1")].old_no == 3 and lines[("-", "x = 1")].new_no is None
    assert lines[(" ", "def f():")].old_no == 4 and lines[(" ", "def f():")].new_no == 5
    # the removed lines shift new numbers: "return 1" is old line 5 replaced at new line 6
    assert lines[("+", "    return 1")].new_no == 6
    # second hunk starts at new line 21: a=21 b=22 c(added)=23 d=24
    assert lines[("+", "c")].new_no == 23
    assert lines[(" ", "d")].new_no == 24 and lines[(" ", "d")].old_no == 22


def test_github_positions_continue_across_hunks(files):
    pos = {(ln.kind, ln.text): ln.position for h in files["app.py"].hunks for ln in h.lines}
    assert pos[(" ", "import os")] == 1  # the line just below the first @@
    assert pos[("+", "    return 1")] == 8
    assert files["app.py"].hunks[1].position == 10  # the second @@ line occupies position 10
    assert pos[(" ", "a")] == 11
    assert pos[("+", "c")] == 13
    assert pos[(" ", "d")] == 14


def test_no_newline_marker_takes_a_position_but_no_line_number(files):
    marker = files["app.py"].hunks[1].lines[-1]
    assert marker.kind == "\\" and marker.new_no is None and marker.position == 15
    assert 25 not in files["app.py"].commentable_lines()


def test_hunk_header_without_counts_defaults_to_one(files):
    hunk = files["renamed.py"].hunks[0]
    assert (hunk.old_len, hunk.new_len) == (1, 1)
    assert files["renamed.py"].added_lines() == {1}


def test_new_file_lines_are_numbered_from_one(files):
    assert files["new.py"].added_lines() == {1, 2}


def test_commentable_lines_are_added_plus_context_only(files):
    app = files["app.py"]
    assert app.commentable_lines() == {1, 2, 3, 4, 5, 6, 7, 21, 22, 23, 24}
    assert app.added_lines() == {3, 4, 6, 23}
    # line 8 exists in the file but is outside every hunk: GitHub would answer 422
    assert 8 not in app.commentable_lines()


def test_snap_line_prefers_added_lines_and_gives_up_far_away(files):
    app = files["app.py"]
    assert snap_line(app, 4) == 4  # already fine
    assert snap_line(app, 8) == 6  # 2 away from added line 6; outside the diff itself
    assert snap_line(app, 9) == 7  # no added line within 2: fall back to the context line
    assert snap_line(app, 10) is None  # 3 away from anything in the diff: drop, don't guess
    assert snap_line(app, 0) == 1


def test_render_hunk_shows_new_line_numbers_and_blank_gutter_for_removals(files):
    text = render_hunk(files["app.py"].hunks[0])
    rows = text.split("\n")
    assert rows[0].startswith("@@ new lines 1-7 @@")
    assert "   3 + x = 2" in rows
    assert "     - x = 1" in rows  # removed line: no number
    assert "   6 + " + "    return 1" in rows


def test_make_units_per_granularity_skips_deleted_and_binary(files):
    parsed = list(files.values())
    hunk_units = make_units(parsed, "hunk")
    assert [u.paths for u in hunk_units] == [["app.py"], ["app.py"], ["new.py"], ["renamed.py"]]
    assert len(make_units(parsed, "file")) == 3
    whole = make_units(parsed, "diff")
    assert len(whole) == 1 and whole[0].paths == ["app.py", "new.py", "renamed.py"]
    assert "gone.py" not in whole[0].text and "img.png" not in whole[0].text
    with pytest.raises(ValueError, match="granularity"):
        make_units(parsed, "line")


def test_empty_diff():
    assert parse_diff("") == []
    assert make_units([], "diff") == []
