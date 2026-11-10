"""review_diff with a fake model client: no network, checks what we send and what we keep."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ch12_review.reviewer import NAIVE_SYSTEM, review_diff

DIFF = """diff --git a/svc/a.py b/svc/a.py
index 1111111..2222222 100644
--- a/svc/a.py
+++ b/svc/a.py
@@ -1,3 +1,3 @@
 import os
-KEY = os.environ["KEY"]
+KEY = "hard-coded"
 X = 1
@@ -20,2 +20,3 @@ def f():
     a = 1
+    b = a / 0
     return a
"""


class FakeClient:
    """Mimics client.chat.completions.create and records every request."""

    def __init__(self, replies, cost_per_call=None):
        self.replies = list(replies)
        self.calls = []
        self.cost = cost_per_call
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        content, finish = self.replies.pop(0)
        msg = SimpleNamespace(content=content)
        usage = None if self.cost is None else SimpleNamespace(
            prompt_tokens=100, completion_tokens=50, cost=self.cost)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason=finish)],
                               usage=usage)


def reply(*comments, finish="stop"):
    return json.dumps({"comments": list(comments)}), finish


def comment(path, line, category="correctness", confidence=0.9):
    return {"path": path, "line": line, "severity": "high", "category": category,
            "body": f"problem at {line}", "confidence": confidence}


def test_raw_mode_sends_the_unnumbered_diff_once_with_the_naive_prompt():
    fake = FakeClient([reply(comment("svc/a.py", 2, "security"), comment("svc/a.py", 99))])
    run = review_diff(DIFF, client=fake, model="m", granularity="raw", workers=1)
    assert len(fake.calls) == 1 and run.n_units == 1
    system, user = (m["content"] for m in fake.calls[0]["messages"])
    assert system == NAIVE_SYSTEM
    assert DIFF in user  # the raw text, hunk headers and all, no line-number gutter
    assert fake.calls[0]["max_tokens"] > 0  # always capped
    # both comments are kept verbatim in `raw`; line 99 is far from the diff and is dropped
    assert [c.line for c in run.raw] == [2, 99]
    assert [c.line for c in run.comments] == [2]


def test_hunk_mode_calls_once_per_hunk_and_trusts_the_unit_path_not_the_model():
    fake = FakeClient([reply(comment("wrong/path.py", 2, "security")),
                       reply(comment("svc/a.py", 21))])
    run = review_diff(DIFF, client=fake, model="m", granularity="hunk", workers=1)
    assert len(fake.calls) == 2
    users = [call["messages"][1]["content"] for call in fake.calls]
    assert "   2 + KEY = \"hard-coded\"" in users[0]  # numbered gutter for the model
    assert {(c.path, c.line) for c in run.comments} == {("svc/a.py", 2), ("svc/a.py", 21)}


def test_truncated_output_becomes_an_error_and_the_other_units_still_count():
    fake = FakeClient([reply(finish="length"), reply(comment("svc/a.py", 21))])
    run = review_diff(DIFF, client=fake, model="m", granularity="hunk", workers=1)
    assert len(run.errors) == 1 and "finish_reason=length" in run.errors[0]
    assert [c.line for c in run.comments] == [21]


def test_failed_calls_are_still_billed_in_the_usage_totals():
    fake = FakeClient([reply(finish="length"), ("not json at all", "stop")], cost_per_call=0.25)
    run = review_diff(DIFF, client=fake, model="m", granularity="hunk", workers=1)
    assert len(run.errors) == 2 and run.comments == []
    assert run.usage == {"prompt_tokens": 200, "completion_tokens": 100, "cost_usd": 0.5}
