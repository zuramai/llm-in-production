"""Turn review comments into one GitHub "create a review" request.

Docs: https://docs.github.com/en/rest/pulls/reviews#create-a-review-for-a-pull-request
We send ``line`` + ``side: RIGHT`` (the new-file line number) rather than the legacy diff
``position``, and always use ``event: COMMENT``: the bot informs, it never blocks a merge.
"""

from __future__ import annotations

from .diff import FileDiff
from .schemas import ReviewComment

API = "https://api.github.com"
FOOTER = ("_Automated review by an LLM. It can be wrong: treat it as an assistant, not a gate._")


def format_body(c: ReviewComment) -> str:
    return f"**{c.severity} · {c.category}** (confidence {c.confidence:.2f})\n\n{c.body}"


def build_review_payload(comments: list[ReviewComment], files: list[FileDiff], *,
                         commit_id: str | None = None, max_inline: int = 15) -> dict:
    """Build the JSON body for POST /repos/{owner}/{repo}/pulls/{n}/reviews.

    A comment whose line is not in the diff would make GitHub answer 422 ("Line could not be
    resolved") and reject the WHOLE review, so those are moved into the review summary instead.
    """
    commentable = {f.path: f.commentable_lines() for f in files}
    inline, general = [], []
    for c in comments:
        ok = c.line in commentable.get(c.path, set()) and len(inline) < max_inline
        if ok:
            inline.append({"path": c.path, "line": c.line, "side": "RIGHT",
                           "body": format_body(c)})
        else:
            general.append(c)
    if comments:
        body = f"Found {len(comments)} possible issue(s).\n\n"
    else:
        body = "No issues found in the changed lines.\n\n"
    for c in general:
        body += f"- `{c.path}:{c.line}` {c.severity}/{c.category}: {c.body}\n"
    body += f"\n{FOOTER}"
    payload: dict = {"body": body, "event": "COMMENT", "comments": inline}
    if commit_id:
        payload["commit_id"] = commit_id
    return payload


def post_review(repo: str, pr_number: int, payload: dict, token: str) -> dict:
    """POST the review. On 422 retry once with everything folded into the summary."""
    import httpx  # imported lazily: dry runs and tests need no HTTP stack

    url = f"{API}/repos/{repo}/pulls/{pr_number}/reviews"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
               "X-GitHub-Api-Version": "2022-11-28"}
    r = httpx.post(url, json=payload, headers=headers, timeout=30)
    if r.status_code == 422 and payload.get("comments"):
        folded = "\n".join(f"- `{c['path']}:{c['line']}` {c['body']}" for c in payload["comments"])
        fallback = {**payload, "comments": [], "body": f"{payload['body']}\n\n{folded}"}
        r = httpx.post(url, json=fallback, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()
