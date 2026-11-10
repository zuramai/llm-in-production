"""The reviewer: diff in, a short list of ReviewComment out."""

from __future__ import annotations

import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import openai
from pydantic import ValidationError

from .diff import ReviewUnit, make_units, parse_diff
from .postprocess import dedupe, map_to_diff
from .schemas import RESPONSE_FORMAT, ReviewComment, ReviewResult

MAX_TOKENS = 3000  # always cap output; reasoning tokens count against it

GUIDELINES = """Repo conventions (from the team's review guide):
- Money is stored as integer cents.
- "Today" and other calendar periods mean the calendar day in the business time zone
  (America/New_York, `BUSINESS_TZ` in service/config.py), never UTC.
- Secrets come from environment variables, never from source code.
- Database access uses parameterised queries (psycopg `%s` placeholders).
- Stock changes must be safe under concurrent requests."""

SYSTEM = """You are a careful senior engineer reviewing a pull request to a Python service.
You see part of a unified diff. Each line starts with its line number in the NEW version of the
file, then `+` (added), a space (unchanged context) or `-` (removed, no number).

Report only real defects that the added lines introduce or expose: security holes, bugs,
concurrency problems, performance problems. Do NOT report style, naming, missing docstrings or
tests, or refactors that keep behaviour. Do not speculate about code you cannot see. If the
change looks fine, return an empty `comments` list. Most hunks are fine.

For each defect return:
- path: exactly as in the `###` header
- line: the number printed in front of the most responsible line (prefer an added line)
- severity: low | medium | high | critical
- category: security | correctness | performance | concurrency | maintainability | style | tests
- body: 1-3 sentences: what is wrong, the concrete failure it causes, and the fix
- confidence: 0 to 1. Use 0.9+ only when you can name the exact input that breaks.

{guidelines}"""

# v0: what everyone writes first. The raw `git diff` text, no line numbers, no rules.
NAIVE_SYSTEM = """You are a code reviewer. Review this pull request diff and point out problems.
For each problem give the file path, the line number in the new version of the file, a severity,
a category, a short explanation and your confidence from 0 to 1."""


@dataclass
class ReviewRun:
    """Everything one review of one diff produced, before the confidence filter."""

    comments: list[ReviewComment]  # snapped onto the diff and deduped, NOT yet filtered
    raw_count: int  # comments the model returned
    n_units: int
    latency_ms: int
    usage: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    raw: list[ReviewComment] = field(default_factory=list)  # as returned, before any cleanup


def git_diff(repo: str, base: str, head: str, context: int = 5) -> str:
    """The PR diff: changes on `head` since it forked from `base` (what GitHub shows)."""
    return subprocess.run(
        ["git", "-C", repo, "diff", "--no-color", f"-U{context}", f"{base}...{head}"],
        check=True, capture_output=True, text=True,
    ).stdout


def parse_result(content: str) -> ReviewResult:
    """Validate the model's JSON; tolerate prose around it (json_object-style replies)."""
    try:
        return ReviewResult.model_validate_json(content)
    except ValidationError:
        start, end = content.find("{"), content.rfind("}")
        if start == -1 or end <= start:
            raise
        return ReviewResult.model_validate(json.loads(content[start:end + 1]))


def _usage(resp) -> dict:
    u = resp.usage
    if u is None:
        return {}
    cost = getattr(u, "cost", None) or (u.model_extra or {}).get("cost") or 0.0
    return {"prompt_tokens": u.prompt_tokens or 0, "completion_tokens": u.completion_tokens or 0,
            "cost_usd": float(cost)}


class UnitFailed(Exception):
    """A call that was answered (and billed) but produced nothing usable."""

    def __init__(self, message: str, usage: dict):
        super().__init__(message)
        self.usage = usage


def review_unit(client: openai.OpenAI, model: str, unit: ReviewUnit, *, system: str,
                options: dict) -> tuple[list[ReviewComment], dict]:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": f"Review this diff:\n\n{unit.text}"}],
        response_format=RESPONSE_FORMAT,
        max_tokens=MAX_TOKENS,
        **options,
    )
    usage = _usage(resp)
    choice = resp.choices[0]
    if choice.finish_reason == "length":
        # reasoning models can spend the whole budget thinking; the tokens are still billed
        raise UnitFailed("output truncated (finish_reason=length)", usage)
    try:
        return parse_result(choice.message.content or "").comments, usage
    except (ValidationError, ValueError) as e:
        raise UnitFailed(f"unparseable output: {type(e).__name__}: {e}", usage) from e


def review_diff(diff_text: str, *, client: openai.OpenAI, model: str, granularity: str = "hunk",
                guidelines: bool = True, options: dict | None = None, workers: int = 4
                ) -> ReviewRun:
    """Review a whole diff: split, call the model per unit, map lines back, dedupe.

    granularity: hunk | file | diff (numbered lines, our prompt), or raw: the naive v0, which
    sends the unmodified `git diff` text with NAIVE_SYSTEM in one call.
    """
    files = parse_diff(diff_text)
    if granularity == "raw":
        reviewable = [f.path for f in files if f.hunks and not f.is_binary and not f.is_deleted]
        units = [ReviewUnit(reviewable, diff_text)] if reviewable else []
        system = NAIVE_SYSTEM
    else:
        units = make_units(files, granularity)
        system = SYSTEM.format(guidelines=GUIDELINES if guidelines else "")
    t0 = time.perf_counter()
    raw: list[ReviewComment] = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}
    errors: list[str] = []

    def work(unit: ReviewUnit):
        return review_unit(client, model, unit, system=system, options=options or {})

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(work, u) for u in units]
        for unit, fut in zip(units, futures, strict=True):
            try:
                comments, u = fut.result()
            except (openai.OpenAIError, UnitFailed) as e:
                errors.append(f"{unit.paths}: {type(e).__name__}: {str(e)[:160]}")
                for k in usage:  # failed calls cost money too
                    usage[k] += getattr(e, "usage", {}).get(k, 0)
                continue
            if granularity in ("hunk", "file"):
                # the unit knows its file; don't trust the model to repeat the path
                comments = [c.model_copy(update={"path": unit.paths[0]}) for c in comments]
            raw.extend(comments)
            for k in usage:
                usage[k] += u.get(k, 0)
    mapped = dedupe(map_to_diff(raw, files))
    return ReviewRun(mapped, len(raw), len(units), int((time.perf_counter() - t0) * 1000),
                     usage, errors, raw)
