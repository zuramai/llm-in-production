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


@dataclass
class ReviewRun:
    """Everything one review of one diff produced, before the confidence filter."""

    comments: list[ReviewComment]  # snapped onto the diff and deduped, NOT yet filtered
    raw_count: int  # comments the model returned
    n_units: int
    latency_ms: int
    usage: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


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


def review_unit(client: openai.OpenAI, model: str, unit: ReviewUnit, *, guidelines: bool,
                options: dict) -> tuple[list[ReviewComment], dict]:
    system = SYSTEM.format(guidelines=GUIDELINES if guidelines else "")
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": f"Review this diff:\n\n{unit.text}"}],
        response_format=RESPONSE_FORMAT,
        max_tokens=MAX_TOKENS,
        **options,
    )
    choice = resp.choices[0]
    if choice.finish_reason == "length":
        raise ValueError("output truncated (finish_reason=length)")
    return parse_result(choice.message.content or "").comments, _usage(resp)


def review_diff(diff_text: str, *, client: openai.OpenAI, model: str, granularity: str = "hunk",
                guidelines: bool = True, options: dict | None = None, workers: int = 4
                ) -> ReviewRun:
    """Review a whole diff: split, call the model per unit, map lines back, dedupe."""
    files = parse_diff(diff_text)
    units = make_units(files, granularity)
    t0 = time.perf_counter()
    raw: list[ReviewComment] = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}
    errors: list[str] = []

    def work(unit: ReviewUnit):
        return review_unit(client, model, unit, guidelines=guidelines, options=options or {})

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(work, u) for u in units]
        for unit, fut in zip(units, futures, strict=True):
            try:
                comments, u = fut.result()
            except (openai.OpenAIError, ValueError, ValidationError) as e:
                errors.append(f"{unit.paths}: {type(e).__name__}: {str(e)[:160]}")
                continue
            if granularity != "diff" and len(unit.paths) == 1:
                # the unit knows its file; don't trust the model to repeat the path
                comments = [c.model_copy(update={"path": unit.paths[0]}) for c in comments]
            raw.extend(comments)
            for k in usage:
                usage[k] += u.get(k, 0)
    mapped = dedupe(map_to_diff(raw, files))
    return ReviewRun(mapped, len(raw), len(units), int((time.perf_counter() - t0) * 1000),
                     usage, errors)
