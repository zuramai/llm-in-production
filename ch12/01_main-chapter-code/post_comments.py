"""Post review comments to a GitHub pull request (or print what we would post).

    # executed locally: print the exact payload, send nothing
    uv run python post_comments.py --comments out/comments.json --repo sample_repo \
        --base main --head feature/order-fixes --dry-run

    # in CI (not executed by the author): needs GITHUB_TOKEN, a repo and a PR number
    python post_comments.py --comments out/comments.json --repo-slug OWNER/REPO --pr 42 \
        --commit-sha "$GITHUB_SHA" --base origin/main --head HEAD
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ch12_review.diff import parse_diff
from ch12_review.github import build_review_payload, post_review
from ch12_review.reviewer import git_diff
from ch12_review.schemas import ReviewComment


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--comments", required=True, help="JSON list written by review.py")
    ap.add_argument("--repo", default=".", help="local git checkout to diff (default: .)")
    ap.add_argument("--base", default="origin/main")
    ap.add_argument("--head", default="HEAD")
    ap.add_argument("--repo-slug", help="OWNER/REPO on GitHub (required unless --dry-run)")
    ap.add_argument("--pr", type=int, help="pull request number (required unless --dry-run)")
    ap.add_argument("--commit-sha", help="head commit the comments refer to")
    ap.add_argument("--dry-run", action="store_true", help="print the payload, send nothing")
    args = ap.parse_args()

    comments = [ReviewComment.model_validate(c) for c in json.loads(Path(args.comments).read_text())]
    files = parse_diff(git_diff(args.repo, args.base, args.head))
    payload = build_review_payload(comments, files, commit_id=args.commit_sha)

    if args.dry_run:
        slug = args.repo_slug or "OWNER/REPO"
        print(f"POST https://api.github.com/repos/{slug}/pulls/{args.pr or 'N'}/reviews")
        print(json.dumps(payload, indent=2))
        return 0

    token = os.environ.get("GITHUB_TOKEN")
    if not (token and args.repo_slug and args.pr):
        ap.error("--repo-slug, --pr and the GITHUB_TOKEN env var are required without --dry-run")
    review = post_review(args.repo_slug, args.pr, payload, token)
    print(f"posted review {review['id']}: {review['html_url']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
