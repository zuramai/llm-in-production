"""Ask the instrumented ShopFlow data bot a question, then rate the answer 👍/👎.

    uv run python ch14/01_main-chapter-code/ch14_cli.py "How many transactions are there today?"
    echo "n
    counted payments, not orders" | uv run python ch14/01_main-chapter-code/ch14_cli.py "..."

Every answer prints the Langfuse trace URL. The rating becomes a `user_feedback` score on
that trace (1 = 👍, 0 = 👎), with an optional comment.
"""

from __future__ import annotations

import argparse
import getpass
import sys

import ch14_obs as obs
from ch14_sqlbot import answer


def main() -> None:
    p = argparse.ArgumentParser(description="Ask ShopFlow's data a question (traced)")
    p.add_argument("question")
    p.add_argument("--model", help="override LLM_MODEL")
    p.add_argument("--user", default=getpass.getuser(), help="user id recorded on the trace")
    p.add_argument("--session", help="session id, e.g. a Slack thread ts")
    p.add_argument("--prompt-label", default="production")
    p.add_argument("--show-sql", action="store_true")
    p.add_argument("--feedback", choices=["up", "down", "skip"],
                   help="rate without prompting (otherwise ask on stdin)")
    p.add_argument("--comment", help="feedback comment (with --feedback)")
    args = p.parse_args()

    lf = obs.client()
    result = answer(args.question, model=args.model, user_id=args.user,
                    session_id=args.session, prompt_label=args.prompt_label, langfuse=lf)
    print(result.answer)
    if args.show_sql:
        print()
        print(result.sql or "(no SQL was run)")
    print(f"\nmodel={result.model} attempts={result.attempts} latency={result.latency_ms} ms "
          f"tokens={result.usage.get('total_tokens', '?')} "
          f"cost=${result.usage.get('cost_usd', 0):.6f}")
    print(f"trace: {obs.trace_url(result.trace_id)}")

    if args.feedback:
        rating, comment = obs.parse_feedback(args.feedback), args.comment
    else:
        print("Was this helpful? [y = 👍 / n = 👎 / Enter = skip] ", end="", flush=True)
        rating = obs.parse_feedback(sys.stdin.readline())
        comment = None
        if rating is False:
            print("What was wrong? (optional) ", end="", flush=True)
            comment = sys.stdin.readline().strip() or None
    if rating is not None:
        obs.record_feedback(result.trace_id, rating, comment, langfuse=lf)
        print(f"\nrecorded {obs.FEEDBACK_SCORE}={int(rating)}" + (f" ({comment})" if comment
                                                               else ""))
    lf.flush()  # a CLI exits right away: flush, or the batch of spans is lost


if __name__ == "__main__":
    main()
