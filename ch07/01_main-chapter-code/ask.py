"""Ask the semantic-layer bot a question from the terminal.

    cd ch07/01_main-chapter-code
    uv run python ask.py "What's our refund rate this month?" --show-sql
"""

from __future__ import annotations

import argparse
import json

from ch07_semantic.agent import answer


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("question")
    p.add_argument("--model", help="defaults to LLM_MODEL")
    p.add_argument("--show-sql", action="store_true")
    args = p.parse_args()
    res = answer(args.question, model=args.model)
    print(res.answer)
    if res.query:
        shown = {k: v for k, v in res.query.items() if v not in ([], None)}
        print("\nMetricQuery:", json.dumps(shown))
    if args.show_sql and res.sql:
        print("\n" + res.sql)
    print(f"\nmodel={res.model} attempts={res.attempts} latency={res.latency_ms} ms "
          f"tokens={res.usage.get('total_tokens')}")


if __name__ == "__main__":
    main()
