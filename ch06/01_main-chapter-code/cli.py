"""Ask the ShopFlow data bot a question from the terminal.

    cd ch06/01_main-chapter-code
    uv run python cli.py "How many transactions are there today?" --show-sql
"""

from __future__ import annotations

import argparse

from sqlbot import answer


def main() -> None:
    p = argparse.ArgumentParser(description="Ask ShopFlow's data a question")
    p.add_argument("question")
    p.add_argument("--model", help="override LLM_MODEL")
    p.add_argument("--show-sql", action="store_true")
    args = p.parse_args()

    result = answer(args.question, model=args.model)
    print(result.answer)
    if args.show_sql:
        print()
        print(result.sql or "(no SQL was run)")
        print(f"\nmodel={result.model} attempts={result.attempts} latency={result.latency_ms} ms "
              f"tokens={result.usage.get('total_tokens', '?')}")


if __name__ == "__main__":
    main()
