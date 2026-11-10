"""Book invoices from the command line: extract, validate, write to Postgres or to review.

    cd ch08/01_main-chapter-code
    uv run python ch08_cli.py data/pdf/inv-001.pdf data/scans/scan-01.png
    uv run python ch08_cli.py --local data/scans/*.png      # local models first, Luna if stuck
    uv run python ch08_cli.py --review                      # show the open review queue
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ch08_extract import ollama_client
from ch08_pipeline import ModelSpec, Outcome, process_file, record, setup_tables

from llmprod import chat_client, connect, get_settings

LOCAL_TEXT = "qwen3:8b"
LOCAL_VISION = "gemma3:4b"


def pipeline_config(local: bool) -> dict[str, ModelSpec]:
    s = get_settings()
    hosted = chat_client(timeout=180)
    if not local:
        return {"text": ModelSpec(hosted, s.llm_model, max_side=1200),
                "strong": ModelSpec(hosted, s.llm_model_strong)}
    ollama = ollama_client()
    return {"text": ModelSpec(ollama, LOCAL_TEXT, {"reasoning_effort": "none"}, printed=True),
            "vision": ModelSpec(ollama, LOCAL_VISION, printed=True),
            "strong": ModelSpec(hosted, s.llm_model, max_side=1200)}


def show(o: Outcome, booked_as: str) -> str:
    inv = o.invoice
    what = (f"{inv.supplier} {inv.invoice_no} {inv.date} {inv.currency} "
            f"{inv.total_cents / 100:,.2f}" if inv else "-")
    stages = "+".join(a.stage for a in o.attempts) or "-"
    why = "; ".join(r.rule for r in o.reasons)
    return (f"{o.file:14} {o.method:7} {stages:22} {booked_as:8} ${o.cost_usd:.5f} "
            f"{o.latency_s:5.1f}s  {what}  {why}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("files", nargs="*", type=Path)
    ap.add_argument("--local", action="store_true", help="local models first, hosted fallback")
    ap.add_argument("--review", action="store_true", help="list the open review queue")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--reset", action="store_true", help="empty the two ch08 tables first")
    a = ap.parse_args()
    with connect() as conn:
        setup_tables(conn, reset=a.reset)
        if a.files:
            cfg = pipeline_config(a.local)
            with ThreadPoolExecutor(a.workers) as pool:
                outcomes = list(pool.map(lambda p: process_file(p, **cfg), a.files))
            for o in outcomes:  # record sequentially, in the order given
                print(show(o, record(conn, o)))
        if a.review:
            rows = conn.execute(
                "SELECT id, source_file, rules, left(reason, 80) FROM ch08_invoice_review "
                "WHERE status = 'open' ORDER BY id").fetchall()
            print(f"{len(rows)} open review items")
            for r in rows:
                print(f"  #{r[0]:<4} {r[1]:14} {','.join(r[2]):24} {r[3]}")


if __name__ == "__main__":
    main()
