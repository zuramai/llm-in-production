"""Chunk, embed and store the ShopFlow docs in `ch05_doc_chunks` (Chapter 4's recipe).

    uv run python setup_db.py            # idempotent: re-running replaces the rows
    uv run python setup_db.py --reset    # drop ch05's own table first

Needs the shared ShopFlow Postgres (port 5433) and Ollama serving `nomic-embed-text`.
"""

from __future__ import annotations

import argparse
import json

from ch05_agentic.prev import prev


def main(reset: bool = False) -> dict:
    conn = prev.open_db()
    prev.setup(conn, reset=reset)
    stats = prev.ingest(conn, prev.load_docs())
    stats["rows"] = conn.execute(f"SELECT count(*) FROM {prev.TABLE}").fetchone()[0]
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reset", action="store_true", help="drop ch05_doc_chunks first")
    print(json.dumps(main(ap.parse_args().reset)))
