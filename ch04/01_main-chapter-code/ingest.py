"""Chunk, embed and store the ShopFlow docs.

    uv run python ingest.py            # idempotent: re-running replaces the rows
    uv run python ingest.py --reset    # drop ch04's own table first

Needs the shared ShopFlow Postgres (port 5433) and Ollama serving `nomic-embed-text`.
"""

from __future__ import annotations

import argparse
import json

from ch04_rag import store
from ch04_rag.chunking import CHUNKERS


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reset", action="store_true", help="drop ch04_doc_chunks first")
    args = ap.parse_args()
    conn = store.open_db()
    store.setup(conn, reset=args.reset)
    docs = store.load_docs()
    for chunker in CHUNKERS:
        print(json.dumps(store.ingest(conn, docs, chunker)))
    print("rows per chunker:", store.counts(conn))


if __name__ == "__main__":
    main()
