"""Ask the ShopFlow agent a question from the terminal and watch the graph run node by node.

    uv run python cli.py "Can customer 4860 still return order 21461?"
    uv run python cli.py "..." --model qwen3:8b
    uv run python cli.py "..." --variant tool_loop          # compare with the plain loop

Needs the ShopFlow Postgres (port 5433), Ollama (embeddings) and `setup_db.py` run once.
"""

from __future__ import annotations

import argparse
import time
import uuid

from ch05_agentic.graph import RECURSION_LIMIT, build_graph, initial_state
from ch05_agentic.llm import LLM
from ch05_agentic.tools import DocSearch, now_text
from ch05_agentic.variants import make_runner


def stream_graph(question: str, llm: LLM, docs: DocSearch) -> None:
    app = build_graph(llm, docs)
    config = {"recursion_limit": RECURSION_LIMIT,
              "configurable": {"thread_id": uuid.uuid4().hex}}
    t0 = time.perf_counter()
    final: dict = {}
    for update in app.stream(initial_state(question, now_text()), config,
                             stream_mode="updates"):
        for node, delta in update.items():
            ms = delta["trace"][0]["ms"]
            if node in ("route", "rewrite"):
                info = f"doc_query={delta['doc_query']!r} sql_question={delta['sql_question']!r}"
            elif node == "retrieve":
                info = " ".join(e["citation"] for e in delta["evidence"])
            elif node == "sql":
                e = delta["evidence"][0]
                info = f"{e['citation']} {len(e['rows'])} row(s)" + (
                    f" ERROR {e['error']}" if e["error"] else "")
            elif node == "grade":
                info = f"sufficient={delta['sufficient']} relevant={delta['relevant']}"
            else:
                info = ""
                final = delta
            print(f"[{node:8} {ms:>5} ms] {info}")
    print(f"\n{final.get('answer', '')}\n")
    cites = final.get("citations", {})
    if cites.get("invalid"):
        print("WARNING: citations not in the evidence:", cites["invalid"])
    print(f"total {round((time.perf_counter() - t0) * 1000)} ms, model={llm.model}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("question")
    ap.add_argument("--model", default=None, help="default: LLM_MODEL from .env")
    ap.add_argument("--variant", choices=["graph", "tool_loop", "ch04_rag"], default="graph")
    args = ap.parse_args()
    llm, docs = LLM(args.model), DocSearch()
    if args.variant == "graph":
        stream_graph(args.question, llm, docs)
        return
    res = make_runner(args.variant, llm, docs)(args.question)
    for s in res.steps:
        print(s)
    print(f"\n{res.answer}\n\ntotal {res.latency_ms} ms, {res.usage['calls']} LLM calls, "
          f"${res.usage['cost_usd']:.5f}")


if __name__ == "__main__":
    main()
