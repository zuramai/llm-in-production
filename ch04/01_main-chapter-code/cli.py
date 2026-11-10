"""Ask the ShopFlow docs a question from the terminal.

    uv run python cli.py "How long do refunds take?" [--mode hybrid_rerank] [--chunker heading]
"""

from __future__ import annotations

import argparse
import time

from ch04_rag import store
from ch04_rag.answer import generate
from ch04_rag.retrieve import MODES, Retriever


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("question")
    ap.add_argument("--mode", choices=MODES, default="hybrid_rerank")
    ap.add_argument("--chunker", choices=["fixed", "heading"], default="heading")
    ap.add_argument("-k", type=int, default=10)
    args = ap.parse_args()

    retriever = Retriever(store.open_db())
    if args.mode == "hybrid_rerank":  # load the model up front so its load time isn't "latency"
        t0 = time.perf_counter()
        ce = Retriever.cross_encoder()
        print(f"reranker loaded on {ce.model.device} in {time.perf_counter() - t0:.1f} s\n")
    hits = retriever.retrieve(args.question, args.mode, args.chunker, args.k)
    ans = generate(args.question, hits)
    print(ans.text, "\n")
    for h in hits:
        label = f" ({h.heading_path})" if h.heading_path else ""
        print(f"  {h.rank}. {h.citation}{label}  score={h.score:.3f}")
    timing = {k: round(v) for k, v in retriever.last_timing.items()}
    print(f"\nretrieval {timing} ms | llm {ans.latency_ms} ms | "
          f"{ans.prompt_tokens}+{ans.completion_tokens} tokens | ${ans.cost_usd:.6f}")
    if ans.invalid_citations:
        print("WARNING: citations not in the retrieved chunks:", ans.invalid_citations)


if __name__ == "__main__":
    main()
