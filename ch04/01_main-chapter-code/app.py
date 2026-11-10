"""FastAPI service: POST /ask returns a cited answer over the ShopFlow docs.

    cd ch04/01_main-chapter-code && uv run uvicorn app:app --port 8004
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Literal

from ch04_rag import store
from ch04_rag.answer import generate
from ch04_rag.retrieve import Retriever
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    # Load the cross-encoder (to the GPU when there is one) before the first request, not in it.
    Retriever.cross_encoder()
    yield


app = FastAPI(title="ShopFlow docs assistant (chapter 4)", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    mode: Literal["vector", "fts", "hybrid", "hybrid_rerank"] = "hybrid_rerank"
    chunker: Literal["fixed", "heading"] = "heading"
    k: int = Field(default=10, ge=1, le=20)  # 10 heading chunks ≈ 1.7k prompt tokens


class Source(BaseModel):
    citation: str
    heading_path: str
    score: float
    snippet: str


class AskResponse(BaseModel):
    answer: str
    refused: bool
    sources: list[Source]
    invalid_citations: list[str]
    timing_ms: dict[str, int]
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


# A plain `def` endpoint runs in FastAPI's threadpool. psycopg connections are not shared between
# threads, so each request opens its own (about 5 ms locally; put a pool such as psycopg_pool in
# front of it once traffic matters, see chapter 20).
@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    try:
        with store.open_db() as conn:
            retriever = Retriever(conn)
            hits = retriever.retrieve(req.question, req.mode, req.chunker, req.k)
            timing = {k: round(v) for k, v in retriever.last_timing.items()}
    except Exception as e:  # DB down, Ollama down: say which dependency, not a stack trace
        raise HTTPException(status_code=503, detail=f"retrieval failed: {e}") from e
    try:
        ans = generate(req.question, hits)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM call failed: {e}") from e
    timing["llm_ms"] = ans.latency_ms
    return AskResponse(
        answer=ans.text,
        refused=ans.refused,
        sources=[Source(citation=h.citation, heading_path=h.heading_path, score=round(h.score, 4),
                        snippet=h.content[:200]) for h in hits],
        invalid_citations=[f"{p}#{i}" for p, i in ans.invalid_citations],
        timing_ms=timing,
        prompt_tokens=ans.prompt_tokens,
        completion_tokens=ans.completion_tokens,
        cost_usd=ans.cost_usd,
    )
