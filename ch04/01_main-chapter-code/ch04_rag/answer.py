"""Turn retrieved chunks into a cited answer, and check the citations are real."""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field

import openai

from llmprod import chat_client, get_settings

from .retrieve import Hit

REFUSAL = "I can't find that in the ShopFlow documents."
MAX_ANSWER_TOKENS = 600

_RULES = f"""You are ShopFlow's internal documentation assistant.
Answer ONLY from the numbered context chunks below. Rules:
1. After every factual sentence, cite the chunk(s) it came from, exactly like [doc:path#N].
   Copy the tag from the chunk header. Never invent a tag.
2. If the context does not contain the answer, reply exactly: {REFUSAL}
   Do not guess, and do not use outside knowledge, even if you are sure.
3. If chunks disagree, say so and cite both."""

PROMPTS = {
    # v1: what we shipped first. "Short" made the model drop conditions (see the chapter).
    "v1": _RULES + """
4. Keep it short: at most 4 sentences. State numbers, commands and names exactly as written.""",
    # v2: same grounding rules; completeness is spelled out instead of a sentence budget.
    "v2": _RULES + """
4. Answer every part of the question. For each part, include the conditions, exceptions and
   limits the context attaches to it (who, how much, how long, starting when, only if), and
   quote commands, flags and names exactly as written.
5. If the context answers only part of the question, answer that part and say what is missing.
6. No greetings and no advice the context doesn't give. Usually 2-6 sentences.""",
}

_CITE = re.compile(r"\[doc:([^\]#\s]+)#(\d+)\]")


def format_context(hits: list[Hit]) -> str:
    parts = []
    for h in hits:
        head = f" ({h.heading_path})" if h.heading_path else ""
        parts.append(f"{h.citation}{head}\n{h.content}")
    return "\n\n---\n\n".join(parts)


def extract_citations(text: str) -> list[tuple[str, int]]:
    return [(m.group(1), int(m.group(2))) for m in _CITE.finditer(text)]


@dataclass
class Answer:
    question: str
    text: str
    hits: list[Hit]
    citations: list[tuple[str, int]]
    invalid_citations: list[tuple[str, int]]
    refused: bool
    model: str
    latency_ms: int
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    context: str = field(repr=False, default="")


OLLAMA_CHAT_URL = os.environ.get("OLLAMA_CHAT_URL", "http://localhost:11434/v1")


def client_for(model: str) -> openai.OpenAI:
    """OpenRouter ids look like `vendor/model`; Ollama tags look like `qwen3:8b` and go to the
    local Ollama server's OpenAI-compatible endpoint (override with OLLAMA_CHAT_URL)."""
    if "/" in model:
        return chat_client(timeout=90)
    return openai.OpenAI(base_url=OLLAMA_CHAT_URL, api_key="ollama", timeout=180)


def generate(
    question: str, hits: list[Hit], model: str | None = None, context: str | None = None,
    prompt: str = "v2",
) -> Answer:
    """One chat completion over `hits` (or a ready-made `context`, e.g. the whole corpus).
    Ollama's default context is 4,096 tokens: five 512-token chunks fit, a whole corpus does not.
    """
    model = model or get_settings().llm_model
    ctx = context if context is not None else format_context(hits)
    t0 = time.perf_counter()
    resp = client_for(model).chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": PROMPTS[prompt]},
            {"role": "user", "content": f"Context:\n\n{ctx}\n\nQuestion: {question}"},
        ],
        max_tokens=MAX_ANSWER_TOKENS,
        reasoning_effort="none",
    )
    ms = int((time.perf_counter() - t0) * 1000)
    text = (resp.choices[0].message.content or "").strip()
    cites = extract_citations(text)
    valid = {(h.doc_path, h.chunk_idx) for h in hits} if context is None else None
    bad = [c for c in cites if valid is not None and c not in valid]
    usage = resp.usage
    return Answer(
        question=question, text=text, hits=hits, citations=cites, invalid_citations=bad,
        refused=text.startswith("I can't find that"), model=model, latency_ms=ms,
        prompt_tokens=usage.prompt_tokens, completion_tokens=usage.completion_tokens,
        cost_usd=float(getattr(usage, "cost", 0.0) or 0.0), context=ctx,
    )
