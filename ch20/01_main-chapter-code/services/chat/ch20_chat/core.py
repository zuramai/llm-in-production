"""Chatbot core, copied from Chapter 1 (ch01/01_main-chapter-code/chat.py).

Changes for production: the system prompt is versioned (prompts/support-<version>.md) instead
of hard-coded, and the OpenAI client is passed in rather than built from the book's `.env`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import openai
import tiktoken

PROMPTS_DIR = Path(__file__).with_name("prompts")
FAQ_PATH = PROMPTS_DIR / "faq.md"

Message = dict[str, str]
_ENCODING = tiktoken.get_encoding("o200k_base")
# Chat formats wrap each message in a few special tokens; 4 is a safe per-message overhead.
_PER_MESSAGE_OVERHEAD = 4


def load_prompts(prompts_dir: Path = PROMPTS_DIR) -> dict[str, str]:
    """Render every prompts/support-<version>.md with the FAQ; return {version: prompt}."""
    faq = (prompts_dir / "faq.md").read_text()
    return {
        path.stem.removeprefix("support-"): path.read_text().replace("{faq}", faq)
        for path in sorted(prompts_dir.glob("support-*.md"))
    }


def count_tokens(messages: list[Message]) -> int:
    """Estimate prompt tokens locally. The provider's `usage` is the authoritative count."""
    return sum(len(_ENCODING.encode(m["content"])) + _PER_MESSAGE_OVERHEAD for m in messages)


def trim_history(
    messages: list[Message],
    max_tokens: int,
    count: Callable[[list[Message]], int] = count_tokens,
) -> list[Message]:
    """Fit a conversation into `max_tokens` by dropping the oldest complete turns.

    Keeps the system message (first) and the latest user message (last). Turns in between are
    dropped oldest-first, a user message together with the assistant reply that follows it, so a
    reply is never kept without its question. Raises ValueError if the system message plus the
    latest user message alone exceed the budget.
    """
    if len(messages) < 2 or messages[0]["role"] != "system" or messages[-1]["role"] != "user":
        raise ValueError("expected [system, ..., user] messages")
    system, latest = messages[0], messages[-1]
    if count([system, latest]) > max_tokens:
        raise ValueError(
            f"system prompt + latest message need {count([system, latest])} tokens, "
            f"budget is {max_tokens}"
        )
    turns: list[list[Message]] = []
    for m in messages[1:-1]:
        if m["role"] == "user" or not turns:
            turns.append([m])
        else:
            turns[-1].append(m)
    while turns and count([system, *(m for t in turns for m in t), latest]) > max_tokens:
        turns.pop(0)
    return [system, *(m for t in turns for m in t), latest]


def stream_reply(
    client: openai.OpenAI,
    history: list[Message],
    *,
    model: str,
    usage: dict | None = None,
    max_tokens: int = 800,
    reasoning_effort: str | None = None,
) -> Iterator[str]:
    """Stream the assistant's reply to `history` as text deltas; fill `usage` at the end."""
    options = {"reasoning_effort": reasoning_effort} if reasoning_effort else {}
    stream = client.chat.completions.create(
        model=model,
        messages=history,
        max_tokens=max_tokens,
        stream=True,
        stream_options={"include_usage": True},
        **options,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
        if chunk.usage is not None and usage is not None:
            usage.update(chunk.usage.model_dump(exclude_none=True))
