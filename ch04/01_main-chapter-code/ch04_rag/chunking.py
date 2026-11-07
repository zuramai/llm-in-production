"""Two ways to cut a Markdown document into retrievable chunks.

* `fixed_chunks`: slide a 512-token window over the whole document, 64 tokens of overlap.
* `heading_chunks`: cut at Markdown headings and keep the heading path ("Doc > Section > Sub")
  as context. Sections longer than the budget are split on paragraph boundaries.

Both work on *character spans* of the original text, so chunks are exact slices (nothing is
re-encoded) and "no lost text" is a property we can test. Token counting is injected through the
`Tokenizer` protocol: production uses tiktoken, tests use a whitespace tokenizer and need no
network (tiktoken downloads its vocabulary on first use).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

FIXED_SIZE = 512
FIXED_OVERLAP = 64
HEADING_MAX = 512

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")


class Tokenizer(Protocol):
    def token_starts(self, text: str) -> list[int]:
        """Character offset where each token starts. Token i spans starts[i]:starts[i+1]."""


class TiktokenTokenizer:
    """tiktoken `o200k_base`, the same tokenizer family the chat models in this book use."""

    def __init__(self, encoding: str = "o200k_base") -> None:
        import tiktoken

        self._enc = tiktoken.get_encoding(encoding)

    def token_starts(self, text: str) -> list[int]:
        _, offsets = self._enc.decode_with_offsets(self._enc.encode(text))
        return offsets


class WhitespaceTokenizer:
    """One token per whitespace-separated word. Deterministic and offline: used in tests."""

    def token_starts(self, text: str) -> list[int]:
        return [m.start() for m in re.finditer(r"\S+", text)]


_DEFAULT: Tokenizer | None = None


def default_tokenizer() -> Tokenizer:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = TiktokenTokenizer()
    return _DEFAULT


@dataclass(frozen=True)
class Chunk:
    doc_path: str  # path relative to the docs root, e.g. "policies/refund-policy.md"
    chunk_idx: int  # ordinal inside (doc, chunker); this is the `#N` in `[doc:path#N]`
    chunker: str  # "fixed" | "heading"
    heading_path: str  # "Refund Policy > Refund Timing"; empty for fixed chunks
    content: str
    token_count: int

    @property
    def embed_text(self) -> str:
        """What we embed and rerank on: heading path (if any) plus the body."""
        return f"{self.heading_path}\n{self.content}" if self.heading_path else self.content

    @property
    def citation(self) -> str:
        return f"[doc:{self.doc_path}#{self.chunk_idx}]"


def count_tokens(text: str, tok: Tokenizer | None = None) -> int:
    return len((tok or default_tokenizer()).token_starts(text))


def _window_spans(
    text: str, size: int, overlap: int, tok: Tokenizer
) -> list[tuple[int, int, int]]:
    """(start_char, end_char, n_tokens) windows of `size` tokens, stepping by size - overlap."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    starts = tok.token_starts(text)
    n = len(starts)
    if n == 0:
        return []
    step = size - overlap
    spans = []
    i = 0
    while True:
        j = min(i + size, n)
        end_char = starts[j] if j < n else len(text)
        spans.append((starts[i], end_char, j - i))
        if j >= n:
            break
        i += step
    return spans


def fixed_chunks(
    text: str,
    doc_path: str,
    size: int = FIXED_SIZE,
    overlap: int = FIXED_OVERLAP,
    tok: Tokenizer | None = None,
) -> list[Chunk]:
    tok = tok or default_tokenizer()
    out = []
    for idx, (a, b, n) in enumerate(_window_spans(text, size, overlap, tok)):
        out.append(Chunk(doc_path, idx, "fixed", "", text[a:b].strip(), n))
    return out


def split_sections(text: str) -> list[tuple[str, str]]:
    """Cut Markdown into (heading_path, body) sections. Headings inside code fences are text."""
    sections: list[tuple[str, list[str]]] = []
    stack: list[tuple[int, str]] = []
    path = ""
    body: list[str] = []
    in_fence = False
    for line in text.splitlines():
        if _FENCE.match(line):
            in_fence = not in_fence
        m = None if in_fence else _HEADING.match(line)
        if m:
            sections.append((path, body))
            level, title = len(m.group(1)), m.group(2)
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            path = " > ".join(t for _, t in stack)
            body = []
        else:
            body.append(line)
    sections.append((path, body))
    return [(p, "\n".join(b).strip()) for p, b in sections if "\n".join(b).strip()]


def _pack_paragraphs(body: str, max_tokens: int, tok: Tokenizer) -> list[str]:
    """Greedy-pack blank-line-separated paragraphs into pieces of <= max_tokens."""
    pieces: list[str] = []
    current: list[str] = []
    used = 0
    for para in re.split(r"\n\s*\n", body):
        para = para.strip()
        if not para:
            continue
        n = len(tok.token_starts(para))
        if n > max_tokens:  # one giant paragraph (or code block): fall back to token windows
            if current:
                pieces.append("\n\n".join(current))
                current, used = [], 0
            pieces.extend(
                para[a:b].strip()
                for a, b, _ in _window_spans(para, max_tokens, FIXED_OVERLAP, tok)
            )
            continue
        if used + n > max_tokens and current:
            pieces.append("\n\n".join(current))
            current, used = [], 0
        current.append(para)
        used += n
    if current:
        pieces.append("\n\n".join(current))
    return pieces


def heading_chunks(
    text: str, doc_path: str, max_tokens: int = HEADING_MAX, tok: Tokenizer | None = None
) -> list[Chunk]:
    tok = tok or default_tokenizer()
    out: list[Chunk] = []
    for path, body in split_sections(text):
        for piece in _pack_paragraphs(body, max_tokens, tok):
            out.append(
                Chunk(doc_path, len(out), "heading", path, piece, len(tok.token_starts(piece)))
            )
    return out


CHUNKERS = {"fixed": fixed_chunks, "heading": heading_chunks}
