import itertools
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ch04_rag.chunking import (
    WhitespaceTokenizer,
    fixed_chunks,
    heading_chunks,
    split_sections,
)

TOK = WhitespaceTokenizer()


def words(n: int, prefix: str = "w") -> str:
    return " ".join(f"{prefix}{i}" for i in range(n))


def test_fixed_chunks_overlap_exactly_the_requested_tokens():
    text = words(100)
    chunks = fixed_chunks(text, "d.md", size=30, overlap=10, tok=TOK)
    # windows start at 0, 20, 40, 60, 80; the last one stops at the end of the document
    assert [c.token_count for c in chunks] == [30, 30, 30, 30, 20]
    for a, b in itertools.pairwise(chunks):
        assert a.content.split()[-10:] == b.content.split()[:10]


def test_fixed_chunks_lose_no_text_and_have_no_redundant_tail():
    text = words(95)
    chunks = fixed_chunks(text, "d.md", size=30, overlap=10, tok=TOK)
    seen = []
    for c in chunks:
        seen.extend(c.content.split())
    assert set(seen) == set(text.split())  # every word appears
    # the last window ends exactly at the end of the document, with no extra chunk inside the
    # previous chunk's overlap
    assert chunks[-1].content.split()[-1] == "w94"
    assert chunks[-1].token_count > 10
    # indexes are consecutive: they become the #N in [doc:path#N]
    assert [c.chunk_idx for c in chunks] == list(range(len(chunks)))


def test_fixed_chunks_short_and_empty_documents():
    assert fixed_chunks("", "d.md", tok=TOK) == []
    only = fixed_chunks("one two three", "d.md", size=30, overlap=10, tok=TOK)
    assert len(only) == 1 and only[0].content == "one two three"


def test_overlap_not_smaller_than_size_is_rejected():
    with pytest.raises(ValueError):
        fixed_chunks("a b c", "d.md", size=10, overlap=10, tok=TOK)


DOC = """# Refund Policy

*Owner: Support*

## Refund Timing

Returns are refunded within 5-7 business days.

### Cancelled orders

Refunded in full.

## Refund Method

Original payment method only.

```bash
# this is a comment, not a heading
deployctl rollback checkout
```
"""


def test_heading_paths_keep_the_whole_ancestry():
    chunks = heading_chunks(DOC, "policies/refund.md", tok=TOK)
    paths = [c.heading_path for c in chunks]
    assert paths == [
        "Refund Policy",
        "Refund Policy > Refund Timing",
        "Refund Policy > Refund Timing > Cancelled orders",
        "Refund Policy > Refund Method",
    ]


def test_heading_chunks_ignore_hashes_inside_code_fences():
    chunks = heading_chunks(DOC, "d.md", tok=TOK)
    last = chunks[-1]
    assert "# this is a comment, not a heading" in last.content
    assert last.heading_path == "Refund Policy > Refund Method"


def test_heading_chunks_lose_no_body_text():
    chunks = heading_chunks(DOC, "d.md", tok=TOK)
    headings = ["# Refund Policy", "## Refund Timing", "### Cancelled orders", "## Refund Method"]
    body = DOC
    for h in headings:
        body = body.replace(h + "\n", "", 1)
    body_words = re.findall(r"\S+", body)
    got = [w for c in chunks for w in c.content.split()]
    assert got == body_words  # same words, same order


def test_oversized_section_is_split_on_paragraphs_under_the_budget():
    paras = "\n\n".join(words(40, f"p{i}x") for i in range(5))  # 5 paragraphs of 40 tokens
    chunks = heading_chunks(f"# T\n\n## Long\n\n{paras}\n", "d.md", max_tokens=100, tok=TOK)
    assert len(chunks) == 3  # 40+40 | 40+40 | 40
    assert all(c.token_count <= 100 for c in chunks)
    assert {c.heading_path for c in chunks} == {"T > Long"}
    assert [c.chunk_idx for c in chunks] == [0, 1, 2]
    # paragraph boundaries are respected: no chunk starts mid-paragraph
    assert all(c.content.split()[0].endswith("x0") for c in chunks)


def test_giant_single_paragraph_falls_back_to_token_windows():
    text = "# T\n\n" + words(250)
    chunks = heading_chunks(text, "d.md", max_tokens=100, tok=TOK)
    assert all(c.token_count <= 100 for c in chunks)
    assert {w for c in chunks for w in c.content.split()} == set(words(250).split())


def test_sections_without_body_emit_nothing_but_still_extend_the_path():
    sections = split_sections("# A\n\n## B\n\n### C\n\ntext\n")
    assert sections == [("A > B > C", "text")]


def test_embed_text_and_citation_format():
    c = heading_chunks("# Doc\n\n## Part\n\nhello", "x/y.md", tok=TOK)[0]
    assert c.embed_text == "Doc > Part\nhello"
    assert c.citation == "[doc:x/y.md#0]"
    f = fixed_chunks("hello world", "x/y.md", tok=TOK)[0]
    assert f.embed_text == "hello world"
