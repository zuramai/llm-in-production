"""Chapter 18: the fine-tuning dataset for ShopFlow ticket triage.

Everything here is plain Python (no torch), so the main ``.venv`` can build, check and test the
data, and the CUDA venv only has to read JSONL files.

* ``template_of``: which of the 30 seed templates produced a templated ticket.
* ``split``: 1,200 / 150 / 150 templated tickets, plus the 60 hard tickets in test only.
* ``to_example``: one record -> the (prompt messages, answer) pair we train on.
* ``prompt_completion``: the same pair rendered as text with the model's chat template.
"""

from __future__ import annotations

import importlib.util
import json
import random
import re
import sys
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
TEMPLATES_PATH = HERE.parents[1] / "datasets" / "shopflow" / "ticket_templates.json"

SPLIT_SIZES = {"train": 1200, "val": 150, "test": 150}
SPLIT_SEED = 18

# The fine-tuned model gets a one-line instruction instead of Chapter 2's ~400-token rulebook:
# the rules are in the weights now. The format line is Chapter 2's FORMAT_HINT, verbatim, so the
# same text works with ``build_request("prompt_only", rules=FT_RULES)``.
FT_RULES = "You triage customer-support tickets for ShopFlow, an online shop."
FORMAT_HINT = (
    "\n\nReply with a single JSON object with exactly these keys: category, priority, "
    "sentiment, order_id (integer or null), summary. Reply with JSON only."
)
ANSWER_KEYS = ("category", "priority", "sentiment", "order_id", "summary")
CATEGORIES = ("shipping", "refund", "product_defect", "account", "payment", "other")
PRIORITIES = ("low", "medium", "high", "urgent")
SENTIMENTS = ("negative", "neutral", "positive")


# --------------------------------------------------------------------------- Chapter 2 code


def previous() -> Any:
    """Chapter 2 code, loaded under a unique name (every chapter ships a previous_chapters.py)."""
    name = "ch18_previous_chapters"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, HERE / "previous_chapters.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def as_ticket(r: dict[str, Any]) -> Any:
    """A JSONL record -> Chapter 2's ``Ticket`` dataclass."""
    return previous().Ticket(r["id"], r["subject"], r["body"], r["channel"], r["category"],
                             r["priority"], r["order_id"], r["hard"])


# --------------------------------------------------------------------------- templates


def load_templates(path: Path = TEMPLATES_PATH) -> list[dict[str, Any]]:
    return json.loads(path.read_text())


def template_of(subject: str, body: str, templates: list[dict[str, Any]]) -> int:
    """Index of the seed template a templated ticket was generated from.

    The literal pieces between ``{placeholders}`` must appear in order, and the text must start
    with the first piece and end with the last. Same logic as the Chapter 2 notebook.
    """
    text = subject + "\n" + body
    for i, tp in enumerate(templates):
        pieces = re.split(r"\{\w+\}", tp["subject"] + "\n" + tp["body"])
        if not (text.startswith(pieces[0]) and text.endswith(pieces[-1])):
            continue
        pos = 0
        for piece in pieces:
            pos = text.find(piece, pos)
            if pos < 0:
                break
            pos += len(piece)
        else:
            return i
    raise LookupError(f"ticket {subject!r} matches no template")


# --------------------------------------------------------------------------- split


def split(records: list[dict[str, Any]], *, sizes: dict[str, int] | None = None,
          seed: int = SPLIT_SEED) -> dict[str, list[dict[str, Any]]]:
    """Seeded split of the templated tickets; every hard ticket goes to test, and only there.

    The hard tickets are the out-of-distribution exam: training never sees one, so the test score
    on them says how the model copes with traffic that doesn't look like its training data.
    """
    sizes = sizes or SPLIT_SIZES
    easy = sorted((r for r in records if not r["hard"]), key=lambda r: r["id"])
    hard = sorted((r for r in records if r["hard"]), key=lambda r: r["id"])
    need = sizes["train"] + sizes["val"] + sizes["test"]
    if len(easy) != need:
        raise ValueError(f"expected {need} templated tickets, got {len(easy)}")
    random.Random(seed).shuffle(easy)
    a, b = sizes["train"], sizes["train"] + sizes["val"]
    by_id = lambda rs: sorted(rs, key=lambda r: r["id"])
    return {"train": by_id(easy[:a]), "val": by_id(easy[a:b]), "test": by_id(easy[b:] + hard)}


def ticket_record(ticket: Any, templates: list[dict[str, Any]]) -> dict[str, Any]:
    """A ``Ticket`` from Chapter 2 -> a JSON-able row. Gold labels keep their DB names."""
    return {
        "id": ticket.id,
        "subject": ticket.subject,
        "body": ticket.body,
        "channel": ticket.channel,
        "category": ticket.category,
        "priority": ticket.priority,
        "order_id": ticket.order_id,
        "hard": ticket.hard,
        "template_id": None if ticket.hard else template_of(ticket.subject, ticket.body,
                                                            templates),
        "source": "shopflow",
    }


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def text_key(r: dict[str, Any]) -> tuple[str, str]:
    """Exact-duplicate key, same normalisation as Chapter 19's ``convert.leakage``."""
    return r["subject"].strip().lower(), r["body"].strip().lower()


def drop_overlap(train: list[dict[str, Any]], *held_out: list[dict[str, Any]]
                 ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Remove training rows whose exact text also appears in a held-out split.

    Templates without a variable field (no order number, product or name) produce the very same
    ticket again and again, so a random split puts copies on both sides. Returns (kept, dropped).
    """
    banned = {text_key(r) for rows in held_out for r in rows}
    kept = [r for r in train if text_key(r) not in banned]
    dropped = [r for r in train if text_key(r) in banned]
    return kept, dropped


# --------------------------------------------------------------------------- training format


def format_ticket(subject: str, body: str, channel: str) -> str:
    """Chapter 2's delimited ticket block (``format_ticket(..., delimited=True)``), verbatim."""
    safe = body.replace("<", "&lt;").replace(">", "&gt;")
    return f"<ticket>\nchannel: {channel}\nsubject: {subject}\nbody: {safe}\n</ticket>"


def answer_json(record: dict[str, Any]) -> str:
    """The target string: compact JSON, keys in schema order, gold routing + teacher text."""
    missing = [k for k in ANSWER_KEYS if k not in record]
    if missing:
        raise KeyError(f"record {record.get('id')} has no {missing}; run the teacher step first")
    if record["category"] not in CATEGORIES or record["priority"] not in PRIORITIES:
        raise ValueError(f"record {record.get('id')}: label outside the schema")
    if record["sentiment"] not in SENTIMENTS:
        raise ValueError(f"record {record.get('id')}: sentiment {record['sentiment']!r}")
    if len(record["summary"]) > 200:
        raise ValueError(f"record {record.get('id')}: summary longer than 200 characters")
    return json.dumps({k: record[k] for k in ANSWER_KEYS}, ensure_ascii=False)


def to_example(record: dict[str, Any]) -> dict[str, Any]:
    """Chat messages for one training example: short system prompt, ticket, answer."""
    return {
        "messages": [
            {"role": "system", "content": FT_RULES + FORMAT_HINT},
            {"role": "user", "content": format_ticket(record["subject"], record["body"],
                                                      record["channel"])},
            {"role": "assistant", "content": answer_json(record)},
        ]
    }


def prompt_completion(record: dict[str, Any], render: Callable[..., str],
                      end_of_turn: str = "<|im_end|>") -> dict[str, str]:
    """Render one example as ``{"prompt", "completion"}`` text for TRL's SFTTrainer.

    ``render`` is ``tokenizer.apply_chat_template``. The prompt is rendered exactly the way the
    server will render it at inference time (generation prompt, thinking switched off), so the
    model is trained on the same tokens it will see in production. Only the completion is in the
    loss. The completion ends with the end-of-turn token: forget it and the model never learns to
    stop.
    """
    messages = to_example(record)["messages"]
    prompt = render(messages[:-1], tokenize=False, add_generation_prompt=True,
                    enable_thinking=False)
    full = render(messages, tokenize=False, enable_thinking=False)
    completion = messages[-1]["content"] + end_of_turn
    if not full.startswith(prompt + completion):
        raise ValueError(
            "chat template mismatch: training text does not start with the inference prompt "
            f"plus the answer.\nprompt tail: {prompt[-80:]!r}\nfull: {full[len(prompt) - 80:]!r}"
        )
    return {"prompt": prompt, "completion": completion}


# --------------------------------------------------------------------------- leakage check


def _shingles(text: str, n: int = 3) -> set[tuple[str, ...]]:
    words = re.findall(r"\w+", text.lower())
    return {tuple(words[i:i + n]) for i in range(max(1, len(words) - n + 1))}


def jaccard(a: str, b: str, n: int = 3) -> float:
    """Word-trigram Jaccard similarity: 1.0 for identical texts, ~0 for unrelated ones."""
    sa, sb = _shingles(a, n), _shingles(b, n)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def nearest(text: str, pool: list[str]) -> tuple[float, int]:
    """Highest Jaccard similarity of ``text`` to any text in ``pool``, and its index."""
    best, idx = 0.0, -1
    for i, other in enumerate(pool):
        s = jaccard(text, other)
        if s > best:
            best, idx = s, i
    return best, idx
