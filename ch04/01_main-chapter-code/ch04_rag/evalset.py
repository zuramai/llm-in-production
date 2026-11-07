"""The 40-question eval set and the retrieval metrics (recall@k, MRR). Pure functions: no I/O
except reading the YAML and the docs, so everything here is unit-tested offline.

    uv run python -m ch04_rag.evalset audit      # run from 01_main-chapter-code/
"""

from __future__ import annotations

import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

from .retrieve import Hit
from .store import DOCS_ROOT, load_docs

QUESTIONS = Path(__file__).resolve().parents[1] / "eval" / "questions.yaml"


@dataclass(frozen=True)
class Question:
    id: str
    category: str
    question: str
    reference: str
    slots: tuple[tuple[str, ...], ...]  # each slot: any of these doc paths satisfies it
    verify: tuple[str, ...]

    @property
    def answerable(self) -> bool:
        return bool(self.slots)


def _normalised_docs(root: Path = DOCS_ROOT) -> dict[str, str]:
    return {p: re.sub(r"\s+", " ", t) for p, t in load_docs(root).items()}


def load_questions(path: Path = QUESTIONS, root: Path = DOCS_ROOT) -> list[Question]:
    docs = None
    out = []
    for q in yaml.safe_load(path.read_text()):
        slots, verify = [], []
        for s in q.get("evidence") or []:
            if s["paths"] == "auto":  # every document that states the fact
                docs = docs or _normalised_docs(root)
                pat = re.compile(s["verify"])
                slots.append(tuple(p for p, t in docs.items() if pat.search(t)))
            else:
                slots.append(tuple(s["paths"]))
            verify.append(s["verify"])
        out.append(Question(
            q["id"], q["category"], q["question"], q["reference"], tuple(slots), tuple(verify)
        ))
    return out


def satisfied_slots(q: Question, doc_paths: Sequence[str]) -> int:
    got = set(doc_paths)
    return sum(1 for slot in q.slots if got.intersection(slot))


def recall_at_k(q: Question, hits: Sequence[Hit], k: int = 5) -> float:
    """Fraction of evidence slots covered by the top-k chunks."""
    return satisfied_slots(q, [h.doc_path for h in hits[:k]]) / len(q.slots)


def reciprocal_rank(q: Question, hits: Sequence[Hit], k: int = 10) -> float:
    """1 / rank of the first chunk (within the top k) that lies in any evidence slot."""
    wanted = {p for slot in q.slots for p in slot}
    for rank, h in enumerate(hits[:k], start=1):
        if h.doc_path in wanted:
            return 1.0 / rank
    return 0.0


def audit(root: Path = DOCS_ROOT) -> tuple[list[str], list[str]]:
    """(errors, notes). Errors: a slot is empty, names a missing file, or lists a path that does
    not match its `verify` regex. Notes: documents that match `verify` but are not in the slot."""
    docs = _normalised_docs(root)
    problems, notes = [], []
    for q in load_questions(root=root):
        if q.answerable != (q.category != "unanswerable"):
            problems.append(f"{q.id}: category {q.category} vs evidence mismatch")
        for slot, rx in zip(q.slots, q.verify, strict=True):
            if not slot:
                problems.append(f"{q.id}: empty slot for /{rx}/")
        for slot, rx in zip(q.slots, q.verify, strict=True):
            pat = re.compile(rx)
            for p in slot:
                if p not in docs:
                    problems.append(f"{q.id}: {p} does not exist")
                elif not pat.search(docs[p]):
                    problems.append(f"{q.id}: {p} does not match /{rx}/")
            extra = sorted(p for p, t in docs.items() if pat.search(t) and p not in slot)
            if extra:
                notes.append(f"{q.id}: also matches /{rx}/ but not listed: {extra}")
    return problems, notes


if __name__ == "__main__":
    if sys.argv[1:] == ["audit"]:
        errors, notes = audit()
        print("\n".join(notes))
        print("\n".join(errors) or "gold labels consistent")
