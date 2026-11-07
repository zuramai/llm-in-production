import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ch04_rag.answer import REFUSAL, extract_citations, format_context
from ch04_rag.evalset import (
    Question,
    audit,
    load_questions,
    recall_at_k,
    reciprocal_rank,
)
from ch04_rag.retrieve import Hit, or_terms, rrf


def test_rrf_matches_hand_computation():
    # a: rank 1 in list 1, rank 3 in list 2; b: rank 2 and rank 1; c: only rank 3 in list 1
    fused = dict(rrf([["a", "b", "c"], ["b", "x", "a"]], k=60))
    assert fused["a"] == pytest.approx(1 / 61 + 1 / 63)
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["c"] == pytest.approx(1 / 63)
    assert fused["x"] == pytest.approx(1 / 62)


def test_rrf_orders_by_score_and_rewards_agreement():
    ranked = [item for item, _ in rrf([["a", "b", "c"], ["b", "x", "a"]])]
    assert ranked == ["b", "a", "x", "c"]  # b is high in both; x (62) beats c (63)


def test_rrf_ignores_score_scale_and_breaks_ties_by_first_seen():
    ranked = [i for i, _ in rrf([["a", "b"], ["b", "a"]])]
    assert ranked == ["a", "b"]  # exactly tied: list order decides, deterministically


def test_rrf_duplicates_inside_one_list_count_once():
    fused = dict(rrf([["a", "a", "b"]], k=60))
    assert fused["a"] == pytest.approx(1 / 61)
    assert fused["b"] == pytest.approx(1 / 63)  # rank counts positions, a repeat does not shift it


def test_rrf_empty_inputs_and_k_changes_the_gap():
    assert rrf([]) == [] and rrf([[], []]) == []
    gap_small_k = dict(rrf([["a", "b"]], k=1))
    gap_big_k = dict(rrf([["a", "b"]], k=60))
    assert gap_small_k["a"] / gap_small_k["b"] > gap_big_k["a"] / gap_big_k["b"]


def test_or_terms_keep_identifiers_intact():
    q = "Which flag? (payments.primary_provider) and db_pool_size, SF-ELE-0073 costs $5.99."
    terms = or_terms(q)
    assert "payments.primary_provider" in terms
    assert "db_pool_size" in terms
    assert "SF-ELE-0073" in terms
    assert "5.99" in terms


def hit(doc: str, idx: int = 0) -> Hit:
    return Hit(0, doc, idx, "", "text")


Q = Question("t", "multi", "q", "r", (("a.md", "b.md"), ("c.md",)), ("x", "y"))


def test_recall_counts_slots_not_chunks():
    hits = [hit("a.md"), hit("a.md", 1), hit("b.md"), hit("z.md"), hit("z.md", 1), hit("c.md")]
    assert recall_at_k(Q, hits, k=5) == 0.5  # slot 1 covered (twice), slot 2 (c.md) is rank 6
    assert recall_at_k(Q, hits, k=6) == 1.0
    assert recall_at_k(Q, [], k=5) == 0.0


def test_reciprocal_rank_uses_first_gold_chunk_within_k():
    assert reciprocal_rank(Q, [hit("z.md"), hit("z.md", 1), hit("c.md")]) == pytest.approx(1 / 3)
    assert reciprocal_rank(Q, [hit("z.md")] * 3) == 0.0
    assert reciprocal_rank(Q, [hit("z.md")] * 10 + [hit("a.md")], k=10) == 0.0  # rank 11 > k


def test_gold_labels_are_consistent_with_the_documents():
    errors, _ = audit()
    assert errors == []


def test_eval_set_shape():
    qs = load_questions()
    assert len(qs) == 40
    assert len({q.id for q in qs}) == 40
    by_cat = {c: sum(q.category == c for q in qs) for c in {q.category for q in qs}}
    assert by_cat["unanswerable"] >= 4
    assert by_cat["multi"] >= 4 and by_cat["keyword"] >= 8 and by_cat["paraphrase"] >= 8
    assert all(q.answerable == (q.category != "unanswerable") for q in qs)
    assert all(len(slot) >= 1 for q in qs for slot in q.slots)


def test_citation_extraction_and_context_format():
    text = f"Roll back [doc:runbooks/a.md#2]. See also [doc:x/b.md#10][doc:x/b.md#11]. {REFUSAL}"
    assert extract_citations(text) == [("runbooks/a.md", 2), ("x/b.md", 10), ("x/b.md", 11)]
    ctx = format_context([Hit(1, "p/q.md", 3, "Doc > Sec", "body text")])
    assert ctx.startswith("[doc:p/q.md#3] (Doc > Sec)\nbody text")
