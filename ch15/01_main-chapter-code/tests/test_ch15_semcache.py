import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ch15_gateway.semcache import (
    DIM,
    TABLE,
    SemanticCache,
    is_cacheable,
    normalize,
    setup,
    sweep,
)


def user(text):
    return {"role": "user", "content": text}


@pytest.mark.parametrize(
    ("messages", "ok"),
    [
        ([user("How long do refunds take?")], True),
        ([{"role": "system", "content": "You are ShopFlow."}, user("What is VIP?")], True),
        # follow-ups depend on the conversation: never shared
        ([user("Can I return shoes?"), {"role": "assistant", "content": "Yes."},
          user("And for VIPs?")], False),
        ([{"role": "system", "content": "s"}], False),
        # personal questions: another customer must never get this answer
        ([user("Where is order #48213?")], False),
        ([user("Where is order number 48213?")], False),
        ([user("I'm ana@example.com, did my refund go through?")], False),
        ([user("I was charged $45.99 twice")], False),
        ([user("How do I track my order?")], True),       # "order" alone is not personal
    ],
)
def test_is_cacheable(messages, ok):
    assert is_cacheable(messages) is ok


def test_normalize_collapses_case_and_whitespace():
    assert normalize("  How long\tdo REFUNDS\n take? ") == "how long do refunds take?"


def test_sweep_threshold_is_inclusive_and_counts_false_hits():
    probes = [
        ({"a"}, "a", 0.95),     # right entry, very similar
        ({"a"}, "b", 0.92),     # wrong entry won the nearest-neighbour search: false hit if served
        (set(), "c", 0.90),     # nothing in the cache is right for it: false hit if served
        ({"d"}, "d", 0.80),     # right entry but not similar enough
    ]
    p90, p93, p99 = sweep(probes, [0.90, 0.93, 0.99])
    assert (p90.served, p90.correct, p90.false_hits, p90.should_hit) == (3, 1, 2, 3)
    assert p90.false_hit_rate == pytest.approx(2 / 3)
    assert p90.hit_rate == pytest.approx(1 / 3)
    assert (p93.served, p93.correct, p93.false_hits) == (1, 1, 0)
    assert (p99.served, p99.false_hit_rate, p99.hit_rate) == (0, 0.0, 0.0)


def test_sweep_accepts_any_listed_entry():
    [p] = sweep([({"express_cost", "ship_cost"}, "ship_cost", 0.9)], [0.9])
    assert (p.correct, p.false_hits) == (1, 0)


def test_threshold_must_be_a_similarity():
    for bad in (0, -0.1, 1.01):
        with pytest.raises(ValueError):
            SemanticCache(conn=None, embed=None, namespace="x", threshold=bad)


def _unit(angle_deg: float) -> list[float]:
    """A 768-d unit vector at `angle_deg` from e0 in the e0/e1 plane: cos(angle) = similarity."""
    v = [0.0] * DIM
    v[0], v[1] = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
    return v


ANGLES = {"how long do refunds take?": 0, "when will i get my refund?": 20,
          "how do i cancel my vip membership?": 60}


def fake_embed(texts):
    return [_unit(ANGLES[t]) for t in texts]


@pytest.fixture
def cache():
    from llmprod import connect

    conn = connect()
    setup(conn)
    c = SemanticCache(conn, fake_embed, namespace="pytest-ch15", threshold=0.9)
    c.clear()
    yield c
    c.clear()
    conn.close()


@pytest.mark.docker
def test_roundtrip_threshold_namespace_and_ttl(cache):
    cache.store("How long do refunds take?", "5-7 business days.")
    hit = cache.lookup("When will I get my refund?")             # cos(20°) = 0.940 >= 0.9
    assert hit is not None and hit.response == "5-7 business days."
    assert hit.similarity == pytest.approx(math.cos(math.radians(20)), abs=1e-4)
    assert cache.lookup("How do I cancel my VIP membership?") is None   # cos(60°) = 0.5
    assert cache.nearest("How do I cancel my VIP membership?").similarity == pytest.approx(0.5, 1e-3)

    other = SemanticCache(cache.conn, fake_embed, namespace="pytest-ch15-other", threshold=0.9)
    assert other.lookup("How long do refunds take?") is None     # another model/prompt version

    expired = SemanticCache(cache.conn, fake_embed, namespace="pytest-ch15", threshold=0.9,
                            ttl_seconds=0)
    assert expired.lookup("How long do refunds take?") is None   # stale entries never serve


@pytest.mark.docker
def test_setup_is_idempotent(cache):
    setup(cache.conn)
    n = cache.conn.execute(f"SELECT count(*) FROM pg_indexes WHERE tablename = '{TABLE}'").fetchone()
    assert n[0] == 3          # primary key, HNSW, (namespace, created_at)
