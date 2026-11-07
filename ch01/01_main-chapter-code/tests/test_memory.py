import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chat import trim_history


def words(messages):
    """Deterministic counter for tests: one token per word."""
    return sum(len(m["content"].split()) for m in messages)


def msg(role, n, tag=""):
    return {"role": role, "content": " ".join([tag or role] * n)}


SYSTEM = msg("system", 10)


def conversation():
    # system(10) + 3 complete turns of (user 5, assistant 5) + latest user(5) = 45 tokens
    return [
        SYSTEM,
        msg("user", 5, "u1"), msg("assistant", 5, "a1"),
        msg("user", 5, "u2"), msg("assistant", 5, "a2"),
        msg("user", 5, "u3"), msg("assistant", 5, "a3"),
        msg("user", 5, "latest"),
    ]


def test_within_budget_is_unchanged():
    conv = conversation()
    assert trim_history(conv, 45, count=words) == conv


def test_keeps_system_and_latest_and_respects_budget():
    out = trim_history(conversation(), 30, count=words)
    assert out[0] == SYSTEM
    assert out[-1]["content"].startswith("latest")
    assert words(out) <= 30


def test_drops_oldest_turns_first():
    # 34 leaves 19 tokens for the middle: only the newest turn (u3, a3) fits.
    out = trim_history(conversation(), 34, count=words)
    tags = [m["content"].split()[0] for m in out[1:-1]]
    assert tags == ["u3", "a3"]


def test_never_splits_a_user_assistant_pair():
    # A budget of 40 leaves room for 25 middle tokens: u2,a2,u3,a3 (20) fit, but so would
    # a1 alone (5). The pair rule forbids keeping a1 without u1.
    out = trim_history(conversation(), 40, count=words)
    tags = [m["content"].split()[0] for m in out[1:-1]]
    assert tags == ["u2", "a2", "u3", "a3"]
    for i, m in enumerate(out[1:-1]):
        if m["role"] == "assistant":
            assert out[1:-1][i - 1]["role"] == "user"


def test_raises_when_system_plus_latest_exceed_budget():
    with pytest.raises(ValueError, match="budget"):
        trim_history(conversation(), 14, count=words)


def test_rejects_malformed_conversation():
    with pytest.raises(ValueError):
        trim_history([msg("user", 1)], 100, count=words)
