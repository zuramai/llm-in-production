import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "services" / "chat"))

from ch20_chat.canary import BUCKETS, bucket, choose_version

SESSIONS = [f"session-{i}" for i in range(20_000)]


def share(percent, canary="v2"):
    return sum(choose_version(s, "v1", canary, percent) == canary for s in SESSIONS) / len(
        SESSIONS)


@pytest.mark.parametrize("percent", [1, 10, 25, 50, 90])
def test_split_matches_configured_percentage(percent):
    assert share(percent) == pytest.approx(percent / 100, abs=0.01)


def test_zero_percent_and_missing_canary_mean_everyone_on_stable():
    assert all(choose_version(s, "v1", "v2", 0) == "v1" for s in SESSIONS[:2000])
    assert all(choose_version(s, "v1", None, 50) == "v1" for s in SESSIONS[:2000])


def test_hundred_percent_means_everyone_on_canary():
    assert all(choose_version(s, "v1", "v2", 100) == "v2" for s in SESSIONS[:2000])


def test_sticky_per_session():
    first = [choose_version(s, "v1", "v2", 10) for s in SESSIONS[:2000]]
    again = [choose_version(s, "v1", "v2", 10) for s in SESSIONS[:2000]]
    assert first == again


def test_ramping_up_never_moves_a_canary_session_back_to_stable():
    for low, high in [(1, 10), (10, 50), (50, 99.99)]:
        at_low = {s for s in SESSIONS if choose_version(s, "v1", "v2", low) == "v2"}
        at_high = {s for s in SESSIONS if choose_version(s, "v1", "v2", high) == "v2"}
        assert at_low <= at_high


def test_fractional_percent_is_honored():
    assert share(0.5) == pytest.approx(0.005, abs=0.002)


def test_each_canary_picks_a_different_slice():
    v2 = {s for s in SESSIONS if choose_version(s, "v1", "v2", 10) == "v2"}
    v3 = {s for s in SESSIONS if choose_version(s, "v1", "v3", 10) == "v3"}
    overlap = len(v2 & v3) / len(v2)
    assert overlap == pytest.approx(0.10, abs=0.03)   # independent slices overlap ~10%


def test_bucket_range():
    assert all(0 <= bucket(s, "v2") < BUCKETS for s in SESSIONS[:2000])
