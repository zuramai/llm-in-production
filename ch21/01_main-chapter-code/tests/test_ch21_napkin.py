import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ch21_napkin import (
    Price,
    Workload,
    daily_cost,
    input_tokens_per_session,
    peak_to_avg_from_hourly,
    plan,
    servers_needed,
)

W = Workload(mau=1_000_000, dau_ratio=0.2, sessions_per_dau=1.0, turns_per_session=3,
             system_tokens=1000, user_tokens=20, output_tokens=60, peak_to_avg=2.0)


def test_session_input_tokens_resend_history():
    # turn 1: 1000+20; turn 2: 1000+40+60; turn 3: 1000+60+120
    assert input_tokens_per_session(1000, 20, 60, 3) == 1020 + 1100 + 1180
    assert input_tokens_per_session(1000, 20, 60, 1) == 1020
    with pytest.raises(ValueError):
        input_tokens_per_session(1000, 20, 60, 0)


def test_plan_volumes_and_littles_law():
    p = plan(W, latency_s=1.5)
    assert p["messages_per_day"] == pytest.approx(200_000 * 3)
    assert p["avg_rps"] == pytest.approx(600_000 / 86_400)
    assert p["peak_rps"] == pytest.approx(2 * p["avg_rps"])
    assert p["peak_rpm"] == pytest.approx(60 * p["peak_rps"])
    assert p["peak_concurrency"] == pytest.approx(1.5 * p["peak_rps"])
    assert p["input_tokens_per_day"] == pytest.approx(200_000 * 3300)
    assert p["peak_uncached_input_tpm"] == p["peak_input_tpm"]  # no cache configured


def test_plan_rejects_nonsense():
    with pytest.raises(ValueError):
        plan(Workload(**{**W.__dict__, "peak_to_avg": 0.5}), 1.0)
    with pytest.raises(ValueError):
        plan(Workload(**{**W.__dict__, "dau_ratio": 1.5}), 1.0)


def test_cache_discount_only_applies_to_cached_share():
    price = Price(usd_per_m_in=1.0, usd_per_m_out=0.0, cached_discount=0.1)
    no_cache = daily_cost(plan(W, 1), W, price)
    half = Workload(**{**W.__dict__, "cached_share": 0.5})
    assert daily_cost(plan(half, 1), half, price) == pytest.approx(no_cache * (0.5 + 0.05))
    assert plan(half, 1)["peak_uncached_input_tpm"] == pytest.approx(
        plan(W, 1)["peak_input_tpm"] / 2)


def test_peak_to_avg():
    assert peak_to_avg_from_hourly({h: 10 for h in range(24)}) == pytest.approx(1.0)
    spiky = {h: 0 for h in range(24)} | {12: 100}
    assert peak_to_avg_from_hourly(spiky) == pytest.approx(24.0)
    with pytest.raises(ValueError):
        peak_to_avg_from_hourly({h: 1 for h in range(23)})


def test_servers_needed_rounds_up_with_headroom():
    assert servers_needed(70, per_server=100, headroom=0.7) == 1
    assert servers_needed(70.1, per_server=100, headroom=0.7) == 2
    assert servers_needed(0.0, per_server=100) == 1
    with pytest.raises(ValueError):
        servers_needed(10, per_server=0)
