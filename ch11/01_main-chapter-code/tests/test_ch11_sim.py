import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "checkout"))

from ch11_sim import FaultState, PoolExhausted, SimPool, query_seconds


def test_pool_exhausts_after_wait_timeout_and_recovers_on_release():
    async def go():
        pool = SimPool(size=2, wait_timeout_s=0.05)
        await pool.acquire()
        await pool.acquire()
        assert pool.in_use == 2
        with pytest.raises(PoolExhausted, match="timed out waiting for connection"):
            await pool.acquire()
        assert pool.in_use == 2  # the failed waiter must not leak a slot
        pool.release()
        waited = await pool.acquire()
        assert pool.in_use == 2 and waited < 0.05
    asyncio.run(go())


def test_fault_state_rejects_unknown_scenario_and_keeps_old_state():
    f = FaultState(deploy_version="v1")
    with pytest.raises(ValueError):
        f.set("meteor-strike")
    assert f.scenario == "none"


def test_db_fault_slows_only_checkout_queries():
    f = FaultState()
    f.set("db-pool-exhaustion")
    assert query_seconds(f, "/checkout") >= 3.0
    assert query_seconds(f, "/products") < 0.1
