"""End-to-end runs of the repros that need no model: the failure must show up and the fix hold."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from repros import (
    c1_hnsw_filter,
    c2_tsquery_and,
    d2_retry_duplicates,
    d4_interrupt_rerun,
    e3_eval_gate_noise,
    f1_429_storm,
    f2_stream_proxy,
    f3_sync_healthz,
    f4_tiktoken_offline,
    f5_compose_env,
    g1_timezone_today,
    g3_strftime_locale,
    g4_dst_arithmetic,
    g5_enum_spelling,
    h1_pii_logs,
    j2_prometheus_missing,
)

LOCAL = [d4_interrupt_rerun, e3_eval_gate_noise, f1_429_storm, f2_stream_proxy, f3_sync_healthz,
         f4_tiktoken_offline, g3_strftime_locale, g4_dst_arithmetic, h1_pii_logs,
         j2_prometheus_missing]
NEEDS_DOCKER = [c1_hnsw_filter, c2_tsquery_and, d2_retry_duplicates, f5_compose_env,
                g1_timezone_today, g5_enum_spelling]


@pytest.mark.parametrize("mod", LOCAL, ids=lambda m: m.ID)
def test_local_repro(mod):
    res = mod.run()
    assert res.kind == "local"
    assert res.fixed, res.evidence
    if mod is not g3_strftime_locale or "id_ID.utf8" in _locales():
        assert res.reproduced, res.evidence


@pytest.mark.docker
@pytest.mark.parametrize("mod", NEEDS_DOCKER, ids=lambda m: m.ID)
def test_db_and_docker_repro(mod):
    res = mod.run()
    assert res.fixed, res.evidence
    assert res.reproduced, res.evidence


def _locales() -> str:
    import subprocess

    return subprocess.run(["locale", "-a"], capture_output=True, text=True, check=False).stdout
