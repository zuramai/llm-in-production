"""ch20_canary.py: overlay generation and the promote/rollback verdict."""

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ch20_canary import overlay_yaml, parse_logs, verdict, version_stats


def test_overlay_merges_only_the_percentage():
    k = yaml.safe_load(overlay_yaml(12.5, "overlays/fakellm"))
    assert k["resources"] == ["../../../overlays/fakellm"]
    gen = k["configMapGenerator"][0]
    assert (gen["name"], gen["behavior"], gen["literals"]) == (
        "chat-config", "merge", ["CANARY_PERCENT=12.5"])
    with pytest.raises(ValueError):
        overlay_yaml(101, "base")


def _ev(version, status="ok", total=1000, sid="s"):
    return {"event": "chat", "prompt_version": version, "status": status, "session_id": sid,
            "ttft_ms": 100, "total_ms": total, "completion_tokens": 10, "cost_usd": 1e-5}


def test_parse_logs_skips_non_json_and_other_events():
    lines = ["INFO: Started server", '{"event": "ask"}', '{"broken',
             '{"event": "chat", "prompt_version": "v1", "status": "ok"}']
    text = "\n".join(lines)
    assert len(parse_logs(text)) == 1


def test_verdict_waits_promotes_and_rolls_back():
    stable = [_ev("v1", sid=f"a{i}") for i in range(100)]
    few = [_ev("v2") for _ in range(5)]
    assert verdict(version_stats(stable + few), "v1", "v2")[0] == "wait"
    good = [_ev("v2", sid=f"b{i}") for i in range(30)]
    assert verdict(version_stats(stable + good), "v1", "v2")[0] == "promote"
    errors = good[:27] + [_ev("v2", status="APITimeoutError") for _ in range(3)]
    decision, reasons = verdict(version_stats(stable + errors), "v1", "v2")
    assert decision == "rollback" and "error rate" in reasons[0]
    slow = [_ev("v2", total=2000, sid=f"c{i}") for i in range(30)]
    assert verdict(version_stats(stable + slow), "v1", "v2")[0] == "rollback"
