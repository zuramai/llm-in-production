import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "checkout"))

from ch11_loki_handler import LokiHandler, build_push_payload


def rec(level, msg, created, **fields):
    r = logging.LogRecord("checkout", level, __file__, 1, msg, None, None)
    r.created = created
    r.fields = fields
    return r


def test_groups_by_level_and_labels_are_exactly_service_and_level():
    p = build_push_payload([rec(logging.INFO, "ok", 1.0), rec(logging.ERROR, "boom", 2.0),
                            rec(logging.INFO, "ok2", 3.0)], "checkout")
    by_level = {s["stream"]["level"]: s for s in p["streams"]}
    assert set(by_level) == {"info", "error"}
    assert all(s["stream"].keys() == {"service", "level"} for s in p["streams"])
    assert len(by_level["info"]["values"]) == 2


def test_line_is_json_with_contract_fields_and_drops_unknown_fields():
    r = rec(logging.ERROR, "payment provider timeout", 1.5, provider="paypal", status=504,
            route="/checkout", deploy_version="v1", secret_card="4242")
    (stream,) = build_push_payload([r], "checkout")["streams"]
    ts, line = stream["values"][0]
    d = json.loads(line)
    assert d == {"msg": "payment provider timeout", "provider": "paypal", "status": 504,
                 "route": "/checkout", "deploy_version": "v1"}  # no secret_card
    assert ts == "1500000000" and isinstance(ts, str)  # nanoseconds as a string


def test_timestamps_strictly_increase_even_when_records_share_a_clock_tick():
    recs = [rec(logging.INFO, f"m{i}", 10.0) for i in range(3)]
    values = build_push_payload(recs, "checkout")["streams"][0]["values"]
    ts = [int(v[0]) for v in values]
    assert ts == sorted(set(ts)) and len(ts) == 3


def test_queue_overflow_drops_instead_of_blocking(monkeypatch):
    monkeypatch.setattr(LokiHandler, "_loop", lambda self: None)  # no background thread
    h = LokiHandler("http://127.0.0.1:1/push", max_queue=2)
    for i in range(5):
        h.emit(rec(logging.INFO, f"m{i}", float(i)))
    assert h.q.qsize() == 2 and h.dropped == 3


def test_push_failure_is_swallowed_and_counted(monkeypatch):
    monkeypatch.setattr(LokiHandler, "_loop", lambda self: None)
    h = LokiHandler("http://127.0.0.1:1/push")  # nothing listens on port 1
    assert h._push([rec(logging.INFO, "x", 1.0)]) is False
    assert h.dropped == 1
