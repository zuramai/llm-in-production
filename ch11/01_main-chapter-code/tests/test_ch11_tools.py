import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))

from ch11_agent import tools
from ch11_agent.tools import (
    get_deploys,
    parse_range,
    raw_logs,
    search_runbooks,
    summarize_logs,
    summarize_matrix,
    truncate,
)

DOCS = CODE.parents[1] / "datasets" / "shopflow" / "docs"


def test_parse_range_units_and_errors():
    assert parse_range("90s").total_seconds() == 90
    assert parse_range("2h").total_seconds() == 7200
    for bad in ("", "15", "m15", "1w", "-5m"):
        with pytest.raises(ValueError, match="bad range"):
            parse_range(bad)


def test_truncate_cuts_on_line_boundary_and_says_how_much_was_lost():
    text = "\n".join(f"line {i:03d}" for i in range(500))
    out = truncate(text, 100)
    assert len(out) < 220 and "truncated" in out
    assert out.split("\n")[0] == "line 000"
    assert truncate("short", 100) == "short"


def test_summarize_matrix_is_much_smaller_than_raw_and_keeps_peak():
    values = [[1700000000 + 5 * i, str(i if i != 400 else 999)] for i in range(720)]
    result = [{"metric": {"__name__": "x", "route": "/checkout"}, "values": values}]
    out = summarize_matrix(result)
    assert "max=999" in out and "route=/checkout" in out
    assert len(out) < len(json.dumps(result)) / 50


def test_summarize_matrix_empty_and_series_cap():
    assert "no data" in summarize_matrix([])
    many = [{"metric": {"i": str(i)}, "values": [[0, "1"], [5, "2"]]} for i in range(20)]
    out = summarize_matrix(many, max_series=8)
    assert out.count("first=") == 8 and "12 more series omitted" in out


def _streams(n_err=300, n_ok=300):
    base = 1_700_000_000_000_000_000
    err = [[str(base + i), json.dumps({"msg": "unhandled exception", "route": "/checkout",
            "status": 500, "deploy_version": "v9", "error": "TypeError"})] for i in range(n_err)]
    ok = [[str(base + i), json.dumps({"msg": "request completed", "route": "/products",
          "status": 200, "deploy_version": "v9"})] for i in range(n_ok)]
    return [{"stream": {"level": "error"}, "values": err}, {"stream": {"level": "info"}, "values": ok}]


def test_summarize_logs_groups_identical_lines_with_counts():
    out = summarize_logs(_streams())
    assert "600 log lines in 2 distinct patterns" in out
    assert "x300" in out and "deploy_version=v9" in out and "error=TypeError" in out
    assert len(out) < len(raw_logs(_streams(), limit=1000)) / 20


def test_summarize_logs_tolerates_non_json_lines():
    s = [{"stream": {"level": "info"}, "values": [["1700000000000000000", "plain text line"]]}]
    assert "plain text line" in summarize_logs(s)


def test_get_deploys_filters_by_window_and_orders_newest_first(tmp_path):
    p = tmp_path / "d.json"
    p.write_text(json.dumps([
        {"version": "v-old", "sha": "a", "deployed_at": "2026-11-05T00:00:00Z", "author": "x",
         "summary": "old"},
        {"version": "v-new", "sha": "b", "deployed_at": "2026-11-07T10:00:00Z", "author": "y",
         "summary": "new"},
        {"version": "v-mid", "sha": "c", "deployed_at": "2026-11-07T06:00:00Z", "author": "z",
         "summary": "mid"}]))
    now = datetime(2026, 11, 7, 12, 0, tzinfo=UTC)
    out = get_deploys("12h", now=now, path=p).splitlines()
    assert [line.split()[1] for line in out] == ["v-new", "v-mid"]
    assert "120 min ago" in out[0]
    assert "no deploys" in get_deploys("1h", now=now, path=p)
    assert get_deploys("soon", now=now, path=p).startswith("ERROR")


@pytest.mark.parametrize(("query", "expected"), [
    ("db pool exhausted connection", "checkout-db-pool-exhaustion.md"),
    ("deployctl rollback unhandled exception new version", "checkout-bad-deploy.md"),
    ("paypal stripe provider timeout failover idempotency", "payment-provider-timeout.md"),
])
def test_search_runbooks_finds_the_right_runbook_first(query, expected):
    assert expected in search_runbooks(query).split("\n", 1)[0]


def test_search_runbooks_output_is_bounded_and_handles_junk():
    assert len(search_runbooks("checkout")) <= tools.MAX_TOOL_CHARS + 120
    assert search_runbooks("zzzqqq") == "no matching runbook sections."
    assert search_runbooks("   ").startswith("ERROR")
    assert DOCS.exists()


def test_dispatch_reports_bad_arguments_as_text_not_exceptions():
    assert tools.dispatch("query_prometheus", {}).startswith("ERROR: missing argument")
    assert tools.dispatch("rm_rf", {}).startswith("ERROR: unknown tool")
