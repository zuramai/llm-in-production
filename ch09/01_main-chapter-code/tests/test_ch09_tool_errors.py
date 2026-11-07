"""Tool registry: schema derivation, and that bad calls become error messages, never exceptions."""

import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ch09_tools as tools
from ch09_mcp_client import mcp_tool_to_openai
from ch09_mcp_server import build_server


def make_toolbox() -> tools.LocalToolbox:
    def add(a: int, b: int = 1) -> dict:
        """Add two ints."""
        return {"sum": a + b}

    def anticipated(x: int) -> dict:
        """Fails the way we planned."""
        raise tools.ToolError("order 9 not found")

    def crash(x: int) -> dict:
        """Fails the way we didn't."""
        raise RuntimeError("password=hunter2 in stack trace")

    def big(x: int) -> dict:
        """Returns far too much."""
        return {"blob": "z" * 50_000}

    return tools.LocalToolbox({f.__name__: f for f in (add, anticipated, crash, big)})


@pytest.mark.parametrize(
    ("name", "raw", "needle"),
    [
        ("nope", "{}", "unknown tool 'nope'"),
        ("add", "{not json", "not valid JSON"),
        ("add", "[1, 2]", "must be a JSON object"),
        ("add", '{"b": 2}', "a: Field required"),
        ("add", '{"a": "seven"}', "a: Input should be a valid integer"),
        ("add", '{"a": 1, "c": 3}', "c: Extra inputs are not permitted"),
        ("anticipated", '{"x": 1}', "order 9 not found"),
    ],
)
def test_bad_calls_return_a_json_error_string(name, raw, needle):
    out = make_toolbox().call(name, raw)  # must not raise
    assert needle in json.loads(out)["error"]


def test_unexpected_crash_does_not_leak_internals_to_the_model():
    out = make_toolbox().call("crash", '{"x": 1}')
    assert "hunter2" not in out
    assert "RuntimeError" in json.loads(out)["error"]


def test_huge_results_are_truncated():
    out = make_toolbox().call("big", '{"x": 1}')
    assert len(out) < tools.MAX_RESULT_CHARS + 100
    assert "truncated" in out


def test_good_call_and_default_argument():
    assert json.loads(make_toolbox().call("add", '{"a": 2}')) == {"sum": 3}
    assert json.loads(make_toolbox().call("add", "")) == {
        "error": "invalid arguments for add: a: Field required"
    }


def test_spec_is_derived_from_the_signature():
    spec = {s["function"]["name"]: s["function"] for s in tools.LocalToolbox().specs}
    refund = spec["create_refund"]["parameters"]
    assert refund["additionalProperties"] is False
    assert set(refund["required"]) == {"order_id", "amount_cents", "reason", "idempotency_key"}
    assert "damaged" in refund["properties"]["reason"]["enum"]
    assert refund["properties"]["amount_cents"]["type"] == "integer"
    assert "title" not in json.dumps(spec)  # titles only cost tokens


def test_mcp_server_advertises_the_same_tools_as_the_raw_toolbox():
    """One definition, two surfaces: the MCP schemas must match the OpenAI ones."""
    raw = {s["function"]["name"]: s["function"] for s in tools.LocalToolbox().specs}
    listed = asyncio.run(build_server().list_tools())
    via_mcp = {t.name: mcp_tool_to_openai(t)["function"] for t in listed}
    assert via_mcp.keys() == raw.keys()
    for name, fn in raw.items():
        a, b = fn["parameters"], via_mcp[name]["parameters"]
        assert set(a["properties"]) == set(b["properties"])
        assert set(a["required"]) == set(b["required"])
        assert via_mcp[name]["description"] == fn["description"]
    reason = via_mcp["create_refund"]["parameters"]["properties"]["reason"]
    assert "damaged" in reason["enum"]


def test_only_create_refund_is_flagged_as_a_write():
    by_name = {t.name: t for t in asyncio.run(build_server().list_tools())}
    assert by_name["get_order"].annotations.read_only_hint is True
    assert by_name["search_orders"].annotations.read_only_hint is True
    assert by_name["create_refund"].annotations.read_only_hint is False
