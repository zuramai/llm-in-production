import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ch10_compaction import Compaction, approx_tokens, prompt_chars, shrink_json


def history(n_orders: int) -> dict:
    return {
        "customer": {"id": 7, "full_name": "Ada"},
        "summary": {"orders": n_orders, "refunds": 4, "refunds_last_90_days": 3},
        "orders": [
            {"id": i, "status": "delivered", "items": [{"sku": f"SF-{i}", "qty": 1}] * 3}
            for i in range(n_orders)
        ],
        "tickets": [{"id": 1, "body": "x" * 5000}],
    }


def test_small_output_is_untouched():
    text = json.dumps({"id": 1, "status": "paid"})
    assert shrink_json(text, 100) == text


def test_shrunk_output_is_valid_json_within_budget_and_keeps_summary():
    text = json.dumps(history(300))
    assert len(text) > 40_000
    out = shrink_json(text, 2_500)
    assert len(out) <= 2_500
    data = json.loads(out)  # still valid JSON: the model can parse it
    assert data["summary"] == {"orders": 300, "refunds": 4, "refunds_last_90_days": 3}
    assert data["customer"]["id"] == 7


def test_omitted_marker_counts_the_dropped_items():
    out = json.loads(shrink_json(json.dumps(history(300)), 2_500))
    orders = out["orders"]
    marker = orders[-1]
    assert isinstance(marker, str) and "more items omitted" in marker
    kept = len(orders) - 1
    assert int(marker.split()[1]) == 300 - kept


def test_long_strings_are_cut_with_their_length():
    out = json.loads(shrink_json(json.dumps(history(1)), 1_000))
    body = out["tickets"][0]["body"]
    assert body.endswith("chars]") and len(body) < 500


def test_non_json_is_clipped_with_marker_inside_budget():
    out = shrink_json("a" * 10_000, 500)
    assert len(out) <= 500
    assert out.endswith("[truncated, 10000 chars total]")


def _conversation(big: str) -> list[dict]:
    call = {"id": "c1", "type": "function", "function": {"name": "h", "arguments": "{}"}}
    call2 = {"id": "c2", "type": "function", "function": {"name": "h", "arguments": "{}"}}
    return [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "", "tool_calls": [call]},
        {"role": "tool", "tool_call_id": "c1", "content": big},  # stale after the next call
        {"role": "assistant", "content": "", "tool_calls": [call2]},
        {"role": "tool", "tool_call_id": "c2", "content": big},  # fresh: not answered yet
    ]


def test_compact_view_gives_stale_outputs_a_smaller_budget():
    msgs = _conversation(json.dumps(history(300)))
    view = Compaction(mode="compact", fresh_chars=2_500, stale_chars=600).view(msgs)
    assert len(view[2]["content"]) <= 600
    assert 600 < len(view[4]["content"]) <= 2_500


def test_view_never_mutates_the_state():
    big = json.dumps(history(300))
    msgs = _conversation(big)
    Compaction().view(msgs)
    Compaction(mode="clip").view(msgs)
    assert msgs[2]["content"] == big and msgs[4]["content"] == big


def test_stable_mode_never_rewrites_earlier_messages():
    """Prompt caches match on an unchanged prefix: what the model saw once must not change."""
    big = json.dumps(history(300))
    msgs = _conversation(big)
    early = msgs[:3]  # the first tool output, before the second call existed
    stable, compact = Compaction(mode="stable"), Compaction(mode="compact")
    assert stable.view(early)[2] == stable.view(msgs)[2]
    assert compact.view(early)[2] != compact.view(msgs)[2]  # compact shrinks it once stale


def test_raw_and_clip_modes():
    big = json.dumps(history(300))
    msgs = _conversation(big)
    assert Compaction(mode="raw").view(msgs) == msgs
    clipped = Compaction(mode="clip").view(msgs)
    assert all(len(m["content"]) <= 6_000 for m in clipped if m["role"] == "tool")
    assert "truncated" in clipped[4]["content"]


def test_size_helpers():
    assert approx_tokens("abcd" * 10) == 10
    msgs = _conversation("xyz")
    assert prompt_chars(msgs) == len("q") + 2 * (len("{}") + len("h")) + 2 * 3
