"""The loop's hard stops: step cap, token budget, tool errors as messages, parallel calls."""

import json
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ch09_agent import run_agent
from ch09_fakes import FakeClient, response, tool_call
from ch09_tools import LocalToolbox, ToolError


def make_toolbox(log: list[str]) -> LocalToolbox:
    def ping(n: int) -> dict:
        """Echo n."""
        log.append(f"ping{n}")
        return {"n": n}

    def broken(n: int) -> dict:
        """Always fails with an anticipated error."""
        raise ToolError(f"nothing known about {n}")

    return LocalToolbox({"ping": ping, "broken": broken})


def test_step_cap_stops_and_does_not_execute_the_last_requested_tools():
    log: list[str] = []
    client = FakeClient([response(tool_calls=[tool_call("c", "ping", '{"n": 1}')])])
    res = run_agent("go", make_toolbox(log), client=client, model="m", max_steps=3)
    assert res.stop_reason == "max_steps"
    assert len(client.calls) == 3
    assert log == ["ping1", "ping1"]  # steps 1 and 2 executed; step 3's call was refused
    # Every assistant tool_call must still get a tool message, or the next API call would 400.
    ids = [m["tool_call_id"] for m in res.messages if m["role"] == "tool"]
    assert len(ids) == 3
    assert "not executed" in res.messages[-1]["content"]
    assert res.answer  # the caller always gets a string back


def test_token_budget_stops_before_running_tools_it_cannot_afford_to_read():
    log: list[str] = []
    step = response(tool_calls=[tool_call("c", "ping", '{"n": 2}')], tokens=600)  # prompt 590
    client = FakeClient([step])
    res = run_agent("go", make_toolbox(log), client=client, model="m", token_budget=2000)
    assert res.stop_reason == "token_budget"
    # step 1: 600 spent + next prompt >= 590 = 1190 < 2000 -> run the tool
    # step 2: 1200 + 590 = 1790 < 2000 -> run the tool
    # step 3: 1800 + 590 = 2390 >= 2000 -> the call that reads the result can't be paid: stop
    assert log == ["ping2", "ping2"]
    assert len(client.calls) == 3
    assert res.total_tokens == 1800  # under the budget: never overshoots by a whole extra call
    assert "not executed" in res.messages[-1]["content"]


def test_budget_counts_the_next_prompt_not_only_what_was_spent():
    """600 < 1000 already spent, but reading the tool result needs >= 590 more: don't run it."""
    log: list[str] = []
    step = response(tool_calls=[tool_call("c", "ping", '{"n": 3}')], tokens=600)
    client = FakeClient([step])
    res = run_agent("go", make_toolbox(log), client=client, model="m", token_budget=1000)
    assert res.stop_reason == "token_budget"
    assert log == []
    assert len(client.calls) == 1


def test_large_tool_results_count_against_the_budget_before_the_next_call():
    """The tool ran, but reading its ~6,000-character (clipped) result can't fit: no 2nd call."""

    def big(n: int) -> dict:
        """Returns a lot."""
        return {"blob": "z" * 8_000}

    client = FakeClient([response(tool_calls=[tool_call("c", "big", '{"n": 1}')], tokens=600)])
    res = run_agent("go", LocalToolbox({"big": big}), client=client, model="m", token_budget=2500)
    assert len(res.tool_calls) == 1  # 600 + 590 < 2500, so the tool ran
    assert res.stop_reason == "token_budget"
    assert len(client.calls) == 1  # 600 + 590 + ~1,500 (6,000-char cap / 4) >= 2500


def test_budget_exhausted_up_front_makes_no_model_call():
    client = FakeClient([response(content="hi")])
    res = run_agent("go", make_toolbox([]), client=client, model="m", token_budget=0)
    assert res.stop_reason == "token_budget"
    assert client.calls == []


def test_tool_error_becomes_a_tool_message_and_the_loop_continues():
    client = FakeClient(
        [
            response(tool_calls=[tool_call("c1", "broken", '{"n": 7}')]),
            response(content="I could not find 7."),
        ]
    )
    res = run_agent("go", make_toolbox([]), client=client, model="m")
    assert res.stop_reason == "done"
    assert res.answer == "I could not find 7."
    sent_back = client.calls[1]["messages"][-1]
    assert sent_back["role"] == "tool" and sent_back["tool_call_id"] == "c1"
    assert json.loads(sent_back["content"]) == {"error": "nothing known about 7"}
    assert res.tool_calls[0].is_error is True


def test_parallel_calls_all_run_and_answer_in_request_order():
    log: list[str] = []
    client = FakeClient(
        [
            response(
                tool_calls=[
                    tool_call("a", "ping", '{"n": 1}'),
                    tool_call("b", "ping", '{"n": 2}'),
                    tool_call("c", "ping", '{"n": 3}'),
                ]
            ),
            response(content="done"),
        ]
    )
    res = run_agent("go", make_toolbox(log), client=client, model="m")
    assert sorted(log) == ["ping1", "ping2", "ping3"]
    tool_msgs = [m for m in client.calls[1]["messages"] if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == ["a", "b", "c"]
    assert [json.loads(m["content"])["n"] for m in tool_msgs] == [1, 2, 3]
    assert res.max_parallel == 3


def test_provider_failure_is_reported_not_raised():
    client = FakeClient([RuntimeError("503 upstream")])
    res = run_agent("go", make_toolbox([]), client=client, model="m")
    assert res.stop_reason == "llm_error"
    assert "503 upstream" in res.answer


def test_max_tokens_is_always_sent():
    client = FakeClient([response(content="ok")])
    run_agent("go", make_toolbox([]), client=client, model="m", max_tokens=321)
    assert client.calls[0]["max_tokens"] == 321


def test_largest_single_prompt_is_tracked_separately_from_the_sum():
    """The budget uses the sum; the context-window check needs the largest single prompt."""
    client = FakeClient(
        [
            response(tool_calls=[tool_call("a", "ping", '{"n": 1}')], tokens=500),
            response(tool_calls=[tool_call("b", "ping", '{"n": 2}')], tokens=900),
            response(content="done", tokens=700),
        ]
    )
    res = run_agent("go", make_toolbox([]), client=client, model="m")
    assert res.prompt_tokens == 490 + 890 + 690
    assert res.max_prompt_tokens == 890
