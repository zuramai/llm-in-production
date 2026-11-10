"""Unit tests for the benchmark helpers: SSE parsing, scoring, log parsing. No GPU, no network."""

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

import ch17_prefix_bench as prefix
import ch17_quant_bench as quant
import ch17_spec_bench as spec
import ch17_stream as stream

# ---------------------------------------------------------------- streaming client


def sse(*chunks: dict) -> bytes:
    lines = [f"data: {json.dumps(c)}\n\n" for c in chunks] + ["data: [DONE]\n\n"]
    return "".join(lines).encode()


def fake_client(payload: bytes, status: int = 200, seen: list | None = None) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(json.loads(request.content))
        return httpx.Response(status, content=payload)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_stream_chat_collects_text_usage_and_llama_timings():
    payload = sse(
        {"choices": [{"delta": {"role": "assistant"}}]},  # role-only chunk: not a token
        {"choices": [{"delta": {"content": "Hel"}}]},
        {"choices": [{"delta": {"content": "lo"}}]},
        {"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 2,
                                  "prompt_tokens_details": {"cached_tokens": 8}},
         "timings": {"prompt_n": 4, "cache_n": 8}},
    )
    seen: list = []
    with fake_client(payload, seen=seen) as c:
        r = stream.stream_chat(c, "http://x", {"messages": []})
    assert r.text == "Hello"
    assert (r.prompt_tokens, r.completion_tokens, r.cached_tokens) == (12, 2, 8)
    assert r.timings == {"prompt_n": 4, "cache_n": 8}
    assert r.ttft_ms <= r.total_ms
    # we always ask for streaming with a final usage chunk
    assert seen[0]["stream"] is True and seen[0]["stream_options"] == {"include_usage": True}


def test_stream_chat_reasoning_tokens_count_for_ttft_but_not_text():
    payload = sse({"choices": [{"delta": {"reasoning_content": "hmm"}}]},
                  {"choices": [{"delta": {"content": "ok"}}]})
    with fake_client(payload) as c:
        r = stream.stream_chat(c, "http://x", {})
    assert r.text == "ok"
    assert r.completion_tokens is None and r.decode_tok_s is None  # no usage chunk: no speed


def test_stream_chat_raises_with_server_message():
    with (fake_client(b'{"error": "model not found"}', status=404) as c,
          pytest.raises(httpx.HTTPStatusError, match="404.*model not found")):
        stream.stream_chat(c, "http://x", {})


def test_decode_speed_excludes_first_token():
    r = stream.StreamResult("", 100, 1100, 10, 11, first_to_last_ms=1000)
    assert r.decode_tok_s == pytest.approx(10.0)  # 10 tokens after the first, in 1 s
    assert stream.StreamResult("", 1, 1, 1, 1, first_to_last_ms=0).decode_tok_s is None


METRICS = """# HELP vllm:spec_decode_num_draft_tokens_total Number of draft tokens.
# TYPE vllm:spec_decode_num_draft_tokens_total counter
vllm:spec_decode_num_draft_tokens_total{engine="0",model_name="Qwen/Qwen3-8B"} 300.0
vllm:spec_decode_num_accepted_tokens_total{engine="0",model_name="Qwen/Qwen3-8B"} 120.0
vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",position="0"} 70.0
vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",position="1"} 50.0
vllm:gpu_cache_usage_perc{engine="0"} NaN
python_gc_objects_collected_total{generation="0"} 5.0
"""


def test_prometheus_counters_sums_labels_and_skips_other_prefixes():
    c = stream.prometheus_counters(METRICS)
    assert c["vllm:spec_decode_num_draft_tokens_total"] == 300.0
    assert c["vllm:spec_decode_num_accepted_tokens_per_pos_total"] == 120.0  # 70 + 50
    assert "vllm:gpu_cache_usage_perc" not in c  # NaN is dropped, not summed
    assert not any(k.startswith("python_") for k in c)


def test_counter_delta_treats_new_counters_as_zero_before():
    assert stream.counter_delta({"a": 5.0}, {"a": 7.5, "b": 2.0}) == {"a": 2.5, "b": 2.0}


# ---------------------------------------------------------------- spec bench


def test_acceptance_prefers_llama_timings_then_vllm_counters():
    llama = {"rows": [{"draft_n": 10, "draft_accepted": 7}, {"draft_n": 10, "draft_accepted": 3}]}
    assert spec.acceptance(llama) == 0.5
    vllm = {"rows": [{"draft_n": None, "draft_accepted": None}],
            "vllm_spec_counters": stream.prometheus_counters(METRICS)}
    assert spec.acceptance(vllm) == 0.4
    assert spec.acceptance({"rows": [{"draft_n": None, "draft_accepted": None}]}) is None


def test_spec_prompts_are_greedy_and_skip_thinking():
    b = spec.body("m", "hi", 64)
    assert b["temperature"] == 0 and b["max_tokens"] == 64 and b["model"] == "m"
    assert b["chat_template_kwargs"] == {"enable_thinking": False}
    assert "model" not in spec.body(None, "hi", 64)  # llama-server serves one model


# ---------------------------------------------------------------- prefix bench


def test_build_prefix_is_stable_bounded_and_faq_first():
    a, b = prefix.build_prefix(12_000), prefix.build_prefix(12_000)
    assert a == b  # a cache needs byte-identical prefixes
    assert len(a) <= 12_000 + 200  # header line on top of the docs budget
    assert a.index("## faq.md") < 200
    assert len(prefix.build_prefix(48_000)) > 3 * len(a) // 2
    assert prefix.build_prefix(48_000).startswith(a.split("\n## ", 2)[0])


def test_prefix_scenarios_per_engine():
    llama = prefix.scenarios("llama", "on")
    assert [s[1] for s in llama] == [False, True, True]  # cache_prompt per request
    assert [s[2] for s in llama] == [False, False, True]  # only the last one busts the cache
    assert [s[0] for s in prefix.scenarios("vllm", "off")] == ["cache off"]
    assert all(s[1] is None for s in prefix.scenarios("vllm", "on"))  # server flag, not a field


def test_prefix_request_body_only_sends_cache_prompt_when_set():
    assert "cache_prompt" not in prefix.request_body("m", "sys", "q", None)
    assert prefix.request_body(None, "sys", "q", False)["cache_prompt"] is False


# ---------------------------------------------------------------- quant bench


@dataclass
class T:
    id: int
    category: str
    priority: str
    hard: bool = False
    subject: str = "s"
    body: str = "b"
    channel: str = "email"


GOOD = json.dumps({"category": "refund", "priority": "high", "sentiment": "negative",
                   "order_id": 12, "summary": "Wants money back."})


def test_score_one_checks_category_and_priority_separately():
    r = quant.score_one(GOOD, T(1, "refund", "medium"))
    assert r["parsed"] and r["category_ok"] and not r["priority_ok"]


@pytest.mark.parametrize("raw,kind", [("", "no_output"), ("sure!", "not_json"),
                                      ('{"category": "refund"}', "invalid")])
def test_score_one_parse_failure_counts_as_wrong(raw, kind):
    r = quant.score_one(raw, T(1, "refund", "high"))
    assert not r["parsed"] and r["error"] == kind
    assert not r["category_ok"] and not r["priority_ok"]


def test_summarize_reports_hard_subset_and_failures():
    rows = [quant.score_one(GOOD, T(i, "refund", "high", hard=i < 2)) for i in range(4)]
    rows.append(quant.score_one("", T(9, "refund", "high", hard=True)))
    s = quant.summarize(rows)
    assert s["n"] == 5 and s["category_acc"] == 80.0 and s["parse_failures"] == 1
    assert s["hard_category_acc"] == pytest.approx(66.7)
    lo, hi = s["category_ci95"]
    assert lo < 80.0 < hi


def test_wilson_interval_bounds():
    lo, hi = quant.wilson(300, 300)
    assert hi == pytest.approx(1.0) and 0.98 < lo < 1.0  # never claims certainty
    lo, hi = quant.wilson(0, 10)
    assert lo == pytest.approx(0.0, abs=1e-12) and hi > 0.2
    with pytest.raises(ValueError):
        quant.wilson(5, 3)


def test_percentile_nearest_rank():
    v = list(range(1, 101))
    assert quant.percentile(v, 95) == 95 and quant.percentile(v, 50) == 50
    assert quant.percentile([7], 99) == 7
    with pytest.raises(ValueError):
        quant.percentile([], 50)


def test_parse_vllm_log_takes_the_last_value():
    log = (
        "INFO Model loading took 15.27 GiB memory and 7.1 seconds\n"
        "INFO Available KV cache memory: 3.12 GiB\n"
        "INFO GPU KV cache size: 22,720 tokens\n"
        "INFO Available KV cache memory: 3.05 GiB\n"
    )
    assert quant.parse_vllm_log(log) == {
        "weights_gib": 15.27, "kv_cache_gib": 3.05, "kv_cache_tokens": 22720}
    assert quant.parse_vllm_log("nothing here") == {}


def test_each_engine_gets_its_own_thinking_switch():
    assert quant.no_thinking("ollama") == {"reasoning_effort": "none"}
    assert quant.no_thinking("vllm") == {"chat_template_kwargs": {"enable_thinking": False}}


def test_triage_request_uses_chapter2_strict_schema():
    body = quant.triage_body(quant.CONFIGS["vllm-awq-int4"], T(1, "refund", "high"))
    fmt = body["response_format"]["json_schema"]
    assert fmt["strict"] is True and fmt["schema"]["additionalProperties"] is False
    assert body["model"] == "Qwen/Qwen3-8B-AWQ" and body["temperature"] == 0
    assert "<ticket>" in body["messages"][1]["content"]
