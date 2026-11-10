import asyncio
import json
import sys
from pathlib import Path

import httpx
import openai
import pytest
import tiktoken

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ch16_bench import (
    RequestRecord,
    build_prompts,
    load_corpus,
    one_request,
    parse_concurrency,
    percentile,
    run_level,
    summarize,
)

ENC = tiktoken.get_encoding("o200k_base")


# --- parse_concurrency -------------------------------------------------------------------------


def test_parse_concurrency_accepts_increasing_levels():
    assert parse_concurrency("1,2,4,8,16,32") == [1, 2, 4, 8, 16, 32]
    assert parse_concurrency(" 1, 3 ,") == [1, 3]


@pytest.mark.parametrize("text", ["", ",", "0,1", "-1", "4,2", "1,1,2", "1,two"])
def test_parse_concurrency_rejects_bad_levels(text):
    with pytest.raises(ValueError):
        parse_concurrency(text)


# --- percentile / tpot / summarize -------------------------------------------------------------


def test_percentile_matches_linear_interpolation():
    xs = [4.0, 1.0, 3.0, 2.0]  # unsorted on purpose
    assert percentile(xs, 50) == pytest.approx(2.5)
    assert percentile(xs, 95) == pytest.approx(3.85)
    assert percentile(xs, 0) == 1.0
    assert percentile(xs, 100) == 4.0
    assert percentile([7.0], 95) == 7.0
    assert percentile([], 50) is None
    with pytest.raises(ValueError):
        percentile(xs, 101)


def test_tpot_excludes_the_first_token():
    # 11 tokens: the first one arrives at TTFT, the other 10 during `decode`.
    r = RequestRecord(ok=True, ttft=0.3, e2e=1.3, decode=1.0, completion_tokens=11)
    assert r.tpot == pytest.approx(0.1)
    assert RequestRecord(ok=True, ttft=0.1, decode=0.0, completion_tokens=1).tpot is None
    assert RequestRecord(ok=False, error="boom").tpot is None


def test_summarize_counts_failures_but_keeps_them_out_of_latency_stats():
    good = [
        RequestRecord(ok=True, ttft=t, e2e=t + 2, decode=2.0, prompt_tokens=500,
                      completion_tokens=201)
        for t in (0.1, 0.2, 0.3)
    ]
    bad = RequestRecord(ok=False, error="APITimeoutError: timed out")
    lv = summarize(4, [*good, bad], wall_s=3.0)
    assert (lv.requests, lv.ok) == (4, 3)
    assert lv.errors == ["APITimeoutError: timed out"]
    assert lv.ttft_p50 == pytest.approx(0.2)
    assert lv.tpot_p50 == pytest.approx(0.01)
    assert lv.output_tok_per_s == pytest.approx(603 / 3.0)  # only successful tokens count
    assert lv.requests_per_s == pytest.approx(1.0)
    assert lv.mean_prompt_tokens == 500
    assert len(lv.raw) == 4


def test_summarize_all_failed_has_no_stats():
    lv = summarize(2, [RequestRecord(ok=False, error="x")] * 2, wall_s=1.0)
    assert lv.ok == 0 and lv.ttft_p50 is None and lv.mean_completion_tokens is None
    assert lv.output_tok_per_s == 0
    with pytest.raises(ValueError):
        summarize(1, [], wall_s=0)


# --- prompts -----------------------------------------------------------------------------------


def test_build_prompts_hit_the_token_target_and_never_share_a_prefix():
    corpus = load_corpus()
    prompts = build_prompts(50, 512, corpus, ENC, seed=1)
    lengths = [len(ENC.encode(p)) for p in prompts]
    assert all(abs(n - 512) <= 4 for n in lengths), lengths  # decode/encode may merge a token
    heads = {tuple(ENC.encode(p)[:8]) for p in prompts}
    assert len(heads) == 50  # unique within the first 8 tokens: nothing for a prefix cache
    assert build_prompts(5, 512, corpus, ENC, seed=1) == prompts[:5]  # reproducible
    assert build_prompts(5, 512, corpus, ENC, seed=2) != prompts[:5]


def test_build_prompts_rejects_impossible_sizes():
    with pytest.raises(ValueError):
        build_prompts(1, 63, "some corpus text " * 50, ENC)
    with pytest.raises(ValueError):
        build_prompts(0, 512, "some corpus text " * 50, ENC)
    (shortest,) = build_prompts(1, 64, "some corpus text " * 50, ENC)
    assert abs(len(ENC.encode(shortest)) - 64) <= 2


# --- streaming against a fake OpenAI-compatible server -----------------------------------------


def _sse(events: list[dict | str]) -> bytes:
    lines = [f"data: {e if isinstance(e, str) else json.dumps(e)}\n\n" for e in events]
    return "".join(lines).encode()


def _chunk(content=None, reasoning=None, usage=None, choices=True):
    delta = {}
    if content is not None:
        delta["content"] = content
    if reasoning is not None:
        delta["reasoning_content"] = reasoning
    return {
        "id": "c1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "m",
        "choices": [{"index": 0, "delta": delta, "finish_reason": None}] if choices else [],
        "usage": usage,
    }


def _client(handler) -> openai.AsyncOpenAI:
    return openai.AsyncOpenAI(
        base_url="http://fake/v1",
        api_key="x",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


def _stream_response(events):
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=_sse(events))


def test_one_request_prefers_server_usage_over_chunk_count():
    # vLLM under load packs several tokens per chunk: 3 chunks, 7 tokens.
    events = [
        _chunk(content="Hello wor"),
        _chunk(content="ld, how are"),
        _chunk(content=" you"),
        _chunk(usage={"prompt_tokens": 512, "completion_tokens": 7, "total_tokens": 519},
               choices=False),
        "[DONE]",
    ]
    rec = asyncio.run(one_request(_client(lambda req: _stream_response(events)), "m", "hi", 8, {}))
    assert rec.ok and rec.completion_tokens == 7 and rec.prompt_tokens == 512
    assert 0 <= rec.ttft <= rec.e2e


def test_one_request_counts_reasoning_tokens_and_falls_back_to_chunks():
    # A thinking model streams reasoning first; Ollama may omit usage.
    events = [_chunk(reasoning="Let me think"), _chunk(content="A"), _chunk(content="B"), "[DONE]"]
    rec = asyncio.run(one_request(_client(lambda req: _stream_response(events)), "m", "hi", 8, {}))
    assert rec.ok and rec.completion_tokens == 3 and rec.prompt_tokens is None


def test_one_request_reports_http_errors_and_empty_streams():
    def overloaded(req):
        return httpx.Response(503, json={"error": {"message": "Service Unavailable"}})

    rec = asyncio.run(one_request(_client(overloaded), "m", "hi", 8, {}))
    assert not rec.ok and rec.error.startswith("InternalServerError")
    empty = asyncio.run(one_request(_client(lambda r: _stream_response(["[DONE]"])), "m", "hi", 8, {}))
    assert not empty.ok and empty.error == "empty stream"


def test_one_request_sends_max_tokens_stream_usage_and_extra_body():
    seen = {}

    def handler(req):
        seen.update(json.loads(req.content))
        return _stream_response([_chunk(content="x"), "[DONE]"])

    extra = {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}, "ignore_eos": True}}
    asyncio.run(one_request(_client(handler), "Qwen/Qwen3-8B", "hi", 256, extra))
    assert seen["max_tokens"] == 256 and seen["stream"] is True
    assert seen["stream_options"] == {"include_usage": True}
    assert seen["ignore_eos"] is True
    assert seen["chat_template_kwargs"] == {"enable_thinking": False}


def test_run_level_never_exceeds_the_concurrency_limit():
    state = {"inflight": 0, "peak": 0}

    async def handler(req):
        state["inflight"] += 1
        state["peak"] = max(state["peak"], state["inflight"])
        await asyncio.sleep(0.02)
        state["inflight"] -= 1
        usage = {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}
        return _stream_response(
            [_chunk(content="a"), _chunk(content="b"), _chunk(usage=usage, choices=False), "[DONE]"]
        )

    lv = asyncio.run(run_level(_client(handler), "m", ["p"] * 12, 3, 8, {}))
    assert state["peak"] == 3
    assert (lv.requests, lv.ok, lv.concurrency) == (12, 12, 3)
    assert lv.output_tok_per_s == pytest.approx(24 / lv.wall_s)
