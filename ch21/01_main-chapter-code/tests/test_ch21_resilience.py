import asyncio
import random
import sys
from pathlib import Path

import httpx2
import openai
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ch21_resilience import (
    AllProvidersFailed,
    CircuitBreaker,
    Provider,
    complete_with_failover,
    exponential_delay,
    full_jitter_delay,
    is_retryable,
    retry_async,
)
from mock_llm_server import MockConfig, create_app


def mock_client(app) -> openai.AsyncOpenAI:
    http = httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app), base_url="http://mock")
    return openai.AsyncOpenAI(base_url="http://mock/v1", api_key="x", max_retries=0,
                              http_client=http)


def fast(**kw) -> MockConfig:
    return MockConfig(**{"ttft_ms": 1, "tokens_per_s": 10_000, "output_tokens": 5, "seed": 0,
                         **kw})


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def test_backoff_bounds():
    assert [exponential_delay(i, 0.5, 8) for i in range(6)] == [0.5, 1, 2, 4, 8, 8]
    rng = random.Random(1)
    for attempt in range(8):
        cap = exponential_delay(attempt, 0.5, 8)
        draws = [full_jitter_delay(attempt, 0.5, 8, rng) for _ in range(200)]
        assert all(0 <= d <= cap for d in draws)
        assert max(draws) - min(draws) > cap / 2  # actually spread out, not a constant
    with pytest.raises(ValueError):
        exponential_delay(-1)


def test_retry_async_retries_only_retryable():
    sleeps: list[float] = []

    async def fake_sleep(s: float) -> None:
        sleeps.append(s)

    calls = {"n": 0}

    async def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise openai.APIConnectionError(request=httpx2.Request("POST", "http://x"))
        return "ok"

    assert asyncio.run(retry_async(flaky, attempts=4, delay=lambda i: i + 1.0,
                                   sleep=fake_sleep)) == "ok"
    assert calls["n"] == 3 and sleeps == [1.0, 2.0]

    async def bad():
        raise ValueError("bug")

    sleeps.clear()
    with pytest.raises(ValueError):
        asyncio.run(retry_async(bad, attempts=4, sleep=fake_sleep))
    assert sleeps == []  # non-retryable: no retries at all


def test_retry_async_gives_up_after_attempts():
    calls = {"n": 0}

    async def always_down():
        calls["n"] += 1
        raise openai.APIConnectionError(request=httpx2.Request("POST", "http://x"))

    async def no_sleep(_):
        return None

    with pytest.raises(openai.APIConnectionError):
        asyncio.run(retry_async(always_down, attempts=3, sleep=no_sleep))
    assert calls["n"] == 3


def test_circuit_breaker_lifecycle():
    clock = Clock()
    b = CircuitBreaker(failure_threshold=2, reset_after_s=10, clock=clock)
    b.record_failure()
    assert b.state == "closed" and b.allow()
    b.record_failure()
    assert b.state == "open" and not b.allow()
    clock.t = 10
    assert b.state == "half-open"
    assert b.allow() and not b.allow()  # exactly one trial call
    b.record_failure()  # trial failed: open again for a full period
    assert b.state == "open"
    clock.t = 20
    assert b.allow()
    b.record_success()
    assert b.state == "closed" and b.failures == 0


def test_mock_streams_openai_compatible_chunks_with_usage():
    async def go():
        client = mock_client(create_app(fast(output_tokens=7)))
        stream = await client.chat.completions.create(
            model="mock", messages=[{"role": "user", "content": "hi" * 40}], max_tokens=50,
            stream=True, stream_options={"include_usage": True})
        parts, usage = [], None
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                parts.append(chunk.choices[0].delta.content)
            if chunk.usage:
                usage = chunk.usage
        return parts, usage

    parts, usage = asyncio.run(go())
    assert len(parts) == 7 and usage.completion_tokens == 7 and usage.prompt_tokens == 20


def test_mock_respects_max_tokens_and_429_rate():
    async def go():
        app = create_app(fast(output_tokens=60, rate_429=1.0, retry_after_s=2))
        client = mock_client(app)
        with pytest.raises(openai.RateLimitError) as e:
            await client.chat.completions.create(model="m", messages=[], max_tokens=3)
        assert e.value.response.headers["retry-after"] == "2"
        ok = mock_client(create_app(fast(output_tokens=60)))
        r = await ok.chat.completions.create(model="m", messages=[], max_tokens=3)
        return r

    r = asyncio.run(go())
    assert r.usage.completion_tokens == 3


def test_mock_provider_bucket_limits_bursts():
    async def go():
        app = create_app(fast(rps_limit=1, burst=3))
        client = mock_client(app)
        results = []
        for _ in range(5):
            try:
                await client.chat.completions.create(model="m", messages=[], max_tokens=1)
                results.append(200)
            except openai.RateLimitError:
                results.append(429)
        return results

    assert asyncio.run(go()) == [200, 200, 200, 429, 429]


def test_failover_skips_failing_provider_and_opens_its_circuit():
    async def go():
        down = Provider("down", mock_client(create_app(fast(rate_500=1.0))), "m",
                        breaker=CircuitBreaker(failure_threshold=2, reset_after_s=60))
        up = Provider("up", mock_client(create_app(fast())), "m")
        served = []
        for _ in range(3):
            name, text, meta = await complete_with_failover([down, up], [], max_tokens=5)
            served.append((name, meta["skipped"].get("down", "")))
            assert text
        return served, down.breaker.state

    served, state = asyncio.run(go())
    assert [s[0] for s in served] == ["up", "up", "up"]
    assert served[0][1].startswith("InternalServerError")
    assert served[2][1] == "circuit open"  # third call doesn't even try the dead provider
    assert state == "open"


def test_failover_raises_when_all_fail_and_never_on_bad_request():
    async def go():
        a = Provider("a", mock_client(create_app(fast(rate_429=1.0))), "m")
        b = Provider("b", mock_client(create_app(fast(rate_500=1.0))), "m")
        with pytest.raises(AllProvidersFailed) as e:
            await complete_with_failover([a, b], [], max_tokens=5)
        assert set(e.value.errors) == {"a", "b"}

    asyncio.run(go())


def test_is_retryable_classification():
    req = httpx2.Request("POST", "http://x")

    def status(code: int) -> openai.APIStatusError:
        return openai.APIStatusError("x", response=httpx2.Response(code, request=req), body=None)

    assert is_retryable(status(429)) and is_retryable(status(503)) and is_retryable(status(529))
    assert not is_retryable(status(400)) and not is_retryable(status(401))
    assert is_retryable(openai.APITimeoutError(request=req))
    assert not is_retryable(ValueError())
