"""Retries with jitter, a circuit breaker, and multi-provider failover for streaming chat calls."""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import TypeVar

import openai

T = TypeVar("T")

# 408 timeout, 409 conflict/lock, 429 rate limit, 5xx server, 529 Anthropic "overloaded".
RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504, 529})


def is_retryable(exc: BaseException) -> bool:
    """Transient failures worth retrying (or failing over). 400/401/404 are bugs: never retry."""
    if isinstance(exc, openai.APIConnectionError):  # includes APITimeoutError
        return True
    if isinstance(exc, openai.APIStatusError):
        return exc.status_code in RETRYABLE_STATUS
    return False


def exponential_delay(attempt: int, base: float = 0.5, cap: float = 8.0) -> float:
    """Backoff without jitter: base, 2·base, 4·base, ... capped. `attempt` starts at 0."""
    if attempt < 0:
        raise ValueError("attempt must be >= 0")
    return min(cap, base * 2**attempt)


def full_jitter_delay(
    attempt: int, base: float = 0.5, cap: float = 8.0, rng: random.Random | None = None
) -> float:
    """AWS "full jitter": uniform in [0, min(cap, base·2^attempt)]. Spreads synchronized clients."""
    return (rng or random).uniform(0, exponential_delay(attempt, base, cap))


async def retry_async(
    fn: Callable[[], Awaitable[T]],
    *,
    attempts: int = 4,
    delay: Callable[[int], float] = full_jitter_delay,
    retryable: Callable[[BaseException], bool] = is_retryable,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> T:
    """Call `fn` up to `attempts` times, sleeping `delay(i)` after the i-th retryable failure."""
    if attempts < 1:
        raise ValueError("attempts must be >= 1")
    for attempt in range(attempts):
        try:
            return await fn()
        except Exception as exc:
            if attempt == attempts - 1 or not retryable(exc):
                raise
            await sleep(delay(attempt))
    raise AssertionError("unreachable")


@dataclass
class CircuitBreaker:
    """Stop calling a provider after `failure_threshold` consecutive failures.

    closed → (N failures) → open → (after `reset_after_s`) → half-open: one trial call.
    Success closes it again; a failure re-opens it for another `reset_after_s`.
    """

    failure_threshold: int = 3
    reset_after_s: float = 30.0
    clock: Callable[[], float] = time.monotonic
    failures: int = 0
    opened_at: float | None = None
    _trial_in_flight: bool = False

    @property
    def state(self) -> str:
        if self.opened_at is None:
            return "closed"
        if self.clock() - self.opened_at >= self.reset_after_s:
            return "half-open"
        return "open"

    def allow(self) -> bool:
        state = self.state
        if state == "closed":
            return True
        if state == "half-open" and not self._trial_in_flight:
            self._trial_in_flight = True
            return True
        return False

    def record_success(self) -> None:
        self.failures, self.opened_at, self._trial_in_flight = 0, None, False

    def record_failure(self) -> None:
        self.failures += 1
        self._trial_in_flight = False
        if self.opened_at is not None or self.failures >= self.failure_threshold:
            self.opened_at = self.clock()


@dataclass
class Provider:
    name: str
    client: openai.AsyncOpenAI
    model: str
    options: dict = field(default_factory=dict)  # e.g. {"reasoning_effort": "none"}
    breaker: CircuitBreaker = field(default_factory=CircuitBreaker)


class AllProvidersFailed(Exception):
    def __init__(self, errors: dict[str, str]):
        super().__init__(f"all providers failed: {errors}")
        self.errors = errors


async def stream_with_failover(
    providers: list[Provider],
    messages: list[dict],
    *,
    max_tokens: int = 800,
    meta: dict | None = None,
) -> AsyncIterator[str]:
    """Stream text deltas from the first healthy provider, failing over in order.

    Failover only happens *before the first token*: once the user has seen part of an answer,
    switching models mid-sentence would splice two different answers, so later errors propagate.
    `meta` (if given) receives "provider", "usage" and "skipped" (name → reason).
    """
    meta = meta if meta is not None else {}
    skipped: dict[str, str] = {}
    meta["skipped"] = skipped
    for p in providers:
        if not p.breaker.allow():
            skipped[p.name] = f"circuit {p.breaker.state}"
            continue
        started = False
        try:
            stream = await p.client.chat.completions.create(
                model=p.model, messages=messages, max_tokens=max_tokens, stream=True,
                stream_options={"include_usage": True}, **p.options,
            )
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    if not started:
                        started = True
                        meta["provider"] = p.name
                    yield chunk.choices[0].delta.content
                if chunk.usage is not None:
                    meta["usage"] = chunk.usage.model_dump(exclude_none=True)
        except Exception as exc:
            if not is_retryable(exc):
                p.breaker.record_success()  # the provider answered; the request was bad
                raise
            p.breaker.record_failure()
            if started:
                raise
            skipped[p.name] = f"{type(exc).__name__}: {getattr(exc, 'status_code', '')}".strip()
            continue
        p.breaker.record_success()
        meta.setdefault("provider", p.name)
        return
    raise AllProvidersFailed(skipped)


async def complete_with_failover(
    providers: list[Provider], messages: list[dict], *, max_tokens: int = 800
) -> tuple[str, str, dict]:
    """Non-streaming convenience wrapper: returns (provider name, text, meta)."""
    meta: dict = {}
    parts = [d async for d in stream_with_failover(providers, messages, max_tokens=max_tokens,
                                                    meta=meta)]
    return meta.get("provider", "?"), "".join(parts), meta
