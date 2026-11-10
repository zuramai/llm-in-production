"""Token-bucket rate limiting: in-process (one server) and in Redis (shared by every replica).

A token bucket holds up to `capacity` tokens and refills at `rate` tokens per second. Each request
spends `cost` tokens (1 per request, or the request's estimated LLM tokens to enforce a TPM
budget). Empty bucket: reject, and tell the caller how long until enough tokens are back.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import redis
import redis.asyncio as aredis


class TokenBucket:
    """In-process token bucket. Exact, but every process has its own: N replicas, N× the limit."""

    def __init__(self, rate: float, capacity: float, clock: Callable[[], float] = time.monotonic):
        if rate <= 0 or capacity <= 0:
            raise ValueError("rate and capacity must be positive")
        self.rate, self.capacity, self._clock = rate, capacity, clock
        self._tokens = capacity
        self._ts = clock()

    def acquire(self, cost: float = 1.0) -> tuple[bool, float]:
        """Spend `cost` tokens if available. Returns (allowed, seconds until it would be)."""
        if cost > self.capacity:
            raise ValueError(f"cost {cost} exceeds bucket capacity {self.capacity}")
        now = self._clock()
        self._tokens = min(self.capacity, self._tokens + (now - self._ts) * self.rate)
        self._ts = now
        if self._tokens >= cost:
            self._tokens -= cost
            return True, 0.0
        return False, (cost - self._tokens) / self.rate


# Runs atomically inside Redis, so concurrent callers on any number of servers never race.
# Uses the Redis server clock (TIME), so app servers with skewed clocks still agree.
TOKEN_BUCKET_LUA = """
local rate = tonumber(ARGV[1])
local capacity = tonumber(ARGV[2])
local cost = tonumber(ARGV[3])
local t = redis.call('TIME')
local now = tonumber(t[1]) + tonumber(t[2]) / 1000000
local b = redis.call('HMGET', KEYS[1], 'tokens', 'ts')
local tokens = tonumber(b[1]) or capacity
local ts = tonumber(b[2]) or now
tokens = math.min(capacity, tokens + math.max(0, now - ts) * rate)
local allowed = 0
local retry_after = 0
if tokens >= cost then
  tokens = tokens - cost
  allowed = 1
else
  retry_after = (cost - tokens) / rate
end
redis.call('HSET', KEYS[1], 'tokens', tokens, 'ts', now)
redis.call('PEXPIRE', KEYS[1], math.ceil(capacity / rate * 1000) + 1000)
return {allowed, tostring(retry_after)}
"""


class RedisTokenBucket:
    """Token bucket stored in Redis: one limit shared by every process that uses the same key."""

    def __init__(self, client: redis.Redis, rate: float, capacity: float, prefix: str = "ch21:tb:"):
        if rate <= 0 or capacity <= 0:
            raise ValueError("rate and capacity must be positive")
        self.rate, self.capacity, self.prefix = rate, capacity, prefix
        self._script = client.register_script(TOKEN_BUCKET_LUA)

    def acquire(self, key: str, cost: float = 1.0) -> tuple[bool, float]:
        if cost > self.capacity:
            raise ValueError(f"cost {cost} exceeds bucket capacity {self.capacity}")
        allowed, retry_after = self._script(
            keys=[self.prefix + key], args=[self.rate, self.capacity, cost]
        )
        return bool(int(allowed)), float(retry_after)


class AsyncRedisTokenBucket:
    """`RedisTokenBucket` for asyncio code (FastAPI `async def` endpoints)."""

    def __init__(
        self, client: aredis.Redis, rate: float, capacity: float, prefix: str = "ch21:tb:"
    ):
        if rate <= 0 or capacity <= 0:
            raise ValueError("rate and capacity must be positive")
        self.rate, self.capacity, self.prefix = rate, capacity, prefix
        self._script = client.register_script(TOKEN_BUCKET_LUA)

    async def acquire(self, key: str, cost: float = 1.0) -> tuple[bool, float]:
        if cost > self.capacity:
            raise ValueError(f"cost {cost} exceeds bucket capacity {self.capacity}")
        allowed, retry_after = await self._script(
            keys=[self.prefix + key], args=[self.rate, self.capacity, cost]
        )
        return bool(int(allowed)), float(retry_after)


def hammer(kind: str, key: str, redis_url: str, seconds: float = 5.0, rate: float = 20,
           capacity: float = 20) -> int:
    """Demo worker: acquire from an in-process or a Redis bucket as fast as possible (every 2 ms)
    for `seconds`. Returns how many acquisitions were allowed. Run one per process."""
    if kind == "in-process":
        local = TokenBucket(rate, capacity)

        def acquire() -> bool:
            return local.acquire()[0]
    else:
        shared = RedisTokenBucket(redis.Redis.from_url(redis_url), rate, capacity)

        def acquire() -> bool:
            return shared.acquire(key)[0]

    allowed, end = 0, time.monotonic() + seconds
    while time.monotonic() < end:
        allowed += acquire()
        time.sleep(0.002)
    return allowed
