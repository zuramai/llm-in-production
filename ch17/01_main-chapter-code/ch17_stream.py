"""One streaming Chat Completions request, timed from the client side.

Every benchmark in this chapter talks to an OpenAI-compatible server (llama-server, vLLM or
Ollama) and needs the same four numbers per request: time to first token, decode speed, token
counts and the server's own accounting if it sends any. We measure on the client, because that
is what a user feels, and keep the server's numbers next to them as a cross-check.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

import httpx


@dataclass
class StreamResult:
    text: str
    ttft_ms: float
    total_ms: float
    prompt_tokens: int | None
    completion_tokens: int | None
    cached_tokens: int | None = None
    first_to_last_ms: float = 0.0
    timings: dict = field(default_factory=dict)  # llama-server only

    @property
    def decode_tok_s(self) -> float | None:
        """Tokens per second after the first one: (n - 1) / (t_last - t_first)."""
        if not self.completion_tokens or self.completion_tokens < 2 or self.first_to_last_ms <= 0:
            return None
        return (self.completion_tokens - 1) / (self.first_to_last_ms / 1000)


def stream_chat(client: httpx.Client, base_url: str, body: dict) -> StreamResult:
    """POST ``body`` to ``{base_url}/v1/chat/completions`` with streaming and time it."""
    body = {**body, "stream": True, "stream_options": {"include_usage": True}}
    t0 = time.perf_counter()
    t_first = t_last = None
    parts: list[str] = []
    usage: dict = {}
    timings: dict = {}
    with client.stream("POST", f"{base_url}/v1/chat/completions", json=body) as resp:
        if resp.status_code >= 400:
            resp.read()
            raise httpx.HTTPStatusError(
                f"{resp.status_code}: {resp.text[:300]}", request=resp.request, response=resp
            )
        for line in resp.iter_lines():
            if not line.startswith("data: ") or line.endswith("[DONE]"):
                continue
            chunk = json.loads(line[6:])
            if chunk.get("usage"):
                usage = chunk["usage"]
            if chunk.get("timings"):
                timings = chunk["timings"]
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                piece = delta.get("content") or delta.get("reasoning_content") or ""
                if piece:
                    now = time.perf_counter()
                    t_first = t_first or now
                    t_last = now
                    parts.append(delta.get("content") or "")
    t_end = time.perf_counter()
    details = usage.get("prompt_tokens_details") or {}
    return StreamResult(
        text="".join(parts),
        ttft_ms=round(((t_first or t_end) - t0) * 1000, 1),
        total_ms=round((t_end - t0) * 1000, 1),
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        cached_tokens=details.get("cached_tokens"),
        first_to_last_ms=round(((t_last or t_end) - (t_first or t_end)) * 1000, 1),
        timings=timings,
    )


_METRIC = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([0-9.eE+-]+|NaN)$")


def prometheus_counters(text: str, prefix: str = "vllm:") -> dict[str, float]:
    """Sum every sample of each metric that starts with ``prefix`` (labels are collapsed)."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        m = _METRIC.match(line.strip())
        if not m or not m.group(1).startswith(prefix) or m.group(3) == "NaN":
            continue
        out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(3))
    return out


def counter_delta(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    """``after - before`` for every counter present after (missing before counts as 0)."""
    return {k: round(v - before.get(k, 0.0), 3) for k, v in after.items()}
