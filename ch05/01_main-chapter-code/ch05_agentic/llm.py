"""One place that talks to models: OpenRouter or Ollama, structured output, usage accounting."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field

import openai
from pydantic import BaseModel

from llmprod import chat_client, get_settings

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")


def client_for(model: str) -> openai.OpenAI:
    """OpenRouter ids look like `vendor/model`; Ollama tags look like `qwen3:8b`. (Chapter 4 sent
    Ollama chat to the embeddings URL; we keep them separate so embeddings can live on their own
    server, which a busy shared GPU box needed while this chapter was written.)"""
    if "/" in model:
        return chat_client(timeout=120)
    return openai.OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama", timeout=600, max_retries=1)


def strict_schema(model: type[BaseModel]) -> dict:
    """Pydantic schema in the shape strict structured outputs accept: every property required,
    no extra keys, at every level."""
    schema = model.model_json_schema()

    def fix(node: dict) -> None:
        if node.get("type") == "object" and "properties" in node:
            node["additionalProperties"] = False
            node["required"] = list(node["properties"])
        for value in node.values():
            if isinstance(value, dict):
                fix(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        fix(item)

    fix(schema)
    return schema


def response_format(model: type[BaseModel]) -> dict:
    return {"type": "json_schema",
            "json_schema": {"name": model.__name__, "strict": True, "schema": strict_schema(model)}}


_OBJ = re.compile(r"\{.*\}", re.DOTALL)


def parse_json(text: str, model: type[BaseModel]) -> BaseModel:
    """Validate the outermost {...} (some models wrap JSON in prose or fences)."""
    m = _OBJ.search(text or "")
    if not m:
        raise ValueError(f"no JSON object in reply: {text[:200]!r}")
    return model.model_validate(json.loads(m.group(0)))


@dataclass
class Usage:
    """Running totals for one question: calls, tokens, dollars (OpenRouter's `usage.cost`)."""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    llm_ms: float = 0.0
    log: list[dict] = field(default_factory=list)

    def add(self, step: str, resp, ms: float) -> None:
        u = resp.usage
        cost = 0.0
        if u is not None:
            cost = float(getattr(u, "cost", None) or (u.model_extra or {}).get("cost") or 0.0)
        pt = (u.prompt_tokens or 0) if u else 0
        ct = (u.completion_tokens or 0) if u else 0
        self.calls += 1
        self.prompt_tokens += pt
        self.completion_tokens += ct
        self.cost_usd += cost
        self.llm_ms += ms
        self.log.append({"step": step, "prompt_tokens": pt, "completion_tokens": ct,
                         "cost_usd": cost, "ms": round(ms)})

    def as_dict(self) -> dict:
        return {"calls": self.calls, "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "cost_usd": round(self.cost_usd, 8), "llm_ms": round(self.llm_ms)}


class LLM:
    """A model plus its client. `reasoning_effort="none"` everywhere: it switches off qwen3's
    thinking on Ollama, and gpt-6-luna only does function calling with it on Chat Completions."""

    def __init__(self, model: str | None = None, max_tokens: int = 800,
                 reasoning_effort: str | None = "none"):
        self.model = model or get_settings().llm_model
        self.client = client_for(self.model)
        self.max_tokens = max_tokens
        self.options = {"reasoning_effort": reasoning_effort} if reasoning_effort else {}

    def complete(self, messages: list[dict], usage: Usage, step: str, **kwargs):
        t0 = time.perf_counter()
        resp = self.client.chat.completions.create(
            model=self.model, messages=messages,
            max_tokens=kwargs.pop("max_tokens", self.max_tokens), **self.options, **kwargs)
        usage.add(step, resp, (time.perf_counter() - t0) * 1000)
        return resp

    def text(self, messages: list[dict], usage: Usage, step: str, **kwargs) -> str:
        resp = self.complete(messages, usage, step, **kwargs)
        return (resp.choices[0].message.content or "").strip()

    def structured(self, messages: list[dict], schema: type[BaseModel], usage: Usage,
                   step: str, **kwargs) -> BaseModel:
        """Strict JSON-schema output, validated with Pydantic. One retry with the error."""
        msgs = list(messages)
        for attempt in (1, 2):
            content = self.text(msgs, usage, step, response_format=response_format(schema),
                                **kwargs)
            try:
                return parse_json(content, schema)
            except ValueError as e:  # pydantic.ValidationError and JSONDecodeError included
                if attempt == 2:
                    raise
                msgs += [{"role": "assistant", "content": content},
                         {"role": "user", "content": f"Invalid JSON for the schema: {e}. "
                                                     "Reply with only the JSON object."}]
        raise AssertionError("unreachable")
