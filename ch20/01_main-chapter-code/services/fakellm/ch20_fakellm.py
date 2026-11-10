"""A fake OpenAI-compatible LLM for load, rollout and autoscaling tests. No model, no cost.

    uvicorn ch20_fakellm:app --port 8220

POST /v1/chat/completions streams FAKE_TOKENS words, one every FAKE_TOKEN_DELAY_MS, after
FAKE_TTFT_MS. The first words report how many user messages the request carried, so a client
can check whether the chat service remembered the conversation:
"[fake] user_messages=2 ...".
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

TOKENS = int(os.environ.get("FAKE_TOKENS", "40"))
TOKEN_DELAY_S = int(os.environ.get("FAKE_TOKEN_DELAY_MS", "50")) / 1000
TTFT_S = int(os.environ.get("FAKE_TTFT_MS", "300")) / 1000

app = FastAPI(title="fake LLM")


def _words(messages: list[dict], n: int) -> list[str]:
    user_messages = sum(1 for m in messages if m.get("role") == "user")
    head = ["[fake]", f"user_messages={user_messages}"]
    return (head + ["lorem"] * max(n - len(head), 0))[:n]


def _usage(messages: list[dict], completion_tokens: int) -> dict:
    prompt_tokens = sum(len(str(m.get("content", ""))) for m in messages) // 4
    return {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens, "cost": 0.0}


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    n = min(TOKENS, int(body.get("max_tokens") or TOKENS))
    words = _words(messages, n)
    rid, created, model = f"chatcmpl-{uuid.uuid4().hex[:12]}", int(time.time()), body["model"]

    if not body.get("stream"):
        await asyncio.sleep(TTFT_S + TOKEN_DELAY_S * n)
        return JSONResponse({
            "id": rid, "object": "chat.completion", "created": created, "model": model,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": " ".join(words)}}],
            "usage": _usage(messages, n),
        })

    def chunk(delta: dict, finish: str | None = None, usage: dict | None = None) -> str:
        choices = [] if usage else [{"index": 0, "delta": delta, "finish_reason": finish}]
        payload = {"id": rid, "object": "chat.completion.chunk", "created": created,
                   "model": model, "choices": choices, "usage": usage}
        return f"data: {json.dumps(payload)}\n\n"

    async def stream():
        await asyncio.sleep(TTFT_S)
        for i, word in enumerate(words):
            yield chunk({"role": "assistant", "content": word if i == 0 else " " + word})
            await asyncio.sleep(TOKEN_DELAY_S)
        yield chunk({}, finish="stop")
        if (body.get("stream_options") or {}).get("include_usage"):
            yield chunk({}, usage=_usage(messages, n))
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")
