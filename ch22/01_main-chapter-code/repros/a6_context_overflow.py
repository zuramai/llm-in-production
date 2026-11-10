"""A6: context overflow is often silent: the router or server trims, the model answers anyway."""

from __future__ import annotations

import os

from ch22_common import Result, chat, model_catalog, timed

ID, TITLE, LAYER, KIND = "A6", "Context overflow is silent", "API & tokens", "hosted"
SMALL = "gryphe/mythomax-l2-13b"  # one of the few small-context models left on OpenRouter

CHARS_PER_TOKEN = 3.0  # deliberately pessimistic for English logs; real ratio is ~4


class ContextOverflow(ValueError):
    pass


def effective_context(meta: dict) -> int:
    """The context the serving provider really gives you, not the model card's number."""
    top = (meta.get("top_provider") or {}).get("context_length")
    return min(x for x in (meta.get("context_length"), top) if x)


def preflight(prompt: str, context: int, max_tokens: int) -> int:
    """Estimate prompt tokens and refuse to send what can't fit with room for the answer."""
    estimate = int(len(prompt) / CHARS_PER_TOKEN) + 1
    if estimate + max_tokens > context:
        raise ContextOverflow(f"~{estimate} prompt + {max_tokens} output > {context} context")
    return estimate


def build_prompt(n_lines: int = 700) -> str:
    lines = [f"Log line {i}: shipment {1000 + i} scanned at hub {i % 7}, no exceptions."
             for i in range(n_lines)]
    lines.insert(n_lines // 2, "Log line X: NOTE the vault code is 7291.")
    return "\n".join(lines) + "\n\nQuestion: what is the vault code? Answer with the number only."


def ollama_truncation(model: str = "qwen3:8b", num_ctx: int = 2048) -> dict:
    """RUN_LOCAL=1 only: the same kind of prompt against Ollama with a small num_ctx.

    Ollama's OpenAI endpoint takes no num_ctx, so this uses the native /api/chat, which reports
    prompt_eval_count (tokens actually evaluated) next to the answer.
    """
    import httpx

    prompt = build_prompt(n_lines=300)
    r = httpx.post("http://localhost:11434/api/chat", timeout=600, json={
        "model": model, "stream": False, "think": False,
        "options": {"num_ctx": num_ctx, "num_predict": 10, "temperature": 0},
        "messages": [{"role": "user", "content": prompt}]}).json()
    return {"model": model, "num_ctx": num_ctx, "sent_chars": len(prompt),
            "prompt_eval_count": r.get("prompt_eval_count"),
            "answer": r.get("message", {}).get("content", "")[:40]}


@timed
def run() -> Result:
    meta = model_catalog()[SMALL]
    ctx = effective_context(meta)
    prompt = build_prompt()
    sent_chars = len(prompt)
    r = chat(SMALL, [{"role": "user", "content": prompt}], max_tokens=10)
    naive_answer = (r.choices[0].message.content or "").strip()
    try:
        preflight(prompt, ctx, max_tokens=10)
        refused = False
    except ContextOverflow:
        refused = True
    # The fix after refusing: send only what's relevant (here, a trivial filter stands in for
    # retrieval), then check usage.prompt_tokens matches what we sent.
    small = "\n".join(x for x in prompt.splitlines() if "vault" in x)
    est = preflight(small, ctx, max_tokens=10)
    r2 = chat(SMALL, [{"role": "user", "content": small}], max_tokens=10)
    fixed_answer = (r2.choices[0].message.content or "").strip()
    local = ollama_truncation() if os.environ.get("RUN_LOCAL") == "1" else None
    return Result(
        ID, TITLE, LAYER, KIND,
        reproduced="7291" not in naive_answer and r.usage.prompt_tokens < sent_chars / 4,
        fixed=refused and "7291" in fixed_answer,
        evidence=(
            f"card says {meta['context_length']} ctx, provider serves {ctx}; sent {sent_chars:,} "
            f"chars, billed {r.usage.prompt_tokens} prompt tokens, no error, answer "
            f"{naive_answer[:30]!r}; preflight refused={refused}; filtered prompt est {est} "
            f"tok -> {fixed_answer[:20]!r}"
            + (f"; Ollama (RUN_LOCAL): {local}" if local else "")
        ),
        data={"card_ctx": meta["context_length"], "provider_ctx": ctx,
              "sent_chars": sent_chars, "billed_prompt_tokens": r.usage.prompt_tokens,
              "ollama": local},
    )
