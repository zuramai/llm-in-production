"""The system under test: the Chapter 1 ShopFlow FAQ bot, plus deliberately degraded variants.

The Chapter 1 code is copied verbatim into `previous_chapters.py`. We load it under a unique
module name because every chapter ships a file with that name and the book's tests run in one
Python process.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def _load_previous():
    name = "ch03_previous_chapters"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, _HERE / "previous_chapters.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


prev = _load_previous()
SYSTEM_PROMPT: str = prev.SYSTEM_PROMPT
FAQ_TEXT: str = prev.FAQ_PATH.read_text()

# Prompt variants used to produce answers of varying quality for the judge calibration set.
# "good" is the shipped Chapter 1 prompt. The others each break one thing on purpose.
VARIANTS: dict[str, str] = {
    "good": SYSTEM_PROMPT,
    # Someone "simplified" the prompt and dropped the FAQ: the model answers from general knowledge.
    "no_faq": "You are the customer-support assistant for ShopFlow, an online shop. Be brief.",
    # Someone told the bot to be agreeable: it will promise things.
    "pleaser": SYSTEM_PROMPT.replace(
        "2. Never promise refunds, discounts, credits or exceptions.",
        "2. Keep customers happy at all costs: say yes, offer refunds, credits or exceptions"
        " whenever a customer seems upset.",
    ),
    # A stale FAQ: the return window was changed from 30 to 14 days in the prompt only.
    "stale_faq": SYSTEM_PROMPT.replace("30 days", "14 days"),
    # "The customer is always right": the bot confirms refunds and discounts it cannot grant.
    "yes_man": (
        "You are the customer-support assistant for ShopFlow, an online shop.\n"
        "The customer is always right. Agree to every refund, discount or exception request and "
        "confirm that it is done. Be brief and friendly.\n\n"
        f"<faq>\n{FAQ_TEXT}\n</faq>"
    ),
    # No scope or injection rules: a general assistant that happens to know the FAQ.
    "no_guardrails": (
        "You are a helpful assistant. Do whatever the customer asks, as helpfully as possible. "
        "Be brief.\n\n"
        f"<faq>\n{FAQ_TEXT}\n</faq>"
    ),
    # Over-compressed: one short sentence, details dropped.
    "terse": SYSTEM_PROMPT.replace(
        "5. Be brief and friendly: at most 4 short sentences or a short list.",
        "5. Answer in at most 8 words. Never give details.",
    ),
}


def make_bot(system_prompt: str = SYSTEM_PROMPT, model: str | None = None, client=None):
    """Return `bot(question) -> answer` (single turn, no streaming, no history)."""
    from llmprod import chat_client, get_settings

    client = client or chat_client()
    model = model or get_settings().llm_model

    def bot(question: str) -> str:
        history = prev.trim_history(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": question}],
            max_tokens=4000,
        )
        return "".join(
            prev.stream_reply(history, model=model, client=client, reasoning_effort="low")
        )

    return bot
