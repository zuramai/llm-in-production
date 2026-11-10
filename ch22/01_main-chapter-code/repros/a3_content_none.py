"""A3: message.content is None (tool call, refusal, truncation) and `.strip()` crashes."""

from __future__ import annotations

from typing import Any

from ch22_common import LUNA, Result, chat, timed

ID, TITLE, LAYER, KIND = "A3", "message.content is None", "API & tokens", "hosted"

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_order_status",
        "description": "Look up the status of an order by its numeric id.",
        "parameters": {
            "type": "object",
            "properties": {"order_id": {"type": "integer"}},
            "required": ["order_id"],
            "additionalProperties": False,
        },
    },
}]


def message_text(choice: Any) -> str:
    """Text to show a user for any choice, never None. Callers branch on tool calls first."""
    msg = choice.message
    if msg.content:
        return msg.content.strip()
    if getattr(msg, "refusal", None):
        return f"[refused] {msg.refusal}"
    if msg.tool_calls:
        names = ", ".join(tc.function.name for tc in msg.tool_calls)
        return f"[tool call: {names}]"
    return f"[no content: finish_reason={choice.finish_reason}]"


@timed
def run() -> Result:
    messages = [{"role": "user", "content": "Where is my order 4512?"}]
    r = chat(LUNA, messages, max_tokens=200, tools=TOOLS, reasoning_effort="none")
    choice = r.choices[0]
    try:
        choice.message.content.strip()
        crashed = None
    except AttributeError as e:
        crashed = str(e)
    text = message_text(choice)
    return Result(
        ID, TITLE, LAYER, KIND,
        reproduced=crashed is not None,
        fixed=isinstance(text, str) and text != "",
        evidence=f"finish_reason={choice.finish_reason}, content={choice.message.content!r}; "
                 f"naive: {crashed}; fixed: {text!r}",
    )
