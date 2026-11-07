"""Scripted stand-in for the OpenAI client, so agent-loop tests need no network."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any


def tool_call(call_id: str, name: str, arguments: str = "{}") -> SimpleNamespace:
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=arguments))


def response(
    *, content: str | None = None, tool_calls: list | None = None, tokens: int = 100
) -> SimpleNamespace:
    usage = SimpleNamespace(
        prompt_tokens=tokens - 10, completion_tokens=10, total_tokens=tokens, model_extra={}
    )
    message = SimpleNamespace(content=content, tool_calls=tool_calls or None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


class FakeClient:
    """`client.chat.completions.create(...)` replays `script`; repeats the last item forever."""

    def __init__(self, script: list[Any]) -> None:
        self.script = script
        self.calls: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> Any:
        # Snapshot the messages: the agent keeps appending to the same list.
        self.calls.append({**kwargs, "messages": [dict(m) for m in kwargs["messages"]]})
        item = self.script[min(len(self.calls) - 1, len(self.script) - 1)]
        if isinstance(item, Exception):
            raise item
        return item
