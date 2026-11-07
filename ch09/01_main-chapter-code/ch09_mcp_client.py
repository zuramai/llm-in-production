"""Use an MCP server's tools from the raw OpenAI-SDK loop.

`McpToolbox` has the same shape as `LocalToolbox` (`.specs`, `.call`), so `run_agent` does not
know or care which one it got. It converts MCP tool definitions to OpenAI function definitions
and forwards calls over an MCP client session (stdio subprocess or streamable HTTP URL).

The agent loop is synchronous and the MCP client is async, so the session lives on its own
event loop in a background thread and `call` hops onto it.
"""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from typing import Any, Self

from ch09_tools import MAX_RESULT_CHARS, _strip_titles, error_json
from mcp.client.client import Client
from mcp.client.stdio import StdioServerParameters
from mcp.shared.exceptions import MCPError

HERE = Path(__file__).resolve().parent


def mcp_tool_to_openai(tool: Any) -> dict[str, Any]:
    """MCP `Tool` -> OpenAI Chat Completions tool definition."""
    schema = _strip_titles(dict(tool.input_schema))
    schema["additionalProperties"] = False
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": " ".join((tool.description or "").split()),
            "parameters": schema,
        },
    }


def stdio_server() -> StdioServerParameters:
    """Spawn our own server as a subprocess, the way Claude Desktop or an IDE would."""
    return StdioServerParameters(
        command="uv",
        args=["run", "python", "ch09_mcp_server.py", "--transport", "stdio"],
        cwd=HERE,
    )


class McpToolbox:
    def __init__(self, server: StdioServerParameters | str, *, startup_timeout: float = 60) -> None:
        self._server = server
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self._client: Client | None = None
        self._stop: asyncio.Event | None = None
        self._thread = threading.Thread(target=self._run, daemon=True, name="mcp-client")
        self._thread.start()
        if not self._ready.wait(startup_timeout) or self._error is not None:
            raise RuntimeError(f"could not connect to MCP server: {self._error!r}")
        self.specs = self._submit(self._list_specs())
        self.protocol_version = str(self._client.protocol_version)  # type: ignore[union-attr]

    # ---- background loop ----
    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._serve())

    async def _serve(self) -> None:
        self._stop = asyncio.Event()
        try:
            async with Client(self._server) as client:
                self._client = client
                self._ready.set()
                await self._stop.wait()
        except BaseException as exc:  # noqa: BLE001
            self._error = exc
            self._ready.set()

    def _submit(self, coro: Any, timeout: float = 60) -> Any:
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout)

    async def _list_specs(self) -> list[dict[str, Any]]:
        result = await self._client.list_tools()  # type: ignore[union-attr]
        return [mcp_tool_to_openai(t) for t in result.tools]

    async def _call(self, name: str, arguments: dict[str, Any]) -> str:
        result = await self._client.call_tool(name, arguments)  # type: ignore[union-attr]
        text = "\n".join(getattr(c, "text", "") for c in result.content)
        if result.is_error:
            return error_json(text or "tool failed")
        if result.structured_content is not None:
            text = json.dumps(result.structured_content)
        return text[:MAX_RESULT_CHARS]

    # ---- Toolbox protocol ----
    def call(self, name: str, raw_arguments: str) -> str:
        try:
            args = json.loads(raw_arguments or "{}")
        except json.JSONDecodeError as exc:
            return error_json(f"arguments are not valid JSON: {exc.msg} (position {exc.pos})")
        if not isinstance(args, dict):
            return error_json("arguments must be a JSON object")
        try:
            return self._submit(self._call(name, args))
        except MCPError as exc:
            return error_json(f"MCP error: {exc}")
        except Exception as exc:  # noqa: BLE001
            return error_json(f"{name} failed unexpectedly ({type(exc).__name__})")

    def close(self) -> None:
        if self._stop is not None:
            self._loop.call_soon_threadsafe(self._stop.set)
        self._thread.join(timeout=10)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
