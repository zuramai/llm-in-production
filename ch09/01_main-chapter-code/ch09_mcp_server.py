"""The same four tools as an MCP server (mcp 2.x `MCPServer`, formerly `FastMCP`).

    uv run python ch09_mcp_server.py                          # stdio (spawned by a client)
    uv run python ch09_mcp_server.py --transport http --port 8009   # streamable HTTP at /mcp

Nothing here re-implements a tool: it registers the functions from ch09_tools.py. The only
translation is error handling: our `ToolError` becomes the SDK's `ToolError`, which the SDK
turns into a normal tool result with `is_error=True` (the model reads it and recovers).
"""

from __future__ import annotations

import argparse
import functools
from collections.abc import Callable
from typing import Any

import ch09_setup
import ch09_tools
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError as McpToolError
from mcp.types import ToolAnnotations

INSTRUCTIONS = (
    "ShopFlow support tools. create_refund only proposes a refund; a human approves it. "
    "Amounts are integer cents."
)


def _wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(fn)  # keeps the signature, so the SDK derives the same JSON schema
    def inner(*args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except ch09_tools.ToolError as exc:
            raise McpToolError(str(exc)) from exc

    return inner


def build_server() -> MCPServer:
    server = MCPServer("shopflow-support", instructions=INSTRUCTIONS, version="0.1.0")
    for name, fn in ch09_tools.TOOL_FUNCTIONS.items():
        server.add_tool(
            _wrap(fn), name=name, annotations=ToolAnnotations(**ch09_tools.TOOL_HINTS[name])
        )
    return server


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8009)
    args = parser.parse_args()
    ch09_setup.setup_tables()
    if args.transport == "stdio":
        build_server().run(transport="stdio")
    else:
        build_server().run(transport="streamable-http", host=args.host, port=args.port)
