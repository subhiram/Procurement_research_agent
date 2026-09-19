"""An MCP server exposing SearchRoute's tools.

Run it directly and any MCP client — Claude Code, Claude Desktop, or your own —
gets web_search, read_page, academic_search, reference_lookup and news_search
with no glue code:

    python -m searchroute.tools.mcp_server

Configuration comes from the environment, exactly like the library itself: set
whichever provider keys you have and the server routes across them. Routing
knobs stay operator-side and out of the tool schemas, so a client model gets
intent parameters only — the same contract as every other integration.

Requires ``pip install 'searchroute[mcp]'`` (mcp >= 2.0).

Written against ``mcp`` 2.1.1. Note that 2.x renamed ``FastMCP`` to
``MCPServer``; on ``mcp<2`` this import will fail with a clear message.
"""

from __future__ import annotations

import os
import sys
from contextlib import redirect_stdout

from ..client import AsyncSearchRoute
from . import Toolset


def build_server(toolset: Toolset | None = None, name: str = "searchroute"):
    """Create the MCP server. Separated from ``main`` so it can be tested and
    embedded in a larger server without spawning a process."""
    try:
        from mcp.server.mcpserver import MCPServer
    except ImportError as exc:  # pragma: no cover - depends on extras
        raise ImportError(
            "MCP is required: pip install 'searchroute[mcp]'. "
            "Note that searchroute targets mcp>=2.0, where FastMCP was renamed "
            "to MCPServer."
        ) from exc

    if toolset is None:
        profile = os.environ.get("SEARCHROUTE_PROFILE") or "quicklook"
        cap = int(os.environ.get("SEARCHROUTE_MAX_RESULTS", "5"))
        toolset = Toolset(AsyncSearchRoute(profile=profile), max_results_cap=cap)

    server = MCPServer(
        name=name,
        instructions=(
            "Web search and page reading. Start with web_search (or "
            "academic_search / reference_lookup / news_search when the question "
            "calls for one of those), then call read_page on any result whose "
            "full text you actually need."
        ),
    )

    # One closure per tool so each keeps its own name, description and schema.
    for tool in toolset.tools:
        server.add_tool(
            _make_handler(toolset, tool),
            name=tool.name,
            description=tool.description,
        )

    return server


async def _dispatch_quietly(toolset: Toolset, name: str, arguments: dict) -> str:
    """Run a tool with stdout redirected to stderr.

    On stdio transport, stdout *is* the JSON-RPC channel. This server runs
    third-party libraries in-process — ddgs, trafilatura — and a single stray
    ``print()`` from any of them corrupts the stream and breaks the session with
    a parse error that points nowhere near the real cause.

    The redirect covers only tool execution, so the transport's own writes
    (which happen after the handler returns) still reach the real stdout.
    """
    with redirect_stdout(sys.stderr):
        result = await toolset.dispatch(name, arguments)
    return result.text


def _make_handler(toolset: Toolset, tool):
    """Build a typed coroutine MCP can introspect.

    MCP derives a tool's schema from the function signature, so the parameters
    are declared explicitly per tool rather than through ``**kwargs`` — which
    would produce an empty schema and leave the client model guessing.
    """
    if tool.name == "read_page":

        async def handler(url: str) -> str:
            return await _dispatch_quietly(toolset, "read_page", {"url": url})

    elif "max_results" in tool.schema.get("properties", {}):

        async def handler(query: str, max_results: int = 5) -> str:
            return await _dispatch_quietly(
                toolset, tool.name, {"query": query, "max_results": max_results}
            )

    else:

        async def handler(query: str) -> str:
            return await _dispatch_quietly(toolset, tool.name, {"query": query})

    handler.__name__ = tool.name
    handler.__doc__ = tool.description
    return handler


def main() -> None:
    build_server().run(transport="stdio")


if __name__ == "__main__":
    main()
