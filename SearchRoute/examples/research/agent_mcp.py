"""Research agent — over MCP.

MCP is consumed two ways, and both are shown here:

**Zero code.** Paste ``mcp_config.json`` into Claude Code or Claude Desktop and
the five tools appear. No Python in your app at all. That is the point of MCP,
and for most people it is the whole integration.

**Programmatically**, below: spawn the server over stdio, convert its tools with
the Anthropic SDK's MCP helpers, and let the tool runner drive the loop.

    pip install 'searchroute[examples]' 'anthropic[mcp]'
    export ANTHROPIC_API_KEY=...
    python examples/research/agent_mcp.py

Note this example uses ``client.beta.messages.tool_runner`` rather than the
manual loop in ``agent_anthropic.py`` — the MCP helpers are built for it, and
here the schemas come from the MCP server rather than from our exporter, so
there is nothing to demonstrate by hand-rolling the loop.
"""

from __future__ import annotations

import asyncio
import sys

from _brief import MODEL, QUESTION, SYSTEM, preflight, print_report, trajectory_summary


async def run(question: str = QUESTION) -> str:
    from anthropic import AsyncAnthropic
    from anthropic.lib.tools.mcp import async_mcp_tool
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    client = AsyncAnthropic()
    tool_calls: list[tuple[str, str]] = []

    # Run the server the same way an MCP client would: as a subprocess over
    # stdio, using this interpreter so it inherits the current environment.
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "searchroute.tools.mcp_server"],
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            listed = await session.list_tools()
            print(f"MCP server exposed: {', '.join(t.name for t in listed.tools)}\n")

            runner = client.beta.messages.tool_runner(
                model=MODEL,
                max_tokens=16000,
                system=SYSTEM,
                tools=[async_mcp_tool(t, session) for t in listed.tools],
                messages=[{"role": "user", "content": question}],
            )

            final = None
            async for message in runner:
                final = message
                for block in message.content:
                    if block.type == "tool_use":
                        detail = block.input.get("query") or block.input.get("url") or ""
                        tool_calls.append((block.name, str(detail)))

    answer = "\n".join(b.text for b in final.content if b.type == "text")
    print_report(answer, tool_calls=tool_calls)
    print(f"\ntrajectory: {trajectory_summary(tool_calls)}")
    return answer


if __name__ == "__main__":
    preflight()
    asyncio.run(run())
