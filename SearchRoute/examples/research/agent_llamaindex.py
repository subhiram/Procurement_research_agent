"""Research agent — LlamaIndex.

Same brief, same tools, through ``searchroute.tools.adapters.to_llamaindex``.

Written against llama-index-core 0.14.24 and llama-index-llms-anthropic 0.12.0.

    pip install 'searchroute[examples]' llama-index-llms-anthropic
    export ANTHROPIC_API_KEY=...
    python examples/research/agent_llamaindex.py
"""

from __future__ import annotations

import asyncio

from _brief import MODEL, QUESTION, SYSTEM, preflight, print_report, trajectory_summary

from searchroute import AsyncSearchRoute
from searchroute.tools import Toolset
from searchroute.tools.adapters import to_llamaindex


async def run(question: str = QUESTION) -> str:
    from llama_index.core.agent.workflow import FunctionAgent
    from llama_index.llms.anthropic import Anthropic

    async with AsyncSearchRoute(profile="quicklook") as sr:
        tools = to_llamaindex(Toolset(sr, max_results_cap=5))

        agent = FunctionAgent(
            tools=tools,
            llm=Anthropic(model=MODEL, max_tokens=16000),
            system_prompt=SYSTEM,
        )

        tool_calls: list[tuple[str, str]] = []
        handler = agent.run(question)

        # Stream events so the tool trajectory is visible, not just the answer.
        from llama_index.core.agent.workflow import ToolCall

        async for event in handler.stream_events():
            if isinstance(event, ToolCall):
                args = event.tool_kwargs or {}
                tool_calls.append(
                    (event.tool_name, str(args.get("query") or args.get("url") or ""))
                )

        response = await handler

    answer = str(response)
    print_report(answer, tool_calls=tool_calls)
    print(f"\ntrajectory: {trajectory_summary(tool_calls)}")
    return answer


if __name__ == "__main__":
    preflight()
    asyncio.run(run())
