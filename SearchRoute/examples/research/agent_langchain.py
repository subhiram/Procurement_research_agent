"""Research agent — LangChain.

Same brief, same tools, through ``searchroute.tools.adapters.to_langchain``.

Written against langchain-core 1.6.2 / langchain 1.x. LangChain's agent
constructors move between versions; if the import below fails, check what your
installed version exposes rather than assuming this line is right.

    pip install 'searchroute[examples]'
    export ANTHROPIC_API_KEY=...
    python examples/research/agent_langchain.py
"""

from __future__ import annotations

import asyncio

from _brief import MODEL, QUESTION, SYSTEM, preflight, print_report, trajectory_summary

from searchroute import AsyncSearchRoute
from searchroute.tools import Toolset
from searchroute.tools.adapters import to_langchain


async def run(question: str = QUESTION) -> str:
    from langchain.agents import create_agent
    from langchain_anthropic import ChatAnthropic

    async with AsyncSearchRoute(profile="quicklook") as sr:
        tools = to_langchain(Toolset(sr, max_results_cap=5))

        agent = create_agent(
            model=ChatAnthropic(model=MODEL, max_tokens=16000),
            tools=tools,
            system_prompt=SYSTEM,
        )

        result = await agent.ainvoke({"messages": [{"role": "user", "content": question}]})

    messages = result["messages"]
    tool_calls = [
        (call["name"], str(call["args"].get("query") or call["args"].get("url") or ""))
        for message in messages
        for call in (getattr(message, "tool_calls", None) or [])
    ]
    answer = messages[-1].content
    if isinstance(answer, list):  # content blocks
        answer = "\n".join(b.get("text", "") for b in answer if isinstance(b, dict))

    print_report(answer, tool_calls=tool_calls)
    print(f"\ntrajectory: {trajectory_summary(tool_calls)}")
    return answer


if __name__ == "__main__":
    preflight()
    asyncio.run(run())
