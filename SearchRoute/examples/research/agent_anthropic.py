"""Research agent — raw schemas, manual tool loop.

The primary example: it feeds ``Toolset.anthropic()` straight into the Messages
API, so it proves the exported schemas actually work end to end.

``client.beta.messages.tool_runner`` with ``@beta_tool`` would be less code, but
it builds schemas from Python function signatures — which would bypass the thing
this example exists to demonstrate. Use the runner in your own app if you prefer;
use this file to see the schemas working.

    pip install 'searchroute[examples]'
    export ANTHROPIC_API_KEY=...
    python examples/research/agent_anthropic.py
"""

from __future__ import annotations

import asyncio

import anthropic
from _brief import MODEL, QUESTION, SYSTEM, preflight, print_report, trajectory_summary

from searchroute import AsyncSearchRoute
from searchroute.tools import Toolset

MAX_TURNS = 12


async def run(question: str = QUESTION) -> str:
    client = anthropic.AsyncAnthropic()
    tool_calls: list[tuple[str, str]] = []

    async with AsyncSearchRoute(profile="quicklook") as sr:
        tools = Toolset(sr, max_results_cap=5)
        schemas = tools.anthropic()

        messages: list[dict] = [{"role": "user", "content": question}]

        for _ in range(MAX_TURNS):
            response = await client.messages.create(
                model=MODEL,
                max_tokens=16000,
                system=SYSTEM,
                thinking={"type": "adaptive"},
                tools=schemas,
                messages=messages,
            )

            if response.stop_reason == "end_turn":
                messages.append({"role": "assistant", "content": response.content})
                break

            messages.append({"role": "assistant", "content": response.content})

            # Every tool_use block in this turn runs, and every result goes back
            # in ONE user message. Splitting them across several messages trains
            # the model to stop calling tools in parallel — which shows up later
            # as an agent that feels inexplicably slow.
            blocks = [b for b in response.content if b.type == "tool_use"]
            results = await asyncio.gather(
                *(tools.dispatch(b.name, b.input) for b in blocks)
            )

            tool_results = []
            for block, result in zip(blocks, results, strict=True):
                detail = block.input.get("query") or block.input.get("url") or ""
                tool_calls.append((block.name, str(detail)))
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result.text,
                        # A failed tool is reported, never dropped — the model
                        # needs to know so it can try another source.
                        "is_error": result.is_error,
                    }
                )

            messages.append({"role": "user", "content": tool_results})

    answer = "\n".join(b.text for b in response.content if b.type == "text")
    print_report(answer, tool_calls=tool_calls)
    print(f"\ntrajectory: {trajectory_summary(tool_calls)}")
    return answer


if __name__ == "__main__":
    preflight()
    asyncio.run(run())
