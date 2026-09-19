#!/usr/bin/env python3
"""Drop LLMRouter into a LangGraph agent.

    python examples/langgraph_agent.py

Needs at least one provider key (GROQ_API_KEY, MISTRAL_API_KEY, GOOGLE_API_KEY,
NVIDIA_API_KEY or OPENROUTER_API_KEY) and langgraph installed. The agent never
names a provider:
swapping which free tiers are in play is a config change, not a code change.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from llm_router import LLMRouter, configure, default_ledger

# Show the router's decisions, including any tier downgrade under `sticky`.
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

# Optional, but worth it: daily and weekly caps outlive a single process, so
# remember what has been spent across restarts.
configure(ledger_path=str(Path.home() / ".cache" / "llm_router" / "ledger.json"))


@tool
def word_count(text: str) -> int:
    """Count the words in a piece of text."""
    return len(text.split())


@tool
def reverse(text: str) -> str:
    """Reverse a piece of text."""
    return text[::-1]


def main() -> int:
    try:
        # LangGraph v1 moved the prebuilt agent here; fall back for older installs.
        from langchain.agents import create_agent
    except ImportError:
        try:
            from langgraph.prebuilt import create_react_agent as create_agent
        except ImportError:
            print("pip install langgraph")
            return 1

    # `sticky` with a session_id keeps a multi-step run on one endpoint, so the
    # agent does not change model between its own reasoning steps. Use
    # `free_first` when you only care that the call lands.
    router = LLMRouter(
        strategy="sticky",
        session_id="example-run",
        tier="A",              # quality floor; the ladder may hold or drop it
        model_kwargs={"temperature": 0.0},
        max_wait=10.0,         # tolerate a short rate-limit blip rather than fail
    )

    agent = create_agent(model=router, tools=[word_count, reverse])
    result = agent.invoke({
        "messages": [("user", "How many words are in 'the quick brown fox'? "
                              "Then reverse that same sentence.")]
    })

    print("\n--- conversation ---")
    for message in result["messages"]:
        kind = type(message).__name__
        content = str(message.content).strip()
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            content += f"  ->  calls {[c['name'] for c in tool_calls]}"
        print(f"{kind:>14}: {content[:160]}")

    print("\n--- how it was routed ---")
    for message in result["messages"]:
        routing = getattr(message, "response_metadata", {}).get("llm_router")
        if not routing:
            continue
        flag = "  <- TIER DOWNGRADE" if routing["tier_downgraded"] else ""
        print(f"  {routing['provider']}/{routing['model_id']} "
              f"[tier {routing['tier']}, step {routing['ladder_step']}] "
              f"{routing['tokens']} tokens{flag}")

    print("\n--- quota used ---")
    for key, bucket in sorted(default_ledger().snapshot().items()):
        counters = ", ".join(
            f"{name} {v['used']}/{v['limit']}"
            for name, v in sorted(bucket["counters"].items())
            if v["used"]
        )
        if counters:
            print(f"  {key}: {counters}")

    assert isinstance(result["messages"][-1], AIMessage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
