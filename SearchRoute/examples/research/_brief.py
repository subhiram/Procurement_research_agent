"""The agent definition, shared by all four integrations.

Keeping the prompt, the question and the output format in one place is what
makes the four examples comparable: they differ only in integration wiring, so
reading them side by side shows you the wiring and nothing else.
"""

from __future__ import annotations

import os
import sys

MODEL = "claude-opus-5"

SYSTEM = """You are a research assistant with web search tools.

Method:
1. Start with web_search to find candidate sources. Snippets only — cheap.
2. Read the few pages that actually matter with read_page. Two or three is
   usually enough; do not read everything you find.
3. Use academic_search for scholarly questions and reference_lookup for
   established background, when the question calls for them.
4. Synthesize an answer grounded in what you read.

Rules:
- Cite every claim with the URL you got it from.
- If sources disagree, say so and give both positions.
- If you could not find or read something, say that plainly. Never fill a gap
  with what you assume to be true — an honest "the sources don't cover this" is
  more useful than a confident guess.
- Prefer reading one good source over skimming ten snippets."""

QUESTION = (
    "What are the main approaches to chunking documents for retrieval-augmented "
    "generation, and what does recent evidence say about which works best?"
)

COST_BANNER = """\
This example spends real money on two axes:
  - search credits (your configured providers)
  - model tokens (Anthropic API)
Defaults are kept small. Ctrl-C now if that's not what you want.
"""


def preflight(*, needs_anthropic: bool = True) -> None:
    """Print the cost banner and check for an API key before spending anything."""
    print(COST_BANNER)
    if needs_anthropic and not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set. Export it and re-run.")
        sys.exit(1)


def print_report(answer: str, *, tool_calls: list[tuple[str, str]] | None = None) -> None:
    """Same output shape across all four examples."""
    if tool_calls:
        print("\n--- tool calls ---")
        for name, detail in tool_calls:
            print(f"  {name:<18} {detail[:70]}")

    print("\n--- answer ---\n")
    print(answer)


def trajectory_summary(tool_calls: list[tuple[str, str]]) -> str:
    """Did the agent actually search *and* read, or just guess off snippets?

    This is the thing worth checking about an example like this: a run that only
    calls web_search once and then answers looks fine but is not doing research.
    """
    names = [n for n, _ in tool_calls]
    searched = any(n.endswith("_search") or n == "reference_lookup" for n in names)
    read = names.count("read_page")
    if searched and read:
        return f"searched, then read {read} page(s) — a real research trajectory"
    if searched:
        return "searched but never read a page — answered from snippets alone"
    return "no tools called"
