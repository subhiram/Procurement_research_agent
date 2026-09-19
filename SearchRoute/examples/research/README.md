# Research agent — four integrations

The same agent, four ways. Each answers the same question with the same tools and
prints the same output, so reading them side by side shows you the integration
wiring and nothing else. The prompt, question and report format live in
[`_brief.py`](_brief.py).

## Which one do I want?

| File | Needs | Use when |
|---|---|---|
| [`agent_anthropic.py`](agent_anthropic.py) | `anthropic` | You own the agent loop, or want to see the raw schemas working. **Start here.** |
| [`agent_mcp.py`](agent_mcp.py) + [`mcp_config.json`](mcp_config.json) | `anthropic[mcp]`, `mcp` | Claude Code / Claude Desktop, or any MCP client. The config alone needs **zero code**. |
| [`agent_langchain.py`](agent_langchain.py) | `langchain`, `langchain-anthropic` | Your app is already on LangChain. |
| [`agent_llamaindex.py`](agent_llamaindex.py) | `llama-index-core`, `llama-index-llms-anthropic` | Your app is already on LlamaIndex. |

If you're choosing fresh: raw schemas for a loop you control, MCP for
desktop/editor clients, and the framework adapters only when you're already
committed to that framework.

## Running them

```bash
pip install 'searchroute[examples]'
export ANTHROPIC_API_KEY=...
# plus whatever search keys you have; with none it still works via DuckDuckGo
export TAVILY_API_KEY=...

python examples/research/agent_anthropic.py
```

**These cost money on two axes** — search credits and model tokens. Defaults are
deliberately small (`profile="quicklook"`, `max_results_cap=5`) and each script
prints a cost banner before it starts.

## Zero-code MCP

You don't need `agent_mcp.py` to use MCP. Paste the `mcpServers` block from
[`mcp_config.json`](mcp_config.json) into Claude Code or Claude Desktop and the
five tools appear. That is the whole integration for most people.

## What the output tells you

Each run ends with a trajectory line:

```
trajectory: searched, then read 2 page(s) — a real research trajectory
```

This is the thing worth checking. An agent that calls `web_search` once and then
answers from snippets *looks* fine but isn't doing research — the line says so
explicitly rather than letting a plausible answer hide it.

## Verification status

Honest accounting of what has and hasn't been checked:

- **Verified in CI** (`tests/test_examples.py`, no model calls, no spend): every
  file imports, each integration builds its five tools from a `Toolset`, all four
  use the shared brief, and the MCP config is well-formed.
- **Verified live**: the MCP server starts as a real subprocess and serves tools
  over stdio.
- **Not yet run against a live model**: the four agent loops themselves. They
  need an `ANTHROPIC_API_KEY`, which wasn't available when they were written.
  Treat the loop code as reviewed but unexercised, and expect the framework
  examples to be the likeliest to need a small fix — LangChain and LlamaIndex
  change their agent constructors between versions.

Versions these were written against: `anthropic` 1.4.0, `langchain` 1.4.0,
`langchain-anthropic` 1.7.1, `llama-index-core` 0.14.24, `mcp` 2.1.1.
