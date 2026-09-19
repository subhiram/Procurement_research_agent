"""The four research agents, checked without spending money.

These examples cost real credits and model tokens to run, so CI checks the parts
that can break for free: that each file imports, that its integration builds
tools from a Toolset, and that the shared brief is coherent. Actually running
them against a model is a manual step, documented in examples/research/README.md.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from searchroute.client import AsyncSearchRoute
from searchroute.tools import Toolset

from .conftest import FakeProvider

EXAMPLES = Path(__file__).resolve().parent.parent / "examples" / "research"
AGENTS = ["agent_anthropic", "agent_langchain", "agent_llamaindex", "agent_mcp"]


@pytest.fixture(autouse=True)
def _importable():
    """The agents import `_brief` as a sibling, the way they run standalone."""
    sys.path.insert(0, str(EXAMPLES))
    yield
    sys.path.remove(str(EXAMPLES))


def toolset(**kwargs) -> Toolset:
    sr = AsyncSearchRoute(custom_providers=[FakeProvider("p", results=3)], quota_store="memory")
    return Toolset(sr, **kwargs)


class TestSharedBrief:
    def test_brief_imports_and_is_complete(self):
        import _brief

        assert _brief.MODEL == "claude-opus-5"
        assert len(_brief.SYSTEM) > 200
        assert _brief.QUESTION.endswith("?")

    def test_trajectory_summary_distinguishes_real_research(self):
        """The check that matters: did it read pages, or just skim snippets?"""
        import _brief

        assert "no tools" in _brief.trajectory_summary([])
        assert "snippets alone" in _brief.trajectory_summary([("web_search", "q")])
        real = [("web_search", "q"), ("read_page", "https://a.test")]
        assert "real research trajectory" in _brief.trajectory_summary(real)

    def test_every_agent_uses_the_shared_brief(self):
        """If one drifts to its own prompt the comparison stops being honest."""
        for name in AGENTS:
            source = (EXAMPLES / f"{name}.py").read_text()
            assert "from _brief import" in source, f"{name} does not use the shared brief"
            assert "SYSTEM" in source and "QUESTION" in source


class TestAgentsAreWellFormed:
    @pytest.mark.parametrize("name", AGENTS)
    def test_module_imports(self, name):
        """Catches a broken import or a renamed framework API for free."""
        pytest.importorskip("anthropic")
        if name == "agent_langchain":
            pytest.importorskip("langchain")
        if name == "agent_llamaindex":
            pytest.importorskip("llama_index.core")
        if name == "agent_mcp":
            pytest.importorskip("mcp")

        module = __import__(name)
        assert callable(module.run)

    @pytest.mark.parametrize("name", AGENTS)
    def test_declares_its_cost(self, name):
        source = (EXAMPLES / f"{name}.py").read_text()
        assert "preflight()" in source, f"{name} must warn before spending"

    def test_anthropic_agent_uses_our_exported_schemas(self):
        """The reason that example uses a manual loop instead of tool_runner."""
        source = (EXAMPLES / "agent_anthropic.py").read_text()
        assert "tools.anthropic()" in source
        assert "tool_result" in source
        assert "is_error" in source, "a failed tool must be reported, not dropped"

    def test_anthropic_agent_batches_parallel_results(self):
        """All tool_results must go back in ONE user message."""
        source = (EXAMPLES / "agent_anthropic.py").read_text()
        assert 'messages.append({"role": "user", "content": tool_results})' in source


class TestIntegrationsBuildTools:
    """Each integration can construct its tools from a Toolset — the part that
    breaks when a framework changes, checked without any model call."""

    def test_anthropic_schemas(self):
        tools = toolset(max_results_cap=5)
        schemas = tools.anthropic()
        assert len(schemas) == 5
        assert all({"name", "description", "input_schema"} <= set(s) for s in schemas)

    def test_langchain_tools(self):
        pytest.importorskip("langchain_core")
        from searchroute.tools.adapters import to_langchain

        built = to_langchain(toolset())
        assert len(built) == 5 and all(t.name and t.description for t in built)

    def test_llamaindex_tools(self):
        pytest.importorskip("llama_index.core")
        from searchroute.tools.adapters import to_llamaindex

        built = to_llamaindex(toolset())
        assert len(built) == 5

    async def test_mcp_server_starts_and_lists_tools(self):
        pytest.importorskip("mcp")
        from searchroute.tools.mcp_server import build_server

        server = build_server(toolset())
        assert len(await server.list_tools()) == 5


class TestMCPConfig:
    def test_config_is_valid_json_with_the_right_shape(self):
        raw = json.loads((EXAMPLES / "mcp_config.json").read_text())
        entry = raw["mcpServers"]["searchroute"]
        assert entry["args"] == ["-m", "searchroute.tools.mcp_server"]

    def test_config_documents_operator_side_cost_controls(self):
        raw = json.loads((EXAMPLES / "mcp_config.json").read_text())
        env = raw["mcpServers"]["searchroute"]["env"]
        assert "SEARCHROUTE_MAX_RESULTS" in env
        assert "SEARCHROUTE_PROFILE" in env
