"""The LLM tool surface.

The schema tests are the load-bearing ones. The entire argument that this is
safe to hand a model is that routing and cost parameters never reach it.
"""

from __future__ import annotations

import json

import pytest

from searchroute.client import AsyncSearchRoute
from searchroute.tools import FORBIDDEN_PARAMS, TOOLS, Toolset
from searchroute.types import Capability, Depth

from .conftest import FakeProvider

FULL_DEPTH = frozenset({Depth.LINKS, Depth.SNIPPETS, Depth.SUMMARY, Depth.CONTENT})
SEARCH_ONLY = frozenset({Capability.SEARCH})


def toolset(providers, **kwargs) -> Toolset:
    client_kwargs = {"quota_store": "memory"}
    sr = AsyncSearchRoute(custom_providers=providers, **client_kwargs)
    return Toolset(sr, **kwargs)


class TestSchemasHideRouting:
    """The safety argument, asserted."""

    @pytest.mark.parametrize("fmt", ["anthropic", "openai", "plain"])
    def test_no_routing_or_cost_parameter_is_ever_exposed(self, fmt):
        sr = AsyncSearchRoute(custom_providers=[FakeProvider("p")], quota_store="memory")
        blob = json.dumps(Toolset(sr).schemas(fmt))

        for param in FORBIDDEN_PARAMS:
            assert f'"{param}"' not in blob, f"{param} leaked into the {fmt} schema"

    def test_schemas_are_well_formed_json_schema(self):
        sr = AsyncSearchRoute(custom_providers=[FakeProvider("p")], quota_store="memory")
        for tool in Toolset(sr).anthropic():
            schema = tool["input_schema"]
            assert schema["type"] == "object"
            assert schema["additionalProperties"] is False
            assert isinstance(schema["required"], list)
            assert set(schema["required"]) <= set(schema["properties"])
            assert tool["name"] and tool["description"]

    def test_anthropic_and_openai_carry_the_same_information(self):
        sr = AsyncSearchRoute(custom_providers=[FakeProvider("p")], quota_store="memory")
        tools = Toolset(sr)
        anthropic = {t["name"]: t for t in tools.anthropic()}
        openai = {t["function"]["name"]: t["function"] for t in tools.openai()}

        assert set(anthropic) == set(openai)
        for name in anthropic:
            assert anthropic[name]["description"] == openai[name]["description"]
            assert anthropic[name]["input_schema"] == openai[name]["parameters"]

    def test_every_tool_description_says_what_it_is_for(self):
        """Models select on description, so an empty or terse one is a bug."""
        for tool in TOOLS:
            assert len(tool.description) > 80, f"{tool.name} description is too thin"


class TestSelection:
    def test_allow_narrows_the_surface(self):
        tools = toolset([FakeProvider("p")], allow=["web_search", "read_page"])
        assert tools.names == ["web_search", "read_page"]
        assert len(tools.anthropic()) == 2

    def test_unknown_tool_name_is_rejected_at_construction(self):
        sr = AsyncSearchRoute(custom_providers=[FakeProvider("p")], quota_store="memory")
        with pytest.raises(ValueError, match="unknown tool"):
            Toolset(sr, allow=["web_search", "nope"])

    async def test_dispatching_a_disallowed_tool_is_an_error_not_a_crash(self):
        tools = toolset([FakeProvider("p")], allow=["web_search"])
        result = await tools.dispatch("read_page", {"url": "https://a.test"})
        assert result.is_error
        assert "Unknown tool" in result.text


class TestCostGuards:
    async def test_max_results_is_clamped_above_the_cap(self):
        provider = FakeProvider("p", results=50)
        tools = toolset([provider], max_results_cap=3)

        result = await tools.dispatch("web_search", {"query": "q", "max_results": 100})

        assert len(result.data["results"]) == 3

    async def test_garbage_max_results_falls_back_to_the_default(self):
        provider = FakeProvider("p", results=20)
        tools = toolset([provider], max_results_cap=10, default_results=4)

        result = await tools.dispatch("web_search", {"query": "q", "max_results": "lots"})

        assert len(result.data["results"]) == 4

    async def test_web_search_never_returns_page_bodies(self):
        """Otherwise the context is full after two calls."""
        provider = FakeProvider("p", depths=FULL_DEPTH, results=3, content_prefix="X" * 5000)
        tools = toolset([provider])

        result = await tools.dispatch("web_search", {"query": "q"})

        assert len(result.text) < 2000
        assert "XXXXX" not in result.text

    async def test_read_page_returns_full_untruncated_markdown(self):
        """The other half of the two-step design."""
        body = "PARAGRAPH " * 2000
        provider = FakeProvider("p", depths=FULL_DEPTH, content_prefix=body)
        tools = toolset([provider])

        result = await tools.dispatch("read_page", {"url": "https://a.test"})

        assert len(result.text) > 10_000
        assert "[truncated]" not in result.text

    async def test_read_page_can_be_capped_when_the_operator_wants(self):
        provider = FakeProvider("p", depths=FULL_DEPTH, content_prefix="Y" * 5000)
        tools = toolset([provider], read_page_chars=500)

        result = await tools.dispatch("read_page", {"url": "https://a.test"})

        assert "[truncated]" in result.text


class TestErrorsNeverEscape:
    async def test_no_provider_returns_readable_text(self):
        """A tool that raises kills the agent turn; one that explains lets the
        model tell the user what's wrong."""
        tools = toolset([FakeProvider("p", capabilities=SEARCH_ONLY)])

        result = await tools.dispatch("read_page", {"url": "https://a.test"})

        assert result.is_error
        assert "unavailable" in result.text.lower() or "could not read" in result.text.lower()

    async def test_provider_outage_is_reported_not_raised(self, fatal_error):
        broken = FakeProvider("broken", fail_with=fatal_error)
        tools = toolset([broken])

        result = await tools.dispatch("web_search", {"query": "q"})

        assert result.is_error
        assert isinstance(result.text, str) and result.text

    async def test_missing_query_is_handled(self):
        tools = toolset([FakeProvider("p")])
        result = await tools.dispatch("web_search", {})
        assert result.is_error and "No query" in result.text

    async def test_unreadable_page_suggests_moving_on(self):
        provider = FakeProvider(
            "p", depths=FULL_DEPTH, extract_fails={"https://blocked.test"}
        )
        tools = toolset([provider])

        result = await tools.dispatch("read_page", {"url": "https://blocked.test"})

        assert result.is_error
        assert "different source" in result.text


class TestRouting:
    async def test_each_tool_uses_its_own_capability(self):
        """The tool name is what selects the capability — the model never sees
        or chooses it."""
        from searchroute.tools.definitions import ROUTING

        assert ROUTING["academic_search"]["capability"] is Capability.ACADEMIC
        assert ROUTING["reference_lookup"]["capability"] is Capability.REFERENCE
        assert ROUTING["news_search"]["capability"] is Capability.NEWS
        assert ROUTING["web_search"]["capability"] is Capability.SEARCH

    async def test_web_search_requests_snippets_not_content(self):
        provider = FakeProvider("p", depths=FULL_DEPTH, results=2)
        tools = toolset([provider])

        await tools.dispatch("web_search", {"query": "q"})

        assert provider.search_calls[0].depth is Depth.SNIPPETS

    async def test_reference_lookup_asks_for_content(self):
        """Wikipedia serves it natively, so this is one call, not two."""
        provider = FakeProvider(
            "wiki",
            capabilities=frozenset({Capability.REFERENCE}),
            depths=FULL_DEPTH,
            results=1,
        )
        tools = toolset([provider])

        result = await tools.dispatch("reference_lookup", {"query": "photosynthesis"})

        assert provider.search_calls[0].depth is Depth.CONTENT
        assert provider.extract_calls == [], "must not need a second hop"
        assert "CONTENT" in result.text


class TestSyncClientWorks:
    async def test_dispatch_accepts_the_sync_client_too(self):
        from searchroute import SearchRoute

        sr = SearchRoute(custom_providers=[FakeProvider("p")], quota_store="memory")
        try:
            result = await Toolset(sr).dispatch("web_search", {"query": "q"})
            assert not result.is_error
            assert result.data["results"]
        finally:
            sr.close()


class TestAdapters:
    """Framework adapters, checked against the installed versions."""

    async def test_langchain_tools_are_built_and_callable(self):
        pytest.importorskip("langchain_core")
        from searchroute.tools.adapters import to_langchain

        tools = toolset([FakeProvider("p", results=2)], allow=["web_search", "read_page"])
        built = to_langchain(tools)

        assert [t.name for t in built] == ["web_search", "read_page"]
        assert set(built[0].args_schema.model_fields) == {"query", "max_results"}

        out = await built[0].ainvoke({"query": "test"})
        assert isinstance(out, str) and "p result 0" in out

    async def test_llamaindex_tools_are_built_and_callable(self):
        pytest.importorskip("llama_index.core")
        from searchroute.tools.adapters import to_llamaindex

        tools = toolset([FakeProvider("p", results=2)], allow=["web_search"])
        built = to_llamaindex(tools)

        assert built[0].metadata.name == "web_search"
        result = await built[0].acall(query="test")
        assert "p result 0" in str(result)

    async def test_adapter_errors_come_back_as_text_not_exceptions(self, fatal_error):
        """An adapter that raises aborts the whole agent run."""
        pytest.importorskip("langchain_core")
        from searchroute.tools.adapters import to_langchain

        tools = toolset([FakeProvider("broken", fail_with=fatal_error)])
        built = to_langchain(tools)

        out = await built[0].ainvoke({"query": "test"})
        assert isinstance(out, str) and "failed" in out.lower()

    async def test_mcp_server_exposes_every_tool_with_a_real_schema(self):
        pytest.importorskip("mcp")
        from searchroute.tools.mcp_server import build_server

        tools = toolset([FakeProvider("p", results=2)])
        server = build_server(tools)

        exposed = await server.list_tools()
        assert {t.name for t in exposed} == set(tools.names)
        # An empty schema would leave the client model guessing at arguments.
        by_name = {t.name: t for t in exposed}
        assert set(by_name["web_search"].input_schema["properties"]) == {
            "query",
            "max_results",
        }
        assert set(by_name["read_page"].input_schema["properties"]) == {"url"}

    async def test_mcp_schemas_also_hide_routing(self):
        pytest.importorskip("mcp")
        from searchroute.tools.mcp_server import build_server

        server = build_server(toolset([FakeProvider("p")]))
        blob = json.dumps([t.input_schema for t in await server.list_tools()])

        for param in FORBIDDEN_PARAMS:
            assert f'"{param}"' not in blob

    async def test_mcp_tool_execution_cannot_corrupt_the_stdio_stream(self):
        """On stdio transport stdout IS the JSON-RPC channel, and this server
        runs third-party libraries that may print. A stray print must land on
        stderr, not in the protocol."""
        pytest.importorskip("mcp")
        import io
        from contextlib import redirect_stdout

        from searchroute.tools.mcp_server import build_server

        class NoisyProvider(FakeProvider):
            async def search(self, query):
                print("a library wrote to stdout")  # noqa: T201
                return await super().search(query)

        server = build_server(toolset([NoisyProvider("noisy", results=1)]))

        captured = io.StringIO()
        with redirect_stdout(captured):
            await server.call_tool("web_search", {"query": "q"})

        assert captured.getvalue() == "", "tool output leaked into the JSON-RPC stream"
