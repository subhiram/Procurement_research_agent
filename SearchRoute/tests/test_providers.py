"""Per-provider tests against mocked HTTP.

One fixture per provider asserting that its real payload shape normalizes into
``SearchResult`` correctly, plus the error mapping that drives fallback.
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from searchroute.errors import AuthError, QuotaExceeded, RateLimited, TransientError
from searchroute.providers.exa import ExaProvider
from searchroute.providers.tavily import TavilyProvider
from searchroute.types import Capability, ContentStatus, Depth, SearchQuery

TAVILY_RESPONSE = {
    "query": "test",
    "answer": "A short answer.",
    "results": [
        {
            "title": "First result",
            "url": "https://example.com/one",
            "content": "A snippet of the first page.",
            "raw_content": "# First\n\nThe full markdown body.",
            "score": 0.92,
            "published_date": "2026-08-01T00:00:00Z",
        },
        {
            "title": "Second result",
            "url": "https://example.com/two",
            "content": "A snippet of the second page.",
            "raw_content": None,
            "score": 0.71,
        },
    ],
}

EXA_RESPONSE = {
    "results": [
        {
            "title": "Neural result",
            "url": "https://example.com/neural",
            "text": "The full page text.",
            "summary": "A generated summary.",
            "highlights": ["a highlight"],
            "score": 0.88,
            "publishedDate": "2026-07-15T00:00:00.000Z",
        }
    ]
}


class TestTavily:
    @respx.mock
    async def test_normalizes_search_results(self):
        respx.post("https://api.tavily.com/search").mock(
            return_value=httpx.Response(200, json=TAVILY_RESPONSE)
        )
        provider = TavilyProvider(api_key="k")

        response = await provider.search(SearchQuery(query="test"))

        assert len(response.results) == 2
        first = response.results[0]
        assert first.title == "First result"
        assert first.url == "https://example.com/one"
        assert first.snippet == "A snippet of the first page."
        assert first.score == 0.92
        assert first.provider == "tavily"
        assert first.published_date is not None
        assert first.published_date.year == 2026
        # Snippets depth was requested, so content is deliberately not surfaced.
        assert first.content is None
        assert first.content_status is ContentStatus.NOT_REQUESTED
        await provider.aclose()

    @respx.mock
    async def test_content_depth_requests_raw_content(self):
        route = respx.post("https://api.tavily.com/search").mock(
            return_value=httpx.Response(200, json=TAVILY_RESPONSE)
        )
        provider = TavilyProvider(api_key="k")

        response = await provider.search(SearchQuery(query="test", depth=Depth.CONTENT))

        sent = json.loads(route.calls[0].request.content)
        assert sent["include_raw_content"] == "markdown"
        assert sent["search_depth"] == "advanced"

        assert response.results[0].content == "# First\n\nThe full markdown body."
        assert response.results[0].content_status is ContentStatus.NATIVE
        # The second result had no raw_content: that is a failure, not silence.
        assert response.results[1].content is None
        assert response.results[1].content_status is ContentStatus.FAILED
        await provider.aclose()

    @respx.mock
    async def test_answer_mode(self):
        respx.post("https://api.tavily.com/search").mock(
            return_value=httpx.Response(200, json=TAVILY_RESPONSE)
        )
        provider = TavilyProvider(api_key="k")

        answer = await provider.answer(SearchQuery(query="test", capability=Capability.ANSWER))

        assert answer.answer == "A short answer."
        assert len(answer.results) == 2
        await provider.aclose()

    @respx.mock
    async def test_extract_preserves_order_and_reports_failures(self):
        respx.post("https://api.tavily.com/extract").mock(
            return_value=httpx.Response(
                200,
                json={
                    "results": [
                        {"url": "https://example.com/b", "raw_content": "B body"},
                    ],
                    "failed_results": [
                        {"url": "https://example.com/a", "error": "403 forbidden"},
                    ],
                },
            )
        )
        provider = TavilyProvider(api_key="k")

        docs = await provider.extract(["https://example.com/a", "https://example.com/b"])

        # Caller order is preserved even though the API grouped them differently.
        assert [d.url for d in docs] == ["https://example.com/a", "https://example.com/b"]
        assert docs[0].ok is False and docs[0].error == "403 forbidden"
        assert docs[1].ok is True and docs[1].content == "B body"
        await provider.aclose()

    def test_content_depth_costs_more(self):
        provider = TavilyProvider(api_key="k")
        assert provider.cost_of(SearchQuery(query="q", depth=Depth.SNIPPETS)) == 1
        assert provider.cost_of(SearchQuery(query="q", depth=Depth.CONTENT)) == 2

    def test_declares_no_native_summary(self):
        """Tavily answers a query; it does not summarize each result."""
        assert Depth.SUMMARY not in TavilyProvider.native_depths


class TestExa:
    @respx.mock
    async def test_normalizes_search_results(self):
        respx.post("https://api.exa.ai/search").mock(
            return_value=httpx.Response(200, json=EXA_RESPONSE)
        )
        provider = ExaProvider(api_key="k")

        response = await provider.search(SearchQuery(query="test", depth=Depth.CONTENT))

        hit = response.results[0]
        assert hit.title == "Neural result"
        assert hit.content == "The full page text."
        assert hit.summary == "A generated summary."
        assert hit.content_status is ContentStatus.NATIVE
        assert hit.published_date.year == 2026
        await provider.aclose()

    @respx.mock
    async def test_depth_controls_contents_payload(self):
        route = respx.post("https://api.exa.ai/search").mock(
            return_value=httpx.Response(200, json=EXA_RESPONSE)
        )
        provider = ExaProvider(api_key="k")

        await provider.search(SearchQuery(query="q", depth=Depth.SNIPPETS))
        shallow = json.loads(route.calls[0].request.content)
        assert "text" not in shallow["contents"], "snippets depth must not pay for full text"

        await provider.search(SearchQuery(query="q", depth=Depth.CONTENT))
        deep = json.loads(route.calls[1].request.content)
        assert deep["contents"]["text"] is True
        await provider.aclose()

    @respx.mock
    async def test_academic_capability_sets_category(self):
        route = respx.post("https://api.exa.ai/search").mock(
            return_value=httpx.Response(200, json=EXA_RESPONSE)
        )
        provider = ExaProvider(api_key="k")

        await provider.search(SearchQuery(query="q", capability=Capability.ACADEMIC))

        sent = json.loads(route.calls[0].request.content)
        assert sent["category"] == "research paper"
        await provider.aclose()

    def test_serves_summary_natively(self):
        """Exa is the reason SUMMARY depth exists at all."""
        assert Depth.SUMMARY in ExaProvider.native_depths

    def test_cost_scales_with_depth(self):
        provider = ExaProvider(api_key="k")
        cheap = provider.cost_of(SearchQuery(query="q", depth=Depth.SNIPPETS, max_results=10))
        rich = provider.cost_of(SearchQuery(query="q", depth=Depth.CONTENT, max_results=10))
        assert rich > cheap


class TestErrorMapping:
    """The status-to-error mapping is what makes fallback behave correctly."""

    @pytest.mark.parametrize(
        "status,expected",
        [
            (401, AuthError),
            (403, AuthError),
            (402, QuotaExceeded),
            (429, RateLimited),
            (500, TransientError),
            (503, TransientError),
        ],
    )
    @respx.mock
    async def test_status_codes_map_to_typed_errors(self, status, expected):
        respx.post("https://api.tavily.com/search").mock(
            return_value=httpx.Response(status, json={"error": "nope"})
        )
        provider = TavilyProvider(api_key="k")

        with pytest.raises(expected):
            await provider.search(SearchQuery(query="q"))
        await provider.aclose()

    @respx.mock
    async def test_retry_after_header_is_captured(self):
        respx.post("https://api.tavily.com/search").mock(
            return_value=httpx.Response(429, headers={"retry-after": "7"}, json={})
        )
        provider = TavilyProvider(api_key="k")

        with pytest.raises(RateLimited) as exc:
            await provider.search(SearchQuery(query="q"))

        assert exc.value.retry_after == 7.0
        await provider.aclose()

    @respx.mock
    async def test_timeout_becomes_transient(self):
        respx.post("https://api.tavily.com/search").mock(
            side_effect=httpx.ConnectTimeout("too slow")
        )
        provider = TavilyProvider(api_key="k")

        with pytest.raises(TransientError):
            await provider.search(SearchQuery(query="q"))
        await provider.aclose()


class TestConfiguration:
    def test_keyed_provider_without_key_is_not_configured(self):
        assert TavilyProvider(api_key=None).configured is False
        assert TavilyProvider(api_key="k").configured is True

    def test_keyless_provider_is_always_configured(self):
        from searchroute.providers.duckduckgo import DuckDuckGoProvider

        assert DuckDuckGoProvider().configured is True

    def test_duckduckgo_is_marked_last_resort(self):
        """It is a safety net, not a tier — it must never be preferred."""
        from searchroute.providers.duckduckgo import DuckDuckGoProvider

        assert DuckDuckGoProvider.last_resort is True

    def test_capability_guard(self):
        from searchroute.errors import CapabilityNotSupported
        from searchroute.providers.duckduckgo import DuckDuckGoProvider

        assert not DuckDuckGoProvider().supports(Capability.EXTRACT)
        with pytest.raises(CapabilityNotSupported):
            import asyncio

            asyncio.run(DuckDuckGoProvider().extract(["https://example.com"]))
