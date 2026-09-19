"""Keyless local extraction — the tail that makes CONTENT depth work with no keys."""

from __future__ import annotations

import httpx
import respx

from searchroute.client import AsyncSearchRoute
from searchroute.providers.http_extract import HTTPExtractProvider
from searchroute.types import Capability, ContentStatus, Depth

from .conftest import FakeProvider

ARTICLE = """
<html><head><title>A Real Article</title></head>
<body>
  <nav>home about contact</nav>
  <article>
    <h1>A Real Article</h1>
    <p>This is the first substantial paragraph of the article body, long enough
       that a readability extractor will treat it as the main content rather
       than as navigation chrome or boilerplate.</p>
    <p>A second paragraph continues the argument with more detail, again with
       enough words to clear the extractor's density threshold comfortably.</p>
  </article>
  <footer>copyright</footer>
</body></html>
"""


class TestHTTPExtract:
    @respx.mock
    async def test_extracts_body_and_drops_chrome(self):
        respx.get("https://example.com/article").mock(
            return_value=httpx.Response(
                200, html=ARTICLE, headers={"content-type": "text/html; charset=utf-8"}
            )
        )
        provider = HTTPExtractProvider()

        docs = await provider.extract(["https://example.com/article"])

        assert docs[0].ok is True
        assert "first substantial paragraph" in docs[0].content
        assert "copyright" not in docs[0].content, "boilerplate should be stripped"
        await provider.aclose()

    @respx.mock
    async def test_identifies_itself_honestly(self):
        """Not etiquette — a spoofed browser UA gets a 403 from Wikipedia while
        an identified one gets a 200."""
        route = respx.get("https://example.com/a").mock(
            return_value=httpx.Response(200, html=ARTICLE)
        )
        provider = HTTPExtractProvider()

        await provider.extract(["https://example.com/a"])

        agent = route.calls[0].request.headers["user-agent"]
        assert agent.startswith("SearchRoute/")
        assert "Mozilla" not in agent and "Chrome" not in agent
        await provider.aclose()

    @respx.mock
    async def test_user_agent_is_overridable(self):
        route = respx.get("https://example.com/a").mock(
            return_value=httpx.Response(200, html=ARTICLE)
        )
        provider = HTTPExtractProvider(user_agent="MyApp/2.0 (+mailto:me@example.com)")

        await provider.extract(["https://example.com/a"])

        assert route.calls[0].request.headers["user-agent"] == "MyApp/2.0 (+mailto:me@example.com)"
        await provider.aclose()

    @respx.mock
    async def test_javascript_page_reports_failure_not_empty_content(self):
        """An agent citing an empty page is worse than one told the page failed."""
        respx.get("https://example.com/spa").mock(
            return_value=httpx.Response(200, html="<html><body><div id='root'></div></body></html>")
        )
        provider = HTTPExtractProvider()

        docs = await provider.extract(["https://example.com/spa"])

        assert docs[0].ok is False
        assert docs[0].content is None
        assert "javascript" in docs[0].error.lower()
        await provider.aclose()

    @respx.mock
    async def test_non_html_is_rejected_plainly(self):
        respx.get("https://example.com/paper.pdf").mock(
            return_value=httpx.Response(
                200, content=b"%PDF-1.4 binary", headers={"content-type": "application/pdf"}
            )
        )
        provider = HTTPExtractProvider()

        docs = await provider.extract(["https://example.com/paper.pdf"])

        assert docs[0].ok is False
        assert "application/pdf" in docs[0].error
        await provider.aclose()

    @respx.mock
    async def test_http_error_is_reported_per_url(self):
        respx.get("https://example.com/gone").mock(return_value=httpx.Response(404))
        respx.get("https://example.com/ok").mock(return_value=httpx.Response(200, html=ARTICLE))
        provider = HTTPExtractProvider()

        docs = await provider.extract(["https://example.com/gone", "https://example.com/ok"])

        assert docs[0].ok is False and "404" in docs[0].error
        assert docs[1].ok is True, "one bad URL must not cost the other its content"
        await provider.aclose()

    def test_costs_nothing_and_needs_no_key(self):
        provider = HTTPExtractProvider()
        assert provider.requires_key is False
        assert provider.configured is True
        assert provider.extract_cost_of(["a", "b", "c"]) == 0

    def test_is_ranked_below_hosted_extractors(self):
        """A genuine fallback, not an equal: no JS rendering, no proxies."""
        from searchroute.providers.tavily import TavilyProvider

        assert HTTPExtractProvider.extract_quality < TavilyProvider.extract_quality
        assert HTTPExtractProvider.default_priority > TavilyProvider.default_priority


class TestKeylessContentDepth:
    """The headline claim: depth='content' works with no API keys at all."""

    async def test_content_depth_works_with_zero_keys(self):
        search_only = frozenset({Capability.SEARCH})
        searcher = FakeProvider(
            "keyless_search",
            capabilities=search_only,
            depths=frozenset({Depth.LINKS, Depth.SNIPPETS}),
            results=2,
            priority=1,
        )
        searcher.requires_key = False

        with respx.mock:
            respx.get(url__regex=r"https://keyless_search\.test/\d").mock(
                return_value=httpx.Response(200, html=ARTICLE)
            )
            sr = AsyncSearchRoute(
                custom_providers=[searcher, HTTPExtractProvider()],
                quota_store="memory",
            )
            response = await sr.search("q", depth="content")

        assert response.depth is Depth.CONTENT
        assert response.degraded is False
        assert all(r.content_status is ContentStatus.HYDRATED for r in response.results)
        assert all(r.content_provider == "http_extract" for r in response.results)
        assert all("first substantial paragraph" in r.content for r in response.results)
        # And it cost nothing, because nothing metered was involved.
        assert response.cost.total == 0 or "http_extract" not in response.cost.by_provider
