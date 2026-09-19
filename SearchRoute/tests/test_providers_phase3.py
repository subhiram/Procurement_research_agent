"""The Phase 3 providers: SERP APIs, Firecrawl, Jina, SearXNG, Brave.

Focus is on the places each API deviates from the common shape — those are where
a router silently does the wrong thing.
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from searchroute.config import discover_providers
from searchroute.errors import (
    AuthError,
    ConfigError,
    ProviderError,
    QuotaExceeded,
)
from searchroute.providers.brave import BraveProvider
from searchroute.providers.firecrawl import FirecrawlProvider
from searchroute.providers.google_cse import GoogleCSEProvider
from searchroute.providers.jina import JinaProvider
from searchroute.providers.searchapi import SearchApiProvider
from searchroute.providers.searxng import SearXNGProvider
from searchroute.providers.serpapi import SerpAPIProvider
from searchroute.providers.serper import SerperProvider
from searchroute.quota import Period
from searchroute.types import Capability, ContentStatus, Depth, ErrorKind, SearchQuery

CSE_URL = "https://www.googleapis.com/customsearch/v1"


def cse_body(n=3, start=0):
    return {
        "items": [
            {
                "link": f"https://example.com/{start + i}",
                "title": f"Result {start + i}",
                "snippet": f"Snippet {start + i}",
            }
            for i in range(n)
        ]
    }


class TestGoogleCSE:
    """Google reports an exhausted daily quota as 403, not 402 or 429."""

    @respx.mock
    async def test_403_with_quota_reason_is_quota_not_auth(self):
        respx.get(CSE_URL).mock(
            return_value=httpx.Response(
                403,
                json={
                    "error": {
                        "errors": [{"reason": "dailyLimitExceeded", "message": "limit"}],
                        "code": 403,
                    }
                },
            )
        )
        provider = GoogleCSEProvider(api_key="k", cx="c")

        with pytest.raises(QuotaExceeded) as exc:
            await provider.search(SearchQuery(query="q"))

        assert provider.classify_error(exc.value) is ErrorKind.QUOTA
        assert "resets at midnight" in str(exc.value)
        await provider.aclose()

    @respx.mock
    async def test_403_without_quota_reason_is_auth(self):
        """A genuinely bad key must still surface loudly."""
        respx.get(CSE_URL).mock(
            return_value=httpx.Response(
                403, json={"error": {"errors": [{"reason": "keyInvalid"}], "code": 403}}
            )
        )
        provider = GoogleCSEProvider(api_key="bad", cx="c")

        with pytest.raises(AuthError):
            await provider.search(SearchQuery(query="q"))
        await provider.aclose()

    def test_needs_both_key_and_cx(self):
        """A key with no search-engine id cannot make a single call."""
        assert GoogleCSEProvider(api_key="k").configured is False
        assert GoogleCSEProvider(api_key=None, cx="c").configured is False
        assert GoogleCSEProvider(api_key="k", cx="c").configured is True

    @respx.mock
    async def test_pages_through_ten_at_a_time(self):
        """The API caps a request at 10 results, so 25 means three calls."""
        route = respx.get(CSE_URL).mock(
            side_effect=[
                httpx.Response(200, json=cse_body(10, 0)),
                httpx.Response(200, json=cse_body(10, 10)),
                httpx.Response(200, json=cse_body(5, 20)),
            ]
        )
        provider = GoogleCSEProvider(api_key="k", cx="c")

        response = await provider.search(SearchQuery(query="q", max_results=25))

        assert len(route.calls) == 3
        assert len(response.results) == 25
        # Each request is billed, so the cost must reflect the paging.
        assert response.cost.by_provider["google_cse"] == 3
        await provider.aclose()

    @respx.mock
    async def test_stops_early_when_results_run_out(self):
        route = respx.get(CSE_URL).mock(
            side_effect=[
                httpx.Response(200, json=cse_body(10, 0)),
                httpx.Response(200, json={"items": []}),
            ]
        )
        provider = GoogleCSEProvider(api_key="k", cx="c")

        response = await provider.search(SearchQuery(query="q", max_results=30))

        assert len(route.calls) == 2, "should not keep paging into nothing"
        assert len(response.results) == 10
        await provider.aclose()

    def test_resets_daily(self):
        """The daily reset is what makes this a good filler provider."""
        assert GoogleCSEProvider.quota.period is Period.DAILY


class TestFirecrawl:
    @respx.mock
    async def test_search_with_content_scrapes_inline(self):
        route = respx.post("https://api.firecrawl.dev/v2/search").mock(
            return_value=httpx.Response(
                200,
                json={
                    "success": True,
                    "data": {
                        "web": [
                            {
                                "url": "https://example.com/a",
                                "title": "A",
                                "description": "snippet",
                                "markdown": "# A\n\nbody",
                            }
                        ]
                    },
                },
            )
        )
        provider = FirecrawlProvider(api_key="k")

        response = await provider.search(SearchQuery(query="q", depth=Depth.CONTENT))

        sent = json.loads(route.calls[0].request.content)
        assert sent["scrapeOptions"]["formats"] == ["markdown"]
        assert response.results[0].content == "# A\n\nbody"
        assert response.results[0].content_status is ContentStatus.NATIVE
        await provider.aclose()

    @respx.mock
    async def test_snippets_depth_does_not_request_scraping(self):
        route = respx.post("https://api.firecrawl.dev/v2/search").mock(
            return_value=httpx.Response(200, json={"success": True, "data": {"web": []}})
        )
        provider = FirecrawlProvider(api_key="k")

        await provider.search(SearchQuery(query="q", depth=Depth.SNIPPETS))

        sent = json.loads(route.calls[0].request.content)
        assert "scrapeOptions" not in sent, "must not pay to scrape when only snippets are wanted"
        await provider.aclose()

    def test_cost_model_matches_published_pricing(self):
        """2 credits per 10 results, plus 1 per page scraped."""
        provider = FirecrawlProvider(api_key="k")
        assert provider.cost_of(SearchQuery(query="q", max_results=10)) == 2
        assert provider.cost_of(SearchQuery(query="q", max_results=20)) == 4
        assert provider.cost_of(SearchQuery(query="q", max_results=10, depth=Depth.CONTENT)) == 12

    @respx.mock
    async def test_extract_failure_is_per_url(self):
        respx.post("https://api.firecrawl.dev/v2/scrape").mock(
            side_effect=[
                httpx.Response(200, json={"data": {"markdown": "good"}}),
                httpx.Response(404, json={"error": "not found"}),
            ]
        )
        provider = FirecrawlProvider(api_key="k")

        docs = await provider.extract(["https://a.test", "https://b.test"])

        assert docs[0].ok is True and docs[0].content == "good"
        assert docs[1].ok is False, "one failure must not sink the batch"
        await provider.aclose()

    def test_is_the_preferred_extractor(self):
        """It renders JavaScript, which the local fallback cannot."""
        from searchroute.providers.http_extract import HTTPExtractProvider
        from searchroute.providers.tavily import TavilyProvider

        assert FirecrawlProvider.extract_quality > TavilyProvider.extract_quality
        assert FirecrawlProvider.extract_quality > HTTPExtractProvider.extract_quality


class TestSerpAPI:
    @respx.mock
    async def test_error_in_200_body_is_raised(self):
        """SerpAPI reports some failures as HTTP 200 with an error field."""
        respx.get("https://serpapi.com/search.json").mock(
            return_value=httpx.Response(200, json={"error": "Invalid API key"})
        )
        provider = SerpAPIProvider(api_key="k")

        with pytest.raises(ProviderError):
            await provider.search(SearchQuery(query="q"))
        await provider.aclose()

    @respx.mock
    async def test_run_out_of_searches_is_quota(self):
        respx.get("https://serpapi.com/search.json").mock(
            return_value=httpx.Response(
                200, json={"error": "Your account has run out of searches."}
            )
        )
        provider = SerpAPIProvider(api_key="k")

        with pytest.raises(QuotaExceeded):
            await provider.search(SearchQuery(query="q"))
        await provider.aclose()

    @respx.mock
    async def test_academic_capability_uses_scholar_engine(self):
        route = respx.get("https://serpapi.com/search.json").mock(
            return_value=httpx.Response(200, json={"scholar_results": []})
        )
        provider = SerpAPIProvider(api_key="k")

        await provider.search(SearchQuery(query="q", capability=Capability.ACADEMIC))

        assert route.calls[0].request.url.params["engine"] == "google_scholar"
        await provider.aclose()


class TestSerper:
    @respx.mock
    async def test_normalizes_organic_results(self):
        respx.post("https://google.serper.dev/search").mock(
            return_value=httpx.Response(
                200,
                json={
                    "organic": [
                        {
                            "link": "https://a.test",
                            "title": "A",
                            "snippet": "s",
                            "date": "2 days ago",
                        }
                    ]
                },
            )
        )
        provider = SerperProvider(api_key="k")

        response = await provider.search(SearchQuery(query="q"))

        assert response.results[0].url == "https://a.test"
        # Relative dates are common in SERP payloads.
        assert response.results[0].published_date is not None
        await provider.aclose()

    def test_one_time_grant_is_modelled_as_non_renewing(self):
        """Serper's 2,500 never comes back — the router must know that."""
        assert SerperProvider.quota.period is Period.ONE_TIME
        assert SerperProvider.quota.renews is False


class TestSearchApi:
    @respx.mock
    async def test_normalizes_results(self):
        respx.get("https://www.searchapi.io/api/v1/search").mock(
            return_value=httpx.Response(
                200, json={"organic_results": [{"link": "https://a.test", "title": "A"}]}
            )
        )
        provider = SearchApiProvider(api_key="k")

        response = await provider.search(SearchQuery(query="q"))

        assert len(response.results) == 1
        await provider.aclose()


class TestJina:
    """Its two endpoints have different auth requirements, verified live:
    r.jina.ai answers keyless, s.jina.ai returns 401 without a key."""

    def test_search_needs_a_key_but_extract_does_not(self):
        keyless = JinaProvider()
        assert keyless.can_serve(Capability.EXTRACT) is True
        assert keyless.can_serve(Capability.SEARCH) is False

        keyed = JinaProvider(api_key="k")
        assert keyed.can_serve(Capability.EXTRACT) is True
        assert keyed.can_serve(Capability.SEARCH) is True

    async def test_router_does_not_offer_keyless_jina_for_search(self):
        """Otherwise the guaranteed 401 would trip the breaker and take the
        keyless extraction down with it."""
        from searchroute.client import AsyncSearchRoute

        sr = AsyncSearchRoute(custom_providers=[JinaProvider()], quota_store="memory")
        query = SearchQuery(query="q")

        assert sr.engine.candidates(Capability.SEARCH, query) == []
        assert [p.name for p in sr.engine.candidates(Capability.EXTRACT, query)] == ["jina"]

    async def test_missing_key_for_one_capability_is_explained(self):
        from searchroute.client import AsyncSearchRoute
        from searchroute.errors import NoProviderAvailable

        sr = AsyncSearchRoute(custom_providers=[JinaProvider()], quota_store="memory")

        with pytest.raises(NoProviderAvailable, match="needs an API key"):
            await sr.search("q")

    @respx.mock
    async def test_search_works_with_a_key(self):
        respx.get("https://s.jina.ai/").mock(
            return_value=httpx.Response(
                200, json={"data": [{"url": "https://a.test", "title": "A", "content": "body"}]}
            )
        )
        provider = JinaProvider(api_key="k")

        response = await provider.search(SearchQuery(query="q", depth=Depth.CONTENT))

        assert response.results[0].content == "body"
        await provider.aclose()

    @respx.mock
    async def test_key_is_used_when_present(self):
        route = respx.get("https://s.jina.ai/").mock(
            return_value=httpx.Response(200, json={"data": []})
        )
        provider = JinaProvider(api_key="k")

        await provider.search(SearchQuery(query="q"))

        assert route.calls[0].request.headers["authorization"] == "Bearer k"
        await provider.aclose()

    @respx.mock
    async def test_snippets_depth_asks_for_less(self):
        route = respx.get("https://s.jina.ai/").mock(
            return_value=httpx.Response(200, json={"data": []})
        )
        provider = JinaProvider(api_key="k")

        await provider.search(SearchQuery(query="q", depth=Depth.SNIPPETS))

        assert route.calls[0].request.headers["x-respond-with"] == "no-content"
        await provider.aclose()

    @respx.mock
    async def test_reader_extracts_a_page(self):
        respx.get(url__startswith="https://r.jina.ai/").mock(
            return_value=httpx.Response(
                200, json={"data": {"url": "https://a.test", "title": "A", "content": "# body"}}
            )
        )
        provider = JinaProvider()

        docs = await provider.extract(["https://a.test"])

        assert docs[0].ok is True and docs[0].content == "# body"
        await provider.aclose()


class TestSearXNG:
    def test_requires_an_instance_url(self):
        assert SearXNGProvider().configured is False
        assert SearXNGProvider(base_url="http://localhost:8080").configured is True

    def test_trailing_slash_is_normalized(self):
        provider = SearXNGProvider(base_url="http://localhost:8080/")
        assert provider.base_url == "http://localhost:8080"

    @respx.mock
    async def test_non_json_response_explains_the_fix(self):
        """Forgetting to enable the JSON format is the usual misconfiguration."""
        respx.get("http://localhost:8080/search").mock(
            return_value=httpx.Response(200, text="<html>not json</html>")
        )
        provider = SearXNGProvider(base_url="http://localhost:8080")

        with pytest.raises(ProviderError, match="settings.yml"):
            await provider.search(SearchQuery(query="q"))
        await provider.aclose()

    @respx.mock
    async def test_costs_nothing(self):
        respx.get("http://localhost:8080/search").mock(
            return_value=httpx.Response(
                200, json={"results": [{"url": "https://a.test", "title": "A", "content": "s"}]}
            )
        )
        provider = SearXNGProvider(base_url="http://localhost:8080")

        response = await provider.search(SearchQuery(query="q"))

        assert response.cost.total == 0
        assert provider.quota is None
        await provider.aclose()


class TestBraveIsOptIn:
    def test_marked_opt_in_only(self):
        """Brave retired its free tier; a key in the env is not consent to spend."""
        assert BraveProvider.opt_in_only is True

    def test_excluded_from_auto_discovery_even_with_a_key(self):
        found = discover_providers({"BRAVE_API_KEY": "k"})
        assert "brave" not in found

    def test_can_still_be_named_explicitly(self):
        from searchroute.client import AsyncSearchRoute

        sr = AsyncSearchRoute(
            providers=["brave"], api_keys={"brave": "k"}, quota_store="memory"
        )
        assert sr.providers == ["brave"]


class TestDiscovery:
    def test_discovers_only_configured_providers(self):
        found = discover_providers({"TAVILY_API_KEY": "k"})
        assert "tavily" in found
        assert "exa" not in found

    def test_google_cse_needs_its_search_engine_id(self):
        assert "google_cse" not in discover_providers({"GOOGLE_API_KEY": "k"})
        assert "google_cse" in discover_providers(
            {"GOOGLE_API_KEY": "k", "GOOGLE_CSE_ID": "c"}
        )

    def test_searxng_needs_an_instance_url(self):
        assert "searxng" not in discover_providers({})
        assert "searxng" in discover_providers({"SEARXNG_URL": "http://localhost:8080"})

    def test_keyless_providers_are_always_available(self):
        found = discover_providers({})
        assert "duckduckgo" in found
        assert "http_extract" in found
        # Jina qualifies keyless because its reader endpoint needs no key.
        assert "jina" in found

    def test_results_are_ordered_by_priority(self):
        found = discover_providers({"TAVILY_API_KEY": "k", "EXA_API_KEY": "k"})
        assert found.index("exa") < found.index("tavily") < found.index("duckduckgo")

    def test_unconfigured_pinned_provider_is_a_clear_error(self, monkeypatch):
        from searchroute.client import AsyncSearchRoute

        # Hermetic: the client falls back to the real environment for keys, so
        # this would pass vacuously (or fail) depending on the dev's shell.
        monkeypatch.delenv("EXA_API_KEY", raising=False)
        monkeypatch.delenv("SEARCHROUTE_EXA_API_KEY", raising=False)

        with pytest.raises(ConfigError, match="no API key"):
            AsyncSearchRoute(providers=["exa"], api_keys={}, quota_store="memory")
