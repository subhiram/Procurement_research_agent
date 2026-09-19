"""Depth and hydration — the second routing axis.

These are the tests that matter for the research use case: asking for full page
content and getting either the content or an honest account of why not.
"""

from __future__ import annotations

from searchroute.client import AsyncSearchRoute
from searchroute.types import Capability, ContentStatus, Depth

from .conftest import FakeProvider

SNIPPET_ONLY = frozenset({Depth.LINKS, Depth.SNIPPETS})
SEARCH_ONLY = frozenset({Capability.SEARCH})
"""A SERP provider like Google CSE: it finds pages, it cannot fetch them."""
FULL_DEPTH = frozenset({Depth.LINKS, Depth.SNIPPETS, Depth.SUMMARY, Depth.CONTENT})


def client(providers, **kwargs) -> AsyncSearchRoute:
    kwargs.setdefault("quota_store", "memory")
    return AsyncSearchRoute(custom_providers=providers, **kwargs)


class TestNativeDepth:
    async def test_native_content_costs_no_extract(self):
        """A provider that returns content inline must not trigger a second hop."""
        rich = FakeProvider("rich", depths=FULL_DEPTH, results=3)
        sr = client([rich])

        response = await sr.search("q", depth="content")

        assert rich.extract_calls == [], "no extract call should have been made"
        assert response.depth is Depth.CONTENT
        assert response.degraded is False
        assert all(r.content_status is ContentStatus.NATIVE for r in response.results)
        assert all(r.content for r in response.results)

    async def test_snippets_depth_never_fetches_content(self):
        provider = FakeProvider("p", depths=SNIPPET_ONLY, results=3)
        sr = client([provider])

        response = await sr.search("q", depth="snippets")

        assert provider.extract_calls == []
        assert response.degraded is False
        assert all(r.content is None for r in response.results)
        assert all(r.content_status is ContentStatus.NOT_REQUESTED for r in response.results)


class TestHydration:
    async def test_depth_gap_triggers_hydration(self):
        """A snippets-only searcher plus an extractor should still yield content."""
        searcher = FakeProvider(
            "searcher", depths=SNIPPET_ONLY, capabilities=SEARCH_ONLY, results=3, priority=1
        )
        extractor = FakeProvider("extractor", depths=FULL_DEPTH, priority=2)
        sr = client([searcher, extractor])

        response = await sr.search("q", depth="content", providers=None)

        assert len(extractor.extract_calls) > 0, "should have hydrated via the extractor"
        assert response.depth is Depth.CONTENT
        assert all(r.content for r in response.results)
        assert all(r.content_status is ContentStatus.HYDRATED for r in response.results)
        # The searcher and the content provider are legitimately different.
        assert all(r.provider == "searcher" for r in response.results)
        assert all(r.content_provider == "extractor" for r in response.results)

    async def test_partial_failure_keeps_the_rest(self):
        """One blocked page must not cost the others their content."""
        searcher = FakeProvider(
            "searcher", depths=SNIPPET_ONLY, capabilities=SEARCH_ONLY, results=3, priority=1
        )
        blocked = {"https://searcher.test/1"}
        extractor = FakeProvider(
            "extractor", depths=FULL_DEPTH, priority=2, extract_fails=blocked
        )
        sr = client([searcher, extractor])

        response = await sr.search("q", depth="content")

        by_url = {r.url: r for r in response.results}
        failed = by_url["https://searcher.test/1"]
        assert failed.content is None
        assert failed.content_status is ContentStatus.FAILED
        assert failed.snippet, "a failed hydration must still leave the snippet usable"

        others = [r for r in response.results if r.url not in blocked]
        assert all(r.content for r in others)
        assert all(r.content_status is ContentStatus.HYDRATED for r in others)
        assert response.degraded is True

    async def test_max_hydrate_budget_marks_overflow_skipped(self):
        searcher = FakeProvider(
            "searcher", depths=SNIPPET_ONLY, capabilities=SEARCH_ONLY, results=5, priority=1
        )
        extractor = FakeProvider("extractor", depths=FULL_DEPTH, priority=2)
        sr = client([searcher, extractor])

        response = await sr.search("q", depth="content", max_hydrate=2)

        hydrated = [r for r in response.results if r.content_status is ContentStatus.HYDRATED]
        skipped = [r for r in response.results if r.content_status is ContentStatus.SKIPPED]
        assert len(hydrated) == 2, "budget should cap the number of pages fetched"
        assert len(skipped) == 3
        assert response.degraded is True
        # Crucially, the budget must actually prevent the spend.
        fetched = sum(len(call) for call in extractor.extract_calls)
        assert fetched == 2

    async def test_urls_are_batched_into_one_extract_call(self):
        """Batch size is a billing decision.

        Providers price extraction per call or per N URLs, so splitting a handful
        of URLs into one call each would multiply the cost for no benefit.
        """
        searcher = FakeProvider(
            "searcher", depths=SNIPPET_ONLY, capabilities=SEARCH_ONLY, results=5, priority=1
        )
        extractor = FakeProvider("extractor", depths=FULL_DEPTH, priority=2)
        sr = client([searcher, extractor])

        await sr.search("q", depth="content")

        assert len(extractor.extract_calls) == 1, (
            f"5 URLs should be one batched call, got {len(extractor.extract_calls)}"
        )
        assert len(extractor.extract_calls[0]) == 5

    async def test_max_hydrate_zero_disables_extraction_entirely(self):
        searcher = FakeProvider(
            "searcher", depths=SNIPPET_ONLY, capabilities=SEARCH_ONLY, results=3, priority=1
        )
        extractor = FakeProvider("extractor", depths=FULL_DEPTH, priority=2)
        sr = client([searcher, extractor])

        response = await sr.search("q", depth="content", max_hydrate=0)

        assert extractor.extract_calls == []
        assert all(r.content_status is ContentStatus.SKIPPED for r in response.results)
        assert response.degraded is True

    async def test_content_survives_a_provider_normalizing_the_url(self):
        """Providers echo the URL they *resolved to*, not the one we sent.

        A trailing slash, an http->https upgrade or a followed redirect all
        produce a different string. Filing the content under that string would
        hand the caller back a "not attempted" placeholder while the content
        sits under an address nobody asked about.
        """
        from searchroute.types import Document

        class RedirectingExtractor(FakeProvider):
            async def extract(self, urls, **kwargs):
                self.extract_calls.append(list(urls))
                # Same page, canonically different string.
                return [
                    Document(
                        url=u.replace("http://", "https://").rstrip("/") + "/",
                        content=f"BODY {u}",
                        ok=True,
                        provider=self.name,
                    )
                    for u in urls
                ]

        sr = client([RedirectingExtractor("redirector", depths=FULL_DEPTH)])

        docs = await sr.extract(["http://example.com", "https://other.test/page"])

        assert all(d.ok for d in docs), [d.error for d in docs]
        # Order and identity are preserved from the caller's point of view.
        assert len(docs) == 2
        assert all(d.content for d in docs)

    async def test_no_extractor_configured_says_so_actionably(self):
        """Pinning search-only providers silently breaks content depth unless
        we say why — "no extractor" looks identical to "every page blocked"."""
        searcher = FakeProvider(
            "searcher", depths=SNIPPET_ONLY, capabilities=SEARCH_ONLY, results=3
        )
        sr = client([searcher])

        response = await sr.search("q", depth="content")

        assert response.degraded is True
        assert all(r.content_status is ContentStatus.FAILED for r in response.results)
        joined = " ".join(response.notes)
        assert "no EXTRACT-capable provider is configured" in joined
        assert "hydrate_providers" in joined

    async def test_no_notes_when_everything_worked(self):
        rich = FakeProvider("rich", depths=FULL_DEPTH, results=3)
        sr = client([rich])

        response = await sr.search("q", depth="content")

        assert response.notes == []

    async def test_total_extract_failure_degrades_but_keeps_results(self):
        """If the whole extract chain is down, we still return the search hits."""
        from searchroute.errors import ProviderError

        searcher = FakeProvider(
            "searcher", depths=SNIPPET_ONLY, capabilities=SEARCH_ONLY, results=3, priority=1
        )
        broken = FakeProvider(
            "broken", depths=FULL_DEPTH, priority=2, fail_with=ProviderError("broken", "down")
        )
        sr = client([searcher, broken])

        response = await sr.search("q", depth="content")

        assert len(response.results) == 3, "search results survive an extraction outage"
        assert all(r.snippet for r in response.results)
        assert all(r.content_status is ContentStatus.FAILED for r in response.results)
        assert response.degraded is True


class TestSummaryIsNeverSynthesized:
    async def test_summary_degrades_rather_than_inventing_one(self):
        """The library has no model and must not pretend otherwise."""
        provider = FakeProvider("p", depths=SNIPPET_ONLY, results=3)
        sr = client([provider])

        response = await sr.search("q", depth="summary")

        assert all(r.summary is None for r in response.results)
        assert response.depth is Depth.SNIPPETS
        assert response.requested_depth is Depth.SUMMARY
        assert response.degraded is True
        assert provider.extract_calls == [], "summary must not fall back to extraction"


class TestProfiles:
    async def test_quicklook_is_cheap(self):
        provider = FakeProvider("p", depths=SNIPPET_ONLY, results=10)
        sr = client([provider], profile="quicklook")

        response = await sr.search("q")

        assert sr.settings.depth is Depth.SNIPPETS
        assert len(response.results) == 5
        assert provider.extract_calls == []

    async def test_rag_profile_requests_content(self):
        provider = FakeProvider("p", depths=FULL_DEPTH, results=10)
        sr = client([provider], profile="rag")

        response = await sr.search("q")

        assert sr.settings.depth is Depth.CONTENT
        assert response.depth is Depth.CONTENT
        assert all(r.content for r in response.results)

    async def test_explicit_kwarg_overrides_profile(self):
        provider = FakeProvider("p", depths=FULL_DEPTH, results=10)
        sr = client([provider], profile="rag", max_results=3)

        response = await sr.search("q")

        assert len(response.results) == 3

    async def test_per_call_depth_overrides_client_default(self):
        provider = FakeProvider("p", depths=FULL_DEPTH, results=5)
        sr = client([provider], profile="rag")

        cheap = await sr.search("q", depth="snippets")

        assert cheap.depth is Depth.SNIPPETS
        assert all(r.content is None for r in cheap.results)


class TestCostReporting:
    async def test_cost_is_reported_per_provider(self):
        provider = FakeProvider("p", depths=FULL_DEPTH, results=3, cost=7)
        sr = client([provider])

        response = await sr.search("q")

        assert response.cost.by_provider["p"] == 7
        assert response.cost.total == 7

    async def test_attempts_record_the_full_trail(self, fatal_error):
        broken = FakeProvider("broken", priority=1, fail_with=fatal_error)
        working = FakeProvider("working", priority=2)
        sr = client([broken, working])

        response = await sr.search("q")

        assert [a.provider for a in response.attempts] == ["broken", "working"]
        assert response.attempts[0].ok is False
        assert response.attempts[0].error is not None
        assert response.attempts[1].ok is True
