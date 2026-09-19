"""The specialized keyless sources, and the capability scoping that contains them.

The routing tests here are the load-bearing ones: the entire safety argument for
adding Wikipedia and arXiv is that they can never appear in a general search.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from searchroute.client import AsyncSearchRoute
from searchroute.providers.arxiv import ArxivProvider
from searchroute.providers.crossref import CrossrefProvider
from searchroute.providers.hackernews import HackerNewsProvider
from searchroute.providers.pubmed import PubMedProvider
from searchroute.providers.wikipedia import WikipediaProvider
from searchroute.types import Capability, ContentStatus, Depth, SearchQuery

SPECIALIZED = ["arxiv", "pubmed", "wikipedia", "crossref", "hackernews"]

ARXIV_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/1706.03762v5</id>
    <title>Attention Is All You Need</title>
    <summary>The dominant sequence transduction models are based on complex
    recurrent or convolutional neural networks.</summary>
    <published>2017-06-12T00:00:00Z</published>
    <updated>2017-12-06T00:00:00Z</updated>
    <author><name>Ashish Vaswani</name></author>
    <author><name>Noam Shazeer</name></author>
    <category term="cs.CL"/>
    <arxiv:primary_category term="cs.CL"/>
    <arxiv:doi>10.1000/example</arxiv:doi>
    <link title="pdf" href="http://arxiv.org/pdf/1706.03762v5"/>
  </entry>
</feed>
"""


class TestCapabilityScoping:
    """The reason these sources are safe to ship."""

    async def test_none_appear_in_general_search(self):
        sr = AsyncSearchRoute(quota_store="memory")
        candidates = sr.engine.candidates(
            Capability.SEARCH, SearchQuery(query="best pizza in NYC")
        )
        names = {p.name for p in candidates}
        for source in SPECIALIZED:
            assert source not in names, f"{source} leaked into general search"

    @pytest.mark.parametrize(
        "capability,expected",
        [
            (Capability.ACADEMIC, {"arxiv", "pubmed", "crossref"}),
            (Capability.REFERENCE, {"wikipedia"}),
            (Capability.DISCUSSION, {"hackernews"}),
        ],
    )
    async def test_each_is_reachable_by_its_capability(self, capability, expected):
        sr = AsyncSearchRoute(quota_store="memory")
        names = {
            p.name for p in sr.engine.candidates(capability, SearchQuery(query="q"))
        }
        assert expected <= names

    async def test_wikipedia_does_not_join_the_extract_chain(self):
        """It serves CONTENT natively instead. Declaring EXTRACT would offer it
        arbitrary URLs it cannot handle."""
        assert Capability.EXTRACT not in WikipediaProvider.capabilities
        assert Depth.CONTENT in WikipediaProvider.native_depths

    async def test_unmetered_sources_are_preferred_over_paid_ones(self):
        """The payoff: an academic query hits free arXiv before spending Exa."""
        sr = AsyncSearchRoute(
            providers=["exa", "arxiv", "pubmed"],
            api_keys={"exa": "x"},
            strategy="quota_aware",
            quota_store="memory",
        )
        query = SearchQuery(query="q", capability=Capability.ACADEMIC)
        ordered = sr.engine._ordered(
            sr.engine.candidates(Capability.ACADEMIC, query), Capability.ACADEMIC, query
        )
        names = [p.name for p in ordered]
        assert names.index("arxiv") < names.index("exa")
        assert names.index("pubmed") < names.index("exa")

    def test_all_are_free(self):
        for cls in (
            ArxivProvider,
            PubMedProvider,
            WikipediaProvider,
            CrossrefProvider,
            HackerNewsProvider,
        ):
            assert cls.requires_key is False
            assert cls.quota is None
            assert cls().cost_of(SearchQuery(query="q")) == 0


class TestArxiv:
    @respx.mock
    async def test_parses_atom_xml(self):
        """arXiv is the only provider here that isn't JSON."""
        respx.get(url__startswith="https://export.arxiv.org/api/query").mock(
            return_value=httpx.Response(200, text=ARXIV_ATOM)
        )
        provider = ArxivProvider()

        response = await provider.search(SearchQuery(query="attention"))

        hit = response.results[0]
        assert hit.title == "Attention Is All You Need"
        assert hit.url == "http://arxiv.org/abs/1706.03762v5"
        assert "sequence transduction" in hit.snippet
        assert hit.published_date.year == 2017
        # Domain metadata belongs in raw, not new first-class fields.
        assert hit.raw["authors"] == ["Ashish Vaswani", "Noam Shazeer"]
        assert hit.raw["primary_category"] == "cs.CL"
        assert hit.raw["doi"] == "10.1000/example"
        assert hit.raw["pdf_url"] == "http://arxiv.org/pdf/1706.03762v5"
        await provider.aclose()

    @respx.mock
    async def test_malformed_xml_is_a_provider_error(self):
        from searchroute.errors import ProviderError

        respx.get(url__startswith="https://export.arxiv.org").mock(
            return_value=httpx.Response(200, text="<not valid xml")
        )
        provider = ArxivProvider()

        with pytest.raises(ProviderError, match="malformed Atom"):
            await provider.search(SearchQuery(query="q"))
        await provider.aclose()

    def test_uses_https_not_the_documented_http_url(self):
        """The http:// URL in arXiv's docs 301s, and we don't follow redirects."""
        assert ArxivProvider.base_url.startswith("https://")

    def test_declares_the_three_second_rate_limit(self):
        assert ArxivProvider.min_interval == 3.0

    def test_abstract_is_a_snippet_not_a_summary(self):
        """The abstract is author-written, so claiming SUMMARY depth would be a
        lie about who generated it."""
        assert Depth.SUMMARY not in ArxivProvider.native_depths


class TestWikipedia:
    @respx.mock
    async def test_search_strips_html_markup_from_snippets(self):
        respx.get(url__startswith="https://en.wikipedia.org/w/api.php").mock(
            return_value=httpx.Response(
                200,
                json={
                    "query": {
                        "search": [
                            {
                                "pageid": 123,
                                "title": "Transformer",
                                "snippet": 'a <span class="searchmatch">neural</span> network',
                                "timestamp": "2026-01-05T00:00:00Z",
                            }
                        ]
                    }
                },
            )
        )
        provider = WikipediaProvider()

        response = await provider.search(SearchQuery(query="transformer"))

        assert response.results[0].snippet == "a neural network"
        assert "<span" not in response.results[0].snippet
        await provider.aclose()

    @respx.mock
    async def test_content_depth_fetches_extracts_natively(self):
        route = respx.get(url__startswith="https://en.wikipedia.org/w/api.php")
        route.mock(
            side_effect=[
                httpx.Response(
                    200,
                    json={"query": {"search": [{"pageid": 7, "title": "T", "snippet": "s"}]}},
                ),
                httpx.Response(
                    200,
                    json={
                        "query": {
                            "pages": {"7": {"pageid": 7, "extract": "Full article body."}}
                        }
                    },
                ),
            ]
        )
        provider = WikipediaProvider()

        response = await provider.search(SearchQuery(query="t", depth=Depth.CONTENT))

        hit = response.results[0]
        assert hit.content == "Full article body."
        assert hit.content_status is ContentStatus.NATIVE
        assert hit.content_provider == "wikipedia"
        assert response.depth is Depth.CONTENT
        await provider.aclose()

    @respx.mock
    async def test_sends_a_descriptive_user_agent(self):
        """Wikimedia 403s generic browser UAs from scripts."""
        route = respx.get(url__startswith="https://en.wikipedia.org").mock(
            return_value=httpx.Response(200, json={"query": {"search": []}})
        )
        provider = WikipediaProvider(contact="me@example.com")

        await provider.search(SearchQuery(query="q"))

        agent = route.calls[0].request.headers["user-agent"]
        assert agent.startswith("SearchRoute/")
        assert "me@example.com" in agent
        assert "Mozilla" not in agent
        await provider.aclose()


class TestPubMed:
    @respx.mock
    async def test_two_call_esearch_then_esummary(self):
        esearch = respx.get(url__startswith="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch").mock(
            return_value=httpx.Response(200, json={"esearchresult": {"idlist": ["111", "222"]}})
        )
        esummary = respx.get(url__startswith="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary").mock(
            return_value=httpx.Response(
                200,
                json={
                    "result": {
                        "uids": ["111", "222"],
                        "111": {
                            "title": "A paper.",
                            "authors": [{"name": "Smith J"}, {"name": "Doe A"}],
                            "fulljournalname": "Nature",
                            "pubdate": "2026 Jan 15",
                        },
                        "222": {"title": "Another.", "authors": [], "source": "Cell"},
                    }
                },
            )
        )
        provider = PubMedProvider()

        response = await provider.search(SearchQuery(query="q"))

        assert len(esearch.calls) == 1 and len(esummary.calls) == 1
        assert response.results[0].url == "https://pubmed.ncbi.nlm.nih.gov/111/"
        assert response.results[0].title == "A paper"  # trailing period stripped
        assert "Smith J" in response.results[0].snippet
        assert "Nature" in response.results[0].snippet
        assert response.results[0].raw["pmid"] == "111"
        await provider.aclose()

    @respx.mock
    async def test_no_results_skips_the_second_call(self):
        respx.get(url__startswith="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch").mock(
            return_value=httpx.Response(200, json={"esearchresult": {"idlist": []}})
        )
        summary = respx.get(url__startswith="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary")
        provider = PubMedProvider()

        response = await provider.search(SearchQuery(query="q"))

        assert response.results == []
        assert len(summary.calls) == 0, "must not pay for a second call with no ids"
        await provider.aclose()

    @respx.mock
    async def test_sends_tool_and_email(self):
        route = respx.get(url__startswith="https://eutils.ncbi.nlm.nih.gov").mock(
            return_value=httpx.Response(200, json={"esearchresult": {"idlist": []}})
        )
        provider = PubMedProvider(contact="me@example.com")

        await provider.search(SearchQuery(query="q"))

        params = route.calls[0].request.url.params
        assert params["tool"] == "searchroute"
        assert params["email"] == "me@example.com"
        await provider.aclose()


class TestCrossref:
    @respx.mock
    async def test_normalizes_and_parses_nested_dates(self):
        respx.get(url__startswith="https://api.crossref.org/works").mock(
            return_value=httpx.Response(
                200,
                json={
                    "message": {
                        "items": [
                            {
                                "DOI": "10.1/abc",
                                "title": ["A Paper"],
                                "author": [{"given": "Jane", "family": "Doe"}],
                                "container-title": ["Journal of Things"],
                                "issued": {"date-parts": [[2024, 3, 15]]},
                                "is-referenced-by-count": 42,
                            }
                        ]
                    }
                },
            )
        )
        provider = CrossrefProvider()

        response = await provider.search(SearchQuery(query="q"))

        hit = response.results[0]
        assert hit.title == "A Paper"
        assert hit.url == "https://doi.org/10.1/abc"
        assert "Jane Doe" in hit.snippet
        assert "cited by 42" in hit.snippet
        assert (hit.published_date.year, hit.published_date.month) == (2024, 3)
        await provider.aclose()

    @respx.mock
    async def test_partial_date_parts_do_not_crash(self):
        """Crossref often gives only a year."""
        respx.get(url__startswith="https://api.crossref.org").mock(
            return_value=httpx.Response(
                200,
                json={
                    "message": {
                        "items": [
                            {"DOI": "10.1/x", "title": ["T"], "issued": {"date-parts": [[2020]]}}
                        ]
                    }
                },
            )
        )
        provider = CrossrefProvider()

        response = await provider.search(SearchQuery(query="q"))

        assert response.results[0].published_date.year == 2020
        await provider.aclose()

    @respx.mock
    async def test_polite_pool_mailto(self):
        route = respx.get(url__startswith="https://api.crossref.org").mock(
            return_value=httpx.Response(200, json={"message": {"items": []}})
        )
        provider = CrossrefProvider(contact="me@example.com")

        await provider.search(SearchQuery(query="q"))

        assert route.calls[0].request.url.params["mailto"] == "me@example.com"
        assert "me@example.com" in route.calls[0].request.headers["user-agent"]
        await provider.aclose()


class TestHackerNews:
    @respx.mock
    async def test_prefers_the_linked_article_but_keeps_the_thread(self):
        respx.get(url__startswith="https://hn.algolia.com/api/v1/search").mock(
            return_value=httpx.Response(
                200,
                json={
                    "hits": [
                        {
                            "objectID": "999",
                            "title": "A Story",
                            "url": "https://example.com/article",
                            "points": 250,
                            "num_comments": 88,
                            "author": "someone",
                            "created_at_i": 1735689600,
                        }
                    ]
                },
            )
        )
        provider = HackerNewsProvider()

        response = await provider.search(SearchQuery(query="q"))

        hit = response.results[0]
        assert hit.url == "https://example.com/article"
        assert hit.raw["discussion_url"] == "https://news.ycombinator.com/item?id=999"
        assert "250 points" in hit.snippet and "88 comments" in hit.snippet
        assert hit.published_date is not None
        await provider.aclose()

    @respx.mock
    async def test_ask_hn_with_no_url_falls_back_to_the_thread(self):
        respx.get(url__startswith="https://hn.algolia.com").mock(
            return_value=httpx.Response(
                200,
                json={"hits": [{"objectID": "5", "title": "Ask HN: ...", "points": 10}]},
            )
        )
        provider = HackerNewsProvider()

        response = await provider.search(SearchQuery(query="q"))

        assert response.results[0].url == "https://news.ycombinator.com/item?id=5"
        await provider.aclose()
