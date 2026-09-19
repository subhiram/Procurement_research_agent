"""Degradation paths for search and page fetching.

These run only when something has already gone wrong, which makes them the least
exercised code here and the most likely to be quietly broken. The rule they all
share: a run that loses a source degrades and says so, rather than failing. A
vendor with no page text still ships with its source URL, because a real company
the buyer can look up beats a gap.
"""

from __future__ import annotations

import pytest

from procurement_agent.graph.nodes import vendor_search as node_mod
from procurement_agent.graph.nodes.vendor_search import _fill_content
from procurement_agent.graph.state import VendorCandidate
from procurement_agent.search.client import SessionBudget
from searchroute import Capability, Depth


class _Page:
    """Stands in for a crawl4ai CrawledPage."""

    def __init__(self, url, content="", useful=True):
        self.url = url
        self.content = content
        self.is_useful = useful


class _Fetcher:
    def __init__(self, enabled=True, pages=None, raises=None):
        self._enabled = enabled
        self._pages = pages or {}
        self._raises = raises
        self.asked: list[str] = []

    def is_enabled(self):
        return self._enabled

    async def fetch_many(self, urls):
        self.asked.extend(urls)
        if self._raises:
            raise self._raises
        return [self._pages.get(u, _Page(u, useful=False)) for u in urls]


def _candidates(*urls):
    return [
        VendorCandidate(company_name=f"Co {i}", url=u, snippet="thin")
        for i, u in enumerate(urls)
    ]


@pytest.fixture
def budget():
    return SessionBudget(credits=10)


@pytest.fixture
def no_extract(monkeypatch):
    """Record what the metered extraction tier was asked for."""
    calls: list[list[str]] = []

    async def _extract(urls, budget, settings=None):
        calls.append(list(urls))
        return {}

    monkeypatch.setattr(node_mod, "extract", _extract)
    return calls


class TestCrawlerTier:
    """crawl4ai runs ahead of metered extraction because it costs nothing and,
    unlike SearchRoute's extractors, renders JavaScript."""

    async def test_a_page_the_crawler_gets_never_reaches_the_paid_tier(
        self, monkeypatch, budget, no_extract
    ):
        fetcher = _Fetcher(pages={"https://a.test": _Page("https://a.test", "x" * 900)})
        monkeypatch.setattr(node_mod, "get_fetcher", lambda s: fetcher)

        candidates = _candidates("https://a.test")
        await _fill_content(budget, candidates)

        assert candidates[0].raw_content.startswith("x")
        assert no_extract == []          # nothing was charged for
        assert budget.crawled_free == 1

    async def test_what_the_crawler_misses_falls_through_to_extraction(
        self, monkeypatch, budget, no_extract
    ):
        """Sites that block a headless browser are exactly why the paid tier
        still exists."""
        fetcher = _Fetcher(pages={"https://ok.test": _Page("https://ok.test", "y" * 900)})
        monkeypatch.setattr(node_mod, "get_fetcher", lambda s: fetcher)

        await _fill_content(budget, _candidates("https://ok.test", "https://blocked.test"))

        assert no_extract == [["https://blocked.test"]]

    async def test_a_browser_that_will_not_start_is_not_fatal(
        self, monkeypatch, budget, no_extract
    ):
        """A container without Chromium, or a machine where it fails to launch,
        must still produce leads - just via the metered tier."""
        monkeypatch.setattr(node_mod, "get_fetcher", lambda s: _Fetcher(enabled=False))

        await _fill_content(budget, _candidates("https://a.test"))

        assert no_extract == [["https://a.test"]]

    async def test_extraction_failing_leaves_the_lead_intact(
        self, monkeypatch, budget, no_extract
    ):
        """The last resort: ship the vendor with its source URL and no page
        text, rather than dropping a real company over a failed fetch."""
        monkeypatch.setattr(node_mod, "get_fetcher", lambda s: _Fetcher(enabled=False))

        candidates = _candidates("https://a.test")
        await _fill_content(budget, candidates)

        assert candidates[0].raw_content is None
        assert candidates[0].url == "https://a.test"

    async def test_content_that_is_already_good_is_left_alone(
        self, monkeypatch, budget, no_extract
    ):
        """Providers that return page text inline save the whole fetch."""
        fetcher = _Fetcher()
        monkeypatch.setattr(node_mod, "get_fetcher", lambda s: fetcher)

        candidates = _candidates("https://a.test")
        candidates[0].raw_content = "z" * 5000
        await _fill_content(budget, candidates)

        assert fetcher.asked == []
        assert no_extract == []

    async def test_the_budget_is_never_spent_on_the_free_tier(
        self, monkeypatch, budget, no_extract
    ):
        fetcher = _Fetcher(pages={"https://a.test": _Page("https://a.test", "x" * 900)})
        monkeypatch.setattr(node_mod, "get_fetcher", lambda s: fetcher)

        await _fill_content(budget, _candidates("https://a.test"))

        assert budget.credits_remaining == 10


class TestCapabilityGating:
    """`capability` is a filter, not a hint.

    It is what routes a query to arXiv/PubMed/Crossref or to Wikipedia rather
    than the open web. Getting it wrong does not degrade the answer visibly - it
    quietly asks the wrong index.
    """

    async def test_the_spec_lookup_asks_for_reference_not_the_open_web(
        self, monkeypatch, fake_search
    ):
        from procurement_agent.graph.nodes.clarify_spec import _lookup_material
        from procurement_agent.graph.state import MaterialSpec

        await _lookup_material(MaterialSpec(material_name="Custom 465"))

        assert fake_search.calls
        assert all(c["capability"] is Capability.REFERENCE for c in fake_search.calls)

    async def test_material_research_uses_both_the_web_and_a_reference(
        self, monkeypatch, fake_search
    ):
        """Two angles, because they surface different things: datasheets list
        equivalents, encyclopedias disambiguate a reused trade name."""
        from procurement_agent.config import get_settings
        from procurement_agent.graph.nodes.material_research import _gather_evidence
        from procurement_agent.graph.state import MaterialSpec

        await _gather_evidence(
            MaterialSpec(material_name="Custom 465"), get_settings(), None
        )

        capabilities = {c["capability"] for c in fake_search.calls}
        assert capabilities == {Capability.SEARCH, Capability.REFERENCE}


class TestDegradedDepth:
    async def test_content_depth_is_requested_for_vendor_pages(
        self, monkeypatch, fake_search
    ):
        """Providers that can return page text inline save an extract call, so
        it is always worth asking even though many will not."""
        from procurement_agent.graph.nodes.vendor_search import vendor_search
        from procurement_agent.graph.state import MaterialSpec

        await vendor_search({"material_spec": MaterialSpec(material_name="Custom 465")})

        assert fake_search.calls
        assert all(c["depth"] is Depth.CONTENT for c in fake_search.calls)
