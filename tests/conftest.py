"""Shared fixtures."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from procurement_agent.config import get_settings
from procurement_agent.graph.state import VendorLead
from searchroute import Capability, ContentStatus


@dataclass
class FakeHit:
    """Stands in for a searchroute.SearchResult.

    Structural rather than a real SearchResult so the tests keep working if
    SearchRoute adds a required field; what the agent actually consumes is
    `url`, `title`, `snippet`, `content` and the `text` property.
    """

    url: str
    title: str = ""
    snippet: str | None = None
    content: str | None = None
    content_status: ContentStatus = ContentStatus.NOT_REQUESTED
    provider: str = "fake"

    @property
    def text(self) -> str:
        return self.content or self.snippet or ""


@dataclass
class FakeSearch:
    """Scripted replacement for `search.client.search`.

    Records every call so a test can assert on which capability was requested -
    that is a real behavioural distinction, not a detail: ACADEMIC is what
    routes a query to arXiv/PubMed/Crossref rather than the open web.
    """

    hits: list = field(default_factory=list)
    by_capability: dict = field(default_factory=dict)
    calls: list = field(default_factory=list)
    error: Exception | None = None

    async def __call__(self, query, budget, *, capability=Capability.SEARCH, **kwargs):
        self.calls.append({"query": query, "capability": capability, **kwargs})
        if self.error is not None:
            raise self.error
        if capability in self.by_capability:
            return list(self.by_capability[capability])
        return list(self.hits)

    @property
    def queries(self) -> list[str]:
        return [c["query"] for c in self.calls]


@pytest.fixture
def fake_search(monkeypatch):
    """Patch the search seam in every node module that imported it.

    The nodes do `from ...search.client import search`, so patching the client
    module alone would not reach them - each importer holds its own reference.
    """
    fake = FakeSearch()
    for module in (
        "procurement_agent.graph.nodes.research_sourcing",
        "procurement_agent.graph.nodes.material_research",
        "procurement_agent.graph.nodes.vendor_search",
        "procurement_agent.graph.nodes.clarify_spec",
        "procurement_agent.graph.nodes.contact_extraction",
    ):
        monkeypatch.setattr(f"{module}.search", fake, raising=False)
    return fake


@pytest.fixture(autouse=True)
def _no_network_crawling(monkeypatch):
    """Keep the crawler out of the unit suite.

    `contact_extraction` now follows contact pages, which starts a real browser
    and hits real vendor websites. Left enabled, the suite hangs for minutes and
    hammers third-party sites. Tests that exercise contact-page following stub
    `_fetch_contact_pages` explicitly instead.
    """
    monkeypatch.setattr(get_settings(), "enable_crawl4ai", False)


@pytest.fixture(autouse=True)
def _assume_ollama_reachable(monkeypatch):
    """Assume the local Ollama daemon is running, as it is on a dev machine.

    LLMRoute's registry only offers Ollama's endpoints when a live TCP probe of
    OLLAMA_BASE_URL succeeds - the fix for a Streamlit Cloud deploy (no daemon
    at all) wrongly being offered it and hanging on every call. Without this
    fixture, this suite's own tier-composition tests (e.g.
    `test_bulk_can_reach_a_free_local_endpoint`) would depend on whichever
    machine runs them actually having Ollama up, which CI does not.
    """
    from llm_router import registry

    monkeypatch.setattr(registry, "_ollama_daemon_reachable", lambda: True)

#: A realistic scraped supplier page. Deliberately includes the noise that
#: breaks naive extraction: an obfuscated address, a careers mailbox, a phone in
#: a different format from how a model would write it, and numbers that look
#: like phone numbers but are dimensions and part codes.
SUPPLIER_PAGE = """
Precision Alloys Ltd - Specialty Stainless & Nickel Alloys

We stock Custom 465 (UNS S46500) round bar in condition H900 and H1000.
Diameters from 0.25 in to 6.00 in. Cut lengths 12 - 144 in available.
Part reference 465-RB-2000 for 2.000 inch diameter bar.

Certifications: AS9100D, ISO 9001:2015, and EN 10204 3.1 material certificates
supplied with every order. NADCAP approved heat treatment.

Sales enquiries: sales [at] precisionalloys [dot] co [dot] uk
Careers: recruitment@precisionalloys.co.uk
Head office: +44 (0)121 555 0147
Registered in England, company number 04412299.

Contact: Margaret Ellison, Sales Director
"""


@pytest.fixture
def supplier_page() -> str:
    return SUPPLIER_PAGE


@pytest.fixture
def lead_factory():
    def make(**overrides) -> VendorLead:
        base = {
            "company_name": "Precision Alloys Ltd",
            "website": "https://precisionalloys.co.uk",
            "source_url": "https://precisionalloys.co.uk/custom-465",
        }
        return VendorLead(**{**base, **overrides})

    return make
