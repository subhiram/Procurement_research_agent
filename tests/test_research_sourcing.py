"""Mining research papers for supplier names.

The value of this path is that the names come from a real sourcing record, so
the guard against the model contributing plausible suppliers from its own
knowledge is the thing most worth testing.
"""

from __future__ import annotations

import pytest

from procurement_agent.extraction.patterns import (
    build_sourcing_digest,
    find_sourcing_mentions,
)
from procurement_agent.graph.nodes import research_sourcing as node_mod
from procurement_agent.graph.nodes.research_sourcing import research_sourcing
from procurement_agent.graph.nodes.vendor_search import build_queries
from procurement_agent.graph.state import MaterialSpec
from searchroute import Capability
from tests.conftest import FakeHit

PAPER = """
Fatigue behaviour of precipitation-hardened stainless steel

Custom 465 bar stock was supplied by Carpenter Technology Corporation,
Philadelphia, PA, USA. Specimens were machined at the University of Sheffield.
Heat treatment was carried out by Bodycote plc. The authors thank the EPSRC for
funding under grant EP/000000/1. Testing equipment was provided by Instron.
"""


class TestSourcingPatterns:
    def test_finds_supplied_by(self):
        mentions = find_sourcing_mentions(PAPER)
        assert any("Carpenter Technology" in m.candidate_text for m in mentions)

    def test_finds_multiple_phrasings(self):
        text = (
            "Material was purchased from Alpha Metals. Powder was obtained from "
            "Beta Alloys. Bar was kindly provided by Gamma Steel."
        )
        found = " ".join(m.candidate_text for m in find_sourcing_mentions(text))
        assert "Alpha Metals" in found
        assert "Beta Alloys" in found
        assert "Gamma Steel" in found

    def test_returns_nothing_when_absent(self):
        assert find_sourcing_mentions("We tested a steel bar at room temperature.") == []

    def test_digest_reports_absence_explicitly(self):
        assert "No sourcing statements" in build_sourcing_digest([])


class _Stub:
    def __init__(self, names):
        self.payload = node_mod.Suppliers(company_names=names)

    async def ainvoke(self, messages, *a, **k):
        return self.payload


def _paper_hit():
    return FakeHit(title="Fatigue behaviour", url="http://arxiv.org/abs/1", content=PAPER)


@pytest.fixture
def state():
    return {"material_spec": MaterialSpec(material_name="Custom 465")}


class TestResearchSourcing:
    async def test_extracts_a_supplier_named_in_a_paper(
        self, monkeypatch, state, fake_search
    ):
        fake_search.hits = [_paper_hit()]
        monkeypatch.setattr(
            node_mod, "get_model", lambda *a, **k: _Stub(["Carpenter Technology Corporation"])
        )

        result = await research_sourcing(state)

        assert result["sourcing_companies"] == ["Carpenter Technology Corporation"]

    async def test_searches_the_academic_capability(
        self, monkeypatch, state, fake_search
    ):
        """Capability is a filter, not a hint.

        ACADEMIC is what routes the query to arXiv, PubMed and Crossref; a plain
        SEARCH call would never reach any of them, so this node would silently
        become an open-web search for papers.
        """
        fake_search.hits = [_paper_hit()]
        monkeypatch.setattr(node_mod, "get_model", lambda *a, **k: _Stub([]))

        await research_sourcing(state)

        assert fake_search.calls
        assert all(c["capability"] is Capability.ACADEMIC for c in fake_search.calls)

    async def test_drops_suppliers_not_present_in_the_papers(
        self, monkeypatch, state, fake_search
    ):
        """The whole point is that the name came from a real sourcing record,
        so a plausible one from the model's own knowledge must not survive."""
        fake_search.hits = [_paper_hit()]
        monkeypatch.setattr(
            node_mod,
            "get_model",
            lambda *a, **k: _Stub(["Carpenter Technology Corporation", "ArcelorMittal"]),
        )

        result = await research_sourcing(state)

        assert result["sourcing_companies"] == ["Carpenter Technology Corporation"]

    async def test_no_papers_yields_no_suppliers(self, state, fake_search):
        fake_search.hits = []
        assert (await research_sourcing(state))["sourcing_companies"] == []

    async def test_model_failure_is_not_fatal(self, monkeypatch, state, fake_search):
        class _Boom:
            async def ainvoke(self, *a, **k):
                raise RuntimeError("provider down")

        fake_search.hits = [_paper_hit()]
        monkeypatch.setattr(node_mod, "get_model", lambda *a, **k: _Boom())

        assert (await research_sourcing(state))["sourcing_companies"] == []

    async def test_can_be_disabled_by_config(self, monkeypatch, state, fake_search):
        settings = node_mod.get_settings()
        monkeypatch.setattr(settings, "enable_research_search", False)

        assert (await research_sourcing(state))["sourcing_companies"] == []
        assert fake_search.calls == []


class TestQueryIntegration:
    def test_paper_sourced_companies_are_searched_first(self):
        """A documented purchase outranks a search ranking, so those queries
        must not be the ones the cap drops."""
        queries = build_queries(
            MaterialSpec(material_name="Custom 465"),
            None,
            ["Carpenter Technology Corporation"],
        )
        assert "Carpenter Technology Corporation" in queries[0]

    def test_works_with_no_research_companies(self):
        assert build_queries(MaterialSpec(material_name="Custom 465"), None, []) == (
            build_queries(MaterialSpec(material_name="Custom 465"), None)
        )
