"""Grounding the material research against retrieved sources.

This node was the largest source of wrong output in the graph. A model asked for
the UNS equivalent of a trade name supplies one whether or not it knows it, and
a fabricated designation is undetectable by inspection: "UNS S46500" reads
exactly as plausibly as "UNS S45500".

The damage is not contained to this node either. `vendor_search.build_queries()`
builds its query set from these designations, so one invented number sends the
whole vendor search after a different alloy — which is why the tests below are
about what gets *discarded*, not about what the model produced.
"""

from __future__ import annotations

import pytest

from procurement_agent.graph.nodes import material_research as node_mod
from procurement_agent.graph.nodes.material_research import material_research
from procurement_agent.graph.nodes.vendor_search import build_queries
from procurement_agent.graph.state import MaterialResearch, MaterialSpec
from tests.conftest import FakeHit

SOURCE = """
Custom 465 is a martensitic age-hardenable stainless steel produced by Carpenter
Technology. It is designated UNS S46500 and covered by AMS 5936 in bar form.
Also sold as Custom465 and covered by ASTM A564 for general stainless bar.
"""


def _hit():
    return FakeHit(
        title="Custom 465 datasheet",
        url="https://example.test/custom-465",
        content=SOURCE,
    )


class _Stub:
    def __init__(self, research: MaterialResearch):
        self.research = research
        self.prompts: list = []

    async def ainvoke(self, messages, *a, **k):
        self.prompts.append(messages)
        return self.research


@pytest.fixture
def state():
    return {"material_spec": MaterialSpec(material_name="Custom 465")}


def _research(**overrides) -> MaterialResearch:
    base = {
        "canonical_name": "Custom 465",
        "designations": ["UNS S46500", "AMS 5936"],
        "synonyms": ["Custom465"],
        "common_forms": ["bar"],
        "ambiguities": [],
    }
    return MaterialResearch(**{**base, **overrides})


class TestVerification:
    async def test_keeps_designations_present_in_the_sources(
        self, monkeypatch, state, fake_search
    ):
        fake_search.hits = [_hit()]
        monkeypatch.setattr(node_mod, "get_model", lambda *a, **k: _Stub(_research()))

        result = await material_research(state)

        assert result["research"].designations == ["UNS S46500", "AMS 5936"]

    async def test_drops_a_designation_the_sources_do_not_support(
        self, monkeypatch, state, fake_search
    ):
        """The core failure. UNS S45500 is a real designation for a different
        alloy, so it is exactly as plausible as the right one and only the
        sources can tell them apart."""
        fake_search.hits = [_hit()]
        monkeypatch.setattr(
            node_mod,
            "get_model",
            lambda *a, **k: _Stub(_research(designations=["UNS S46500", "UNS S45500"])),
        )

        result = await material_research(state)

        assert result["research"].designations == ["UNS S46500"]

    async def test_says_in_the_notes_what_was_dropped(
        self, monkeypatch, state, fake_search
    ):
        """A thin designation list must read as "we could not source these",
        not as "this material has no equivalents"."""
        fake_search.hits = [_hit()]
        monkeypatch.setattr(
            node_mod,
            "get_model",
            lambda *a, **k: _Stub(_research(designations=["UNS S45500"])),
        )

        result = await material_research(state)

        assert "UNS S45500" in result["research"].notes

    async def test_drops_a_family_standard_even_when_it_is_in_the_sources(
        self, monkeypatch, state, fake_search
    ):
        """ASTM A564 genuinely appears in the source text and still must not
        survive: it covers every age-hardening stainless bar, so as a search
        term it finds the wrong alloys and as a designation it makes a
        neighbouring grade look like a match."""
        fake_search.hits = [_hit()]
        monkeypatch.setattr(
            node_mod,
            "get_model",
            lambda *a, **k: _Stub(_research(designations=["UNS S46500", "ASTM A564"])),
        )

        result = await material_research(state)

        assert "ASTM A564" not in result["research"].designations

    async def test_drops_everything_when_no_sources_were_retrieved(
        self, monkeypatch, state, fake_search
    ):
        """An unsourced designation is precisely what this node exists to stop,
        so no evidence means no designations rather than falling back to the
        model's own memory."""
        fake_search.hits = []
        monkeypatch.setattr(node_mod, "get_model", lambda *a, **k: _Stub(_research()))

        result = await material_research(state)

        assert result["research"].designations == []
        assert "No reference sources" in result["research"].notes

    async def test_ambiguities_are_never_filtered(
        self, monkeypatch, state, fake_search
    ):
        """This field is the model reporting its own uncertainty. There is no
        corpus a doubt could be verified against, and filtering it would delete
        exactly the warning the buyer most needs."""
        fake_search.hits = [_hit()]
        doubt = "Could be the Carpenter alloy or an unrelated tool steel"
        monkeypatch.setattr(
            node_mod, "get_model", lambda *a, **k: _Stub(_research(ambiguities=[doubt]))
        )

        result = await material_research(state)

        assert result["research"].ambiguities == [doubt]

    async def test_canonical_name_is_not_filtered(
        self, monkeypatch, state, fake_search
    ):
        """It is the model's judgment about naming, not a citable designation."""
        fake_search.hits = [_hit()]
        monkeypatch.setattr(
            node_mod,
            "get_model",
            lambda *a, **k: _Stub(_research(canonical_name="Some name not in sources")),
        )

        result = await material_research(state)

        assert result["research"].canonical_name == "Some name not in sources"

    async def test_the_sources_are_actually_given_to_the_model(
        self, monkeypatch, state, fake_search
    ):
        """Verification alone would just silently discard everything; the model
        has to be handed the evidence it is expected to work from."""
        fake_search.hits = [_hit()]
        stub = _Stub(_research())
        monkeypatch.setattr(node_mod, "get_model", lambda *a, **k: stub)

        await material_research(state)

        prompt = "".join(m.content for m in stub.prompts[0])
        assert "UNS S46500" in prompt


class TestDownstreamEffect:
    def test_a_verified_designation_becomes_a_vendor_query(self):
        queries = build_queries(
            MaterialSpec(material_name="Custom 465"),
            _research(designations=["UNS S46500"]),
        )
        assert any("UNS S46500" in q for q in queries)

    def test_a_family_standard_never_becomes_a_vendor_query(self):
        """The second guard, downstream of the node's own filtering: a query
        for "ASTM B348 supplier stockist" returns every titanium bar vendor
        alive rather than the one being sourced."""
        queries = build_queries(
            MaterialSpec(material_name="Titanium Grade 5"),
            _research(designations=["ASTM B348"], synonyms=[]),
        )
        assert not any("ASTM B348" in q for q in queries)
