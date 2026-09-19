"""Query construction, ranking, deduplication, and the clarification guard."""

from __future__ import annotations

from procurement_agent.graph.nodes.clarify_spec import _protect_spec
from procurement_agent.graph.nodes.contact_extraction import _rank_certifications
from procurement_agent.graph.nodes.vendor_search import build_queries
from procurement_agent.graph.nodes.vendor_summary import vendor_summary
from procurement_agent.graph.state import MaterialResearch, MaterialSpec, VendorLead


def spec(**overrides) -> MaterialSpec:
    return MaterialSpec(**{"material_name": "Custom 465", **overrides})


def lead(name, url, **overrides) -> VendorLead:
    return VendorLead(
        company_name=name, website=url, source_url=f"{url}/page", **overrides
    )


class TestQueryBuilding:
    def test_covers_distinct_commercial_roles(self):
        queries = build_queries(spec(), None)
        joined = " ".join(queries)
        # Distributors and manufacturers are genuinely different companies.
        assert "supplier" in joined
        assert "distributor" in joined
        assert "manufacturer" in joined

    def test_searches_formal_designations_too(self):
        """Many vendors list the UNS number and never the trade name."""
        research = MaterialResearch(
            canonical_name="Custom 465", designations=["UNS S46500", "AMS 5936"]
        )
        joined = " ".join(build_queries(spec(), research))
        assert "UNS S46500" in joined

    def test_deduplicates_and_caps(self):
        research = MaterialResearch(
            canonical_name="Custom 465",
            designations=["Custom 465", "Custom 465"],
            synonyms=["Custom 465"],
        )
        queries = build_queries(spec(), research)
        assert len(queries) == len(set(queries))
        assert len(queries) <= 6


class TestRanking:
    async def test_manufacturers_outrank_distributors_and_traders(self):
        leads = [
            lead("Trader", "https://trader.example", kind="trader"),
            lead("Maker", "https://maker.example", kind="manufacturer"),
            lead("Stockist", "https://stockist.example", kind="distributor"),
        ]
        result = await vendor_summary({"extracted_leads": leads})
        assert [v.company_name for v in result["vendor_leads"]] == [
            "Maker",
            "Stockist",
            "Trader",
        ]

    async def test_directory_listings_sink_below_real_vendors(self):
        """Directories aggregate suppliers; they are not themselves suppliers."""
        leads = [
            lead("ThomasNet", "https://thomasnet.com", kind="manufacturer"),
            lead("Real Vendor", "https://realvendor.example", kind="unknown"),
        ]
        result = await vendor_summary({"extracted_leads": leads})
        assert result["vendor_leads"][0].company_name == "Real Vendor"

    async def test_reachable_vendors_outrank_unreachable_ones(self):
        leads = [
            lead("No Contact", "https://a.example", kind="distributor"),
            lead("Reachable", "https://b.example", kind="distributor", email="s@b.example"),
        ]
        result = await vendor_summary({"extracted_leads": leads})
        assert result["vendor_leads"][0].company_name == "Reachable"

    async def test_marks_the_session_done(self):
        assert (await vendor_summary({"extracted_leads": []}))["status"] == "done"


class TestDeduplication:
    async def test_collapses_the_same_domain(self):
        leads = [
            lead("Acme", "https://acme.example", kind="distributor"),
            lead("Acme Metals", "https://www.acme.example", kind="manufacturer"),
        ]
        result = await vendor_summary({"extracted_leads": leads})
        assert len(result["vendor_leads"]) == 1

    async def test_merges_fields_from_the_duplicate(self):
        leads = [
            lead("Acme", "https://acme.example", kind="manufacturer", email="s@acme.example"),
            lead("Acme", "https://acme.example", kind="distributor", phone="+44 121 555 0147"),
        ]
        [merged] = (await vendor_summary({"extracted_leads": leads}))["vendor_leads"]
        assert merged.email == "s@acme.example"
        assert merged.phone == "+44 121 555 0147"
        assert merged.kind == "manufacturer"  # the better-ranked lead wins


class TestClarificationGuard:
    def test_reverts_a_generalised_material_name(self):
        """"Custom 465" -> "Stainless Steel" destroys the vendor search."""
        original = spec()
        degraded = MaterialSpec(material_name="Stainless Steel", grade="UNS S46500")
        assert _protect_spec(original, degraded).material_name == "Custom 465"

    def test_allows_a_refinement_that_keeps_the_name(self):
        original = spec()
        refined = MaterialSpec(material_name="Custom 465 (UNS S46500)", form="bar")
        assert _protect_spec(original, refined).material_name == "Custom 465 (UNS S46500)"

    def test_never_erases_a_field_the_buyer_gave(self):
        original = spec(quantity=200, unit="KG", dimensions="Dia 2 inch")
        dropped = MaterialSpec(material_name="Custom 465", form="bar")
        result = _protect_spec(original, dropped)

        assert result.quantity == 200
        assert result.unit == "KG"
        assert result.dimensions == "Dia 2 inch"
        assert result.form == "bar"  # the genuine addition survives

    def test_keeps_standards_when_the_update_drops_them(self):
        original = spec(standards=["AMS 5936"])
        assert _protect_spec(original, spec()).standards == ["AMS 5936"]


class TestSupplierViability:
    """From a live 150 kg titanium enquiry: 4 of 12 results were not viable
    suppliers — NIST selling reference chips, and three retail sites. They are
    ranked last and labelled rather than dropped, so nothing is silently lost.
    """

    async def test_retail_and_non_suppliers_sink_below_real_vendors(self):
        leads = [
            lead("NIST", "https://shop.nist.gov", kind="not_a_supplier"),
            lead("Online Metals", "https://www.onlinemetals.com", kind="retail"),
            lead("Unclassified Co", "https://unknown.example", kind="unknown"),
            lead("Real Mill", "https://mill.example", kind="manufacturer"),
        ]
        result = await vendor_summary({"extracted_leads": leads})

        assert [v.company_name for v in result["vendor_leads"]] == [
            "Real Mill",
            "Unclassified Co",
            "Online Metals",
            "NIST",
        ]

    async def test_non_viable_vendors_are_kept_not_dropped(self):
        leads = [lead("NIST", "https://shop.nist.gov", kind="not_a_supplier")]
        result = await vendor_summary({"extracted_leads": leads})
        assert len(result["vendor_leads"]) == 1

    async def test_a_directory_never_outranks_a_real_vendor(self):
        """ThomasNet calling a page a "manufacturer" does not make ThomasNet a
        mill; an unclassified real vendor is the better lead."""
        leads = [
            lead("ThomasNet", "https://thomasnet.com", kind="manufacturer"),
            lead("Real Vendor", "https://realvendor.example", kind="unknown"),
        ]
        result = await vendor_summary({"extracted_leads": leads})
        assert result["vendor_leads"][0].company_name == "Real Vendor"


class TestCertificationRelevance:
    """A live titanium enquiry returned a vendor listing eight AMS numbers, none
    of which was AMS 4928 — the titanium spec from the research step. Listing
    another material's standards implies an approval the vendor may not hold."""

    def _spec(self):
        return MaterialSpec(material_name="Titanium", grade="Grade 5", standards=["ASTM B348"])

    def _research(self):
        return MaterialResearch(
            canonical_name="Ti-6Al-4V",
            designations=["UNS R56400", "AMS 4928", "ASTM B348"],
        )

    def test_drops_standards_belonging_to_other_materials(self):
        found = ["AMS 5629", "AMS 6414", "AMS 4928", "AMS 5643"]
        kept = _rank_certifications(found, self._spec(), self._research())
        assert kept == ["AMS 4928"]

    def test_always_keeps_company_level_approvals(self):
        """AS9100 and ISO 9001 describe how the vendor operates, so they matter
        whatever the material."""
        found = ["AS9100D", "ISO 9001:2015", "NADCAP", "AMS 5629"]
        kept = _rank_certifications(found, self._spec(), self._research())
        assert "AMS 5629" not in kept
        assert {"AS9100D", "ISO 9001:2015", "NADCAP"} <= set(kept)

    def test_quality_approvals_are_listed_first(self):
        found = ["AMS 4928", "AS9100D"]
        assert _rank_certifications(found, self._spec(), self._research())[0] == "AS9100D"

    def test_works_without_research(self):
        found = ["ASTM B348", "AMS 5629"]
        assert _rank_certifications(found, self._spec(), None) == ["ASTM B348"]
