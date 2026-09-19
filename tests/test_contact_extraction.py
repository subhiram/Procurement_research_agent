"""End-to-end: a model that fabricates a contact must not get one published.

`test_grounding.py` checks the verifier in isolation. This checks that the node
actually routes its model output through it — the wiring, not just the function.
"""

from __future__ import annotations

import pytest

from procurement_agent.graph.nodes import contact_extraction as node_mod
from procurement_agent.graph.nodes.contact_extraction import contact_extraction
from procurement_agent.graph.state import (
    MaterialResearch,
    MaterialSpec,
    VendorCandidate,
)


class _StubModel:
    """Stands in for the `bulk` chain, returning whatever we tell it to."""

    def __init__(self, payload):
        self.payload = payload
        self.prompts: list[str] = []

    async def ainvoke(self, messages, *args, **kwargs):
        self.prompts.append(messages[-1].content)
        return self.payload


@pytest.fixture
def candidate(supplier_page):
    return VendorCandidate(
        company_name="Precision Alloys Ltd",
        url="https://precisionalloys.co.uk/custom-465",
        raw_content=supplier_page,
    )


async def test_fabricated_email_never_reaches_the_lead(monkeypatch, candidate):
    """The exact failure this app exists to prevent."""
    stub = _StubModel(
        node_mod.Extraction(
            email="purchasing@precisionalloys.co.uk",  # plausible, and absent
            phone="+44 121 555 9999",  # plausible, and absent
            contact_name="James Whitfield",  # plausible, and absent
            kind="distributor",
        )
    )
    monkeypatch.setattr(node_mod, "get_model", lambda *a, **k: stub)

    [lead] = (await contact_extraction({"candidate": candidate}))["extracted_leads"]

    assert lead.email is None
    assert lead.phone is None
    assert lead.contact_name is None
    # The lead itself survives, with its provenance and an explanation.
    assert lead.company_name == "Precision Alloys Ltd"
    assert lead.source_url == "https://precisionalloys.co.uk/custom-465"
    assert len(lead.confidence_notes) >= 3


async def test_genuine_contact_is_kept(monkeypatch, candidate):
    stub = _StubModel(
        node_mod.Extraction(
            email="sales@precisionalloys.co.uk",  # obfuscated on the page
            phone="+44 (0)121 555 0147",
            contact_name="Margaret Ellison",
            kind="distributor",
            country="United Kingdom",
        )
    )
    monkeypatch.setattr(node_mod, "get_model", lambda *a, **k: stub)

    [lead] = (await contact_extraction({"candidate": candidate}))["extracted_leads"]

    assert lead.email == "sales@precisionalloys.co.uk"
    assert lead.phone == "+44 (0)121 555 0147"
    assert lead.contact_name == "Margaret Ellison"
    assert lead.kind == "distributor"
    assert "AS9100D" in lead.certifications_found


async def test_prompt_stays_small_on_a_realistic_page(monkeypatch, supplier_page):
    """The prompt carries the regex digest, not the page. This is the budget.

    Sending the page itself would consume a full minute of Groq's 6,000 TPM per
    vendor. Real supplier pages are mostly navigation, product tables and legal
    boilerplate, so the digest should be a small fraction of the source.
    """
    boilerplate = (
        "Home About Products Services Quality Contact Terms Privacy Cookies. "
        "Grade 304 316L 321 410 420 431 17-4PH 15-5PH Inconel 625 718 Monel 400. "
        "Delivery worldwide. Cut to length. Saw cutting. Waterjet. "
    ) * 200
    page = supplier_page + "\n" + boilerplate

    candidate = VendorCandidate(
        company_name="Precision Alloys Ltd",
        url="https://precisionalloys.co.uk/custom-465",
        raw_content=page,
    )
    stub = _StubModel(node_mod.Extraction())
    monkeypatch.setattr(node_mod, "get_model", lambda *a, **k: stub)

    await contact_extraction({"candidate": candidate})
    prompt = stub.prompts[0]

    assert "EMAIL CANDIDATES" in prompt
    # The whole point: an order-of-magnitude reduction before the model sees it.
    assert len(prompt) < len(page) / 10
    assert "Waterjet" not in prompt


async def test_thin_page_emits_a_lead_without_calling_the_model(monkeypatch):
    called = False

    def _fail(*a, **k):
        nonlocal called
        called = True
        raise AssertionError("model should not be called for an empty page")

    monkeypatch.setattr(node_mod, "get_model", _fail)
    candidate = VendorCandidate(
        company_name="Thin Co", url="https://thin.example/x", raw_content="too short"
    )

    [lead] = (await contact_extraction({"candidate": candidate}))["extracted_leads"]

    assert not called
    assert lead.email is None
    assert lead.source_url == "https://thin.example/x"


async def test_model_failure_degrades_to_a_bare_lead(monkeypatch, candidate):
    """One unreadable page must not take down the whole research run."""

    class _Boom:
        async def ainvoke(self, *a, **k):
            raise RuntimeError("provider exploded")

    monkeypatch.setattr(node_mod, "get_model", lambda *a, **k: _Boom())

    [lead] = (await contact_extraction({"candidate": candidate}))["extracted_leads"]

    assert lead.email is None
    assert any("failed" in note for note in lead.confidence_notes)


class TestCertificationsAndRelevance:
    """Distributors publish their whole stock list; one real page yielded 52
    AMS numbers, which says what they carry, not what they are approved for."""

    async def test_certification_list_is_capped(self, monkeypatch, supplier_page):
        dump = " ".join(f"AMS {5500 + i}" for i in range(40))
        candidate = VendorCandidate(
            company_name="Stock Everything Ltd",
            url="https://stock.example/x",
            raw_content=supplier_page + "\n" + dump,
        )
        monkeypatch.setattr(
            node_mod, "get_model", lambda *a, **k: _StubModel(node_mod.Extraction())
        )

        [lead] = (await contact_extraction({"candidate": candidate}))["extracted_leads"]

        assert len(lead.certifications_found) <= node_mod.MAX_CERTIFICATIONS

    async def test_quality_approvals_survive_the_cap(self, monkeypatch, supplier_page):
        """AS9100 and ISO 9001 matter to a defense buyer far more than an
        arbitrary entry from a stock list."""
        dump = " ".join(f"AMS {5500 + i}" for i in range(40))
        candidate = VendorCandidate(
            company_name="Stock Everything Ltd",
            url="https://stock.example/x",
            raw_content=supplier_page + "\n" + dump,
        )
        monkeypatch.setattr(
            node_mod, "get_model", lambda *a, **k: _StubModel(node_mod.Extraction())
        )

        [lead] = (await contact_extraction({"candidate": candidate}))["extracted_leads"]
        found = " ".join(lead.certifications_found).upper()

        assert "AS9100" in found
        assert "ISO 9001" in found

    async def test_flags_a_page_that_never_names_the_material(self, monkeypatch):
        candidate = VendorCandidate(
            company_name="Generic Stainless Co",
            url="https://generic.example/x",
            raw_content="We supply 304 and 316L stainless in all forms. " * 20,
        )
        monkeypatch.setattr(
            node_mod, "get_model", lambda *a, **k: _StubModel(node_mod.Extraction())
        )
        state = {
            "candidate": candidate,
            "material_spec": MaterialSpec(material_name="Custom 465"),
        }

        [lead] = (await contact_extraction(state))["extracted_leads"]

        assert lead.mentions_material is False
        assert any("does not name" in n for n in lead.confidence_notes)

    async def test_does_not_flag_a_page_that_names_the_material(
        self, monkeypatch, supplier_page
    ):
        candidate = VendorCandidate(
            company_name="Precision Alloys Ltd",
            url="https://precisionalloys.co.uk/custom-465",
            raw_content=supplier_page,
        )
        monkeypatch.setattr(
            node_mod, "get_model", lambda *a, **k: _StubModel(node_mod.Extraction())
        )
        state = {
            "candidate": candidate,
            "material_spec": MaterialSpec(material_name="Custom 465"),
        }

        [lead] = (await contact_extraction(state))["extracted_leads"]

        assert lead.mentions_material is True


CONTACT_PAGE_TEXT = """
Precision Alloys Ltd - Contact Us
Sales enquiries: sales@precisionalloys.co.uk
Head office: Tel: +44 (0)121 555 0147
"""

PRODUCT_PAGE_TEXT = """
Precision Alloys Ltd stocks Custom 465 (UNS S46500) round bar in H900.
Certifications: AS9100D, ISO 9001:2015 and EN 10204 3.1 certificates.
Diameters 0.25 in to 6.00 in. See our contact page to enquire.
"""


class TestContactPageFollowing:
    """The phase-2 quality win: a product page with no contacts on it should
    still yield a reachable vendor, via the vendor's own contact page."""

    def _patch_pages(self, monkeypatch, pages):
        async def _fake(product_url):
            return pages

        monkeypatch.setattr(node_mod, "_fetch_contact_pages", _fake)

    async def test_recovers_a_contact_from_the_contact_page(self, monkeypatch):
        self._patch_pages(
            monkeypatch,
            {"https://precisionalloys.co.uk/contact": CONTACT_PAGE_TEXT},
        )
        monkeypatch.setattr(
            node_mod,
            "get_model",
            lambda *a, **k: _StubModel(
                node_mod.Extraction(
                    email="sales@precisionalloys.co.uk",
                    phone="+44 (0)121 555 0147",
                    kind="distributor",
                )
            ),
        )
        candidate = VendorCandidate(
            company_name="Precision Alloys Ltd",
            url="https://precisionalloys.co.uk/custom-465",
            raw_content=PRODUCT_PAGE_TEXT,
        )

        [lead] = (await contact_extraction({"candidate": candidate}))["extracted_leads"]

        assert lead.email == "sales@precisionalloys.co.uk"
        assert lead.phone == "+44 (0)121 555 0147"
        # Provenance: the contact came from the contact page, not the product page.
        assert lead.contact_source_url == "https://precisionalloys.co.uk/contact"
        assert lead.source_url == "https://precisionalloys.co.uk/custom-465"

    async def test_digest_labels_each_page(self, monkeypatch):
        self._patch_pages(
            monkeypatch,
            {"https://precisionalloys.co.uk/contact": CONTACT_PAGE_TEXT},
        )
        stub = _StubModel(node_mod.Extraction())
        monkeypatch.setattr(node_mod, "get_model", lambda *a, **k: stub)
        candidate = VendorCandidate(
            company_name="Precision Alloys Ltd",
            url="https://precisionalloys.co.uk/custom-465",
            raw_content=PRODUCT_PAGE_TEXT,
        )

        await contact_extraction({"candidate": candidate})
        prompt = stub.prompts[0]

        assert "--- from https://precisionalloys.co.uk/custom-465 ---" in prompt
        assert "--- from https://precisionalloys.co.uk/contact ---" in prompt

    async def test_relevance_is_judged_on_the_product_page_only(self, monkeypatch):
        """A contact page never names the material, so including it in the
        relevance check would make every vendor look irrelevant."""
        self._patch_pages(
            monkeypatch,
            {"https://precisionalloys.co.uk/contact": CONTACT_PAGE_TEXT},
        )
        monkeypatch.setattr(
            node_mod, "get_model", lambda *a, **k: _StubModel(node_mod.Extraction())
        )
        candidate = VendorCandidate(
            company_name="Precision Alloys Ltd",
            url="https://precisionalloys.co.uk/custom-465",
            raw_content=PRODUCT_PAGE_TEXT,
        )
        state = {
            "candidate": candidate,
            "material_spec": MaterialSpec(material_name="Custom 465"),
        }

        [lead] = (await contact_extraction(state))["extracted_leads"]

        assert lead.mentions_material is True

    async def test_a_contact_page_fetch_failure_is_not_fatal(self, monkeypatch):
        """A crashing crawler must degrade to product-page-only extraction, not
        fail the Send() branch and take the whole research run with it."""

        async def _boom(product_url):
            raise RuntimeError("browser died")

        monkeypatch.setattr(node_mod, "_fetch_contact_pages", _boom)
        monkeypatch.setattr(
            node_mod, "get_model", lambda *a, **k: _StubModel(node_mod.Extraction())
        )
        candidate = VendorCandidate(
            company_name="Precision Alloys Ltd",
            url="https://precisionalloys.co.uk/custom-465",
            raw_content=PRODUCT_PAGE_TEXT,
        )

        [lead] = (await contact_extraction({"candidate": candidate}))["extracted_leads"]

        assert lead.company_name == "Precision Alloys Ltd"
        assert "AS9100D" in lead.certifications_found


class TestSupplierClassification:
    """URL-level backstop for the model's classification. Host facts are
    certain in a way a judgement call is not: a .gov domain does not sell
    titanium bar, whatever the page claims."""

    def test_government_and_standards_hosts_are_not_suppliers(self):
        for url in (
            "https://shop.nist.gov/ccrz__ProductDetails?sku=173c",
            "https://www.matweb.com/search/datasheet.aspx",
            "https://example.edu/materials",
        ):
            kind, note = node_mod._classify_by_url(url, "manufacturer")
            assert kind == "not_a_supplier", url
            assert note

    def test_video_and_social_platforms_are_not_suppliers(self):
        """A live run returned a YouTube product video as a vendor."""
        for url in (
            "https://www.youtube.com/watch?v=qKGLZFCwYdk",
            "https://www.linkedin.com/company/some-metals",
            "https://www.amazon.com/dp/B000000",
        ):
            kind, _ = node_mod._classify_by_url(url, "distributor")
            assert kind == "not_a_supplier", url

    def test_a_retail_url_downgrades_an_unclassified_page(self):
        kind, note = node_mod._classify_by_url(
            "https://store.tmstitanium.com/products/titanium-bar-rod", "unknown"
        )
        assert kind == "retail"
        assert "quantity" in note

    def test_per_unit_pricing_overrides_the_model(self):
        """A page selling bar by the inch cannot fill a 150 kg order, whatever
        else the company does - this was classed a manufacturer and ranked 2nd."""
        kind, note = node_mod._classify_by_url(
            "https://racetechtitanium.com/product/3-16-titanium-bar-sold-by-the-inchgrade-5",
            "manufacturer",
        )
        assert kind == "retail" and note

    def test_the_model_calling_it_retail_is_respected(self):
        kind, note = node_mod._classify_by_url("https://anything.example", "retail")
        assert kind == "retail" and note

    def test_a_real_distributor_is_not_downgraded_by_its_url(self):
        """Only ever downgrades. A genuine distributor keeps its class even
        with a /products/ path."""
        kind, note = node_mod._classify_by_url(
            "https://www.flightmetals.com/product/ams-4928-titanium-6al-4v-bar",
            "distributor",
        )
        assert kind == "distributor"
        assert note is None

    def test_an_ordinary_vendor_url_is_left_alone(self):
        kind, note = node_mod._classify_by_url(
            "https://www.altempalloys.com/titanium.html", "manufacturer"
        )
        assert kind == "manufacturer" and note is None


class TestMaterialRelevance:
    """Matched on identifying designations, not the buyer's phrasing.

    A live run parsed material_name as "Titanium Grade 5 round bar". That exact
    string appears on no vendor page, so every result was flagged irrelevant and
    the warning became noise — hiding a genuine miss, a Grade 7 page for a
    Grade 5 enquiry.
    """

    SPEC = MaterialSpec(material_name="Titanium Grade 5 round bar", form="bar")
    RESEARCH = MaterialResearch(
        canonical_name="Ti-6Al-4V",
        designations=["UNS R56400", "AMS 4928", "ASTM B348"],
    )

    def _check(self, text):
        return node_mod._mentions_material(text, self.SPEC, self.RESEARCH)

    def test_form_words_are_stripped_from_the_material_name(self):
        assert self._check("We supply Titanium Grade 5 in all sizes")

    def test_matches_a_designation_instead_of_the_buyers_phrasing(self):
        assert self._check("Ti-6Al-4V alloy bar stock available")
        assert self._check("UNS R56400 round bar")

    def test_punctuation_variants_still_match(self):
        """Ti-6Al-4V, Ti 6Al 4V and Ti6Al4V are the same alloy."""
        assert self._check("Ti6Al4V bar in stock")

    def test_word_order_does_not_matter(self):
        """Vendors write both "Grade 5 Titanium" and "Titanium Grade 5"."""
        assert self._check("Grade 5 Titanium Round Bar 6Al-4V")

    def test_a_family_standard_alone_is_not_a_match(self):
        """ASTM B348 covers every titanium bar grade, so citing it proves
        nothing about the grade on offer - this passed Grade 7 pages."""
        assert not self._check(
            "Titanium Grade 7 UNS R52400 round bars, ASTM B348 compliant"
        )

    def test_a_digit_inside_another_code_is_not_a_match(self):
        """The "5" in R52400 must not satisfy a search for grade 5."""
        assert not self._check("Titanium Grade 7 UNS R52400 bars")

    def test_flags_a_neighbouring_grade(self):
        """Grade 7 is Ti-0.2Pd, a different alloy entirely."""
        assert not self._check("Titanium Grade 7 (Ti-0.2Pd) round bars supplier")

    def test_flags_a_generic_material_page(self):
        assert not self._check("We stock titanium products of all kinds")

    def test_no_spec_means_no_opinion(self):
        assert node_mod._mentions_material("anything", None, None)
