"""The critical tests: a fabricated contact must never survive verification.

A hallucinated email here is well-formed, plausibly named after the company, and
flows straight into a live procurement workflow. These tests are the check that
it cannot.
"""

from __future__ import annotations

from procurement_agent.grounding import (
    email_is_grounded,
    phone_is_grounded,
    text_is_grounded,
    verify_lead,
    verify_lead_across,
    verify_leads,
)


class TestEmailGrounding:
    def test_accepts_email_present_verbatim(self, supplier_page):
        assert email_is_grounded("recruitment@precisionalloys.co.uk", supplier_page)

    def test_accepts_obfuscated_email(self, supplier_page):
        """`sales [at] x [dot] com` on the page is the same address as sales@x.com."""
        assert email_is_grounded("sales@precisionalloys.co.uk", supplier_page)

    def test_accepts_differing_case(self, supplier_page):
        assert email_is_grounded("Recruitment@PrecisionAlloys.co.uk", supplier_page)

    def test_rejects_plausible_but_absent_email(self, supplier_page):
        """The exact shape a hallucination takes: right domain, invented mailbox."""
        assert not email_is_grounded("info@precisionalloys.co.uk", supplier_page)

    def test_rejects_wrong_domain(self, supplier_page):
        assert not email_is_grounded("sales@precision-alloys.com", supplier_page)

    def test_rejects_empty(self, supplier_page):
        assert not email_is_grounded("", supplier_page)


class TestPhoneGrounding:
    def test_accepts_reformatted_phone(self, supplier_page):
        """Page says `+44 (0)121 555 0147`; the model may normalise it."""
        assert phone_is_grounded("+440121 555 0147", supplier_page)
        assert phone_is_grounded("4401215550147", supplier_page)

    def test_rejects_absent_phone(self, supplier_page):
        assert not phone_is_grounded("+44 121 555 9999", supplier_page)

    def test_rejects_short_number(self, supplier_page):
        """Short digit runs match dimensions and part codes by accident."""
        assert not phone_is_grounded("144", supplier_page)
        assert not phone_is_grounded("465", supplier_page)


class TestTextGrounding:
    def test_accepts_name_on_page(self, supplier_page):
        assert text_is_grounded("Margaret Ellison", supplier_page)

    def test_accepts_whitespace_variation(self, supplier_page):
        assert text_is_grounded("Margaret   Ellison", supplier_page)

    def test_rejects_absent_name(self, supplier_page):
        assert not text_is_grounded("James Whitfield", supplier_page)


class TestVerifyLead:
    def test_keeps_grounded_fields(self, supplier_page, lead_factory):
        lead = lead_factory(
            email="sales@precisionalloys.co.uk",
            phone="+44 (0)121 555 0147",
            contact_name="Margaret Ellison",
            certifications_found=["AS9100D", "ISO 9001:2015"],
        )
        result = verify_lead(lead, supplier_page)

        assert result.email == "sales@precisionalloys.co.uk"
        assert result.phone == "+44 (0)121 555 0147"
        assert result.contact_name == "Margaret Ellison"
        assert result.certifications_found == ["AS9100D", "ISO 9001:2015"]
        assert result.confidence_notes == []

    def test_drops_fabricated_email_and_explains(self, supplier_page, lead_factory):
        lead = lead_factory(email="purchasing@precisionalloys.co.uk")
        result = verify_lead(lead, supplier_page)

        assert result.email is None
        assert any("purchasing@" in note for note in result.confidence_notes)

    def test_drops_fabricated_contact_name(self, supplier_page, lead_factory):
        lead = lead_factory(
            email="sales@precisionalloys.co.uk", contact_name="James Whitfield"
        )
        result = verify_lead(lead, supplier_page)

        assert result.contact_name is None
        assert result.email == "sales@precisionalloys.co.uk"

    def test_drops_unverified_certifications(self, supplier_page, lead_factory):
        lead = lead_factory(certifications_found=["AS9100D", "ITAR registered"])
        result = verify_lead(lead, supplier_page)

        assert result.certifications_found == ["AS9100D"]

    def test_lead_survives_losing_every_contact_field(self, supplier_page, lead_factory):
        """A real company with no verifiable contact is still a useful result."""
        lead = lead_factory(email="fake@elsewhere.com", phone="+1 555 000 1111")
        result = verify_lead(lead, supplier_page)

        assert result.email is None
        assert result.phone is None
        assert result.company_name == "Precision Alloys Ltd"
        assert result.source_url == "https://precisionalloys.co.uk/custom-465"
        assert any("No verifiable contact" in n for n in result.confidence_notes)

    def test_never_mutates_the_input(self, supplier_page, lead_factory):
        lead = lead_factory(email="fake@elsewhere.com")
        verify_lead(lead, supplier_page)
        assert lead.email == "fake@elsewhere.com"


class TestVerifyLeads:
    def test_withholds_contacts_when_source_is_missing(self, lead_factory):
        """If we cannot check it, we do not publish it."""
        lead = lead_factory(email="sales@precisionalloys.co.uk", phone="+44 121 555 0147")
        [result] = verify_leads([lead], sources={})

        assert result.email is None
        assert result.phone is None
        assert any("could not be verified" in n for n in result.confidence_notes)

    def test_verifies_against_the_matching_source(self, supplier_page, lead_factory):
        lead = lead_factory(email="sales@precisionalloys.co.uk")
        [result] = verify_leads(
            [lead], sources={"https://precisionalloys.co.uk/custom-465": supplier_page}
        )
        assert result.email == "sales@precisionalloys.co.uk"


CONTACT_PAGE = """
Precision Alloys Ltd - Contact Us

Sales enquiries: sales@precisionalloys.co.uk
Head office: +44 (0)121 555 0147
Contact: Margaret Ellison, Sales Director
"""

PRODUCT_PAGE_NO_CONTACTS = """
Precision Alloys Ltd - Custom 465 (UNS S46500) round bar in H900.
Certifications: AS9100D, ISO 9001:2015.
See our contact page for enquiries.
"""


class TestVerifyLeadAcross:
    """Contacts usually come from a different page than the product page, so
    verification spans every page fetched — while still recording which one
    matched, or the guarantee weakens to 'somewhere on one of these pages'."""

    def _sources(self):
        return {
            "https://precisionalloys.co.uk/custom-465": PRODUCT_PAGE_NO_CONTACTS,
            "https://precisionalloys.co.uk/contact": CONTACT_PAGE,
        }

    def test_keeps_a_contact_found_on_the_contact_page(self, lead_factory):
        lead = lead_factory(
            email="sales@precisionalloys.co.uk", phone="+44 (0)121 555 0147"
        )
        result = verify_lead_across(lead, self._sources())

        assert result.email == "sales@precisionalloys.co.uk"
        assert result.phone == "+44 (0)121 555 0147"

    def test_records_which_page_the_contact_came_from(self, lead_factory):
        lead = lead_factory(email="sales@precisionalloys.co.uk")
        result = verify_lead_across(lead, self._sources())

        assert result.contact_source_url == "https://precisionalloys.co.uk/contact"
        # The product page it was found on is preserved separately.
        assert result.source_url == "https://precisionalloys.co.uk/custom-465"

    def test_drops_a_contact_absent_from_every_page(self, lead_factory):
        lead = lead_factory(email="purchasing@precisionalloys.co.uk")
        result = verify_lead_across(lead, self._sources())

        assert result.email is None
        assert result.contact_source_url is None
        assert any("not found on any page" in n for n in result.confidence_notes)

    def test_verifies_certifications_across_all_pages(self, lead_factory):
        """Certifications live on the product page, contacts on the contact
        page; both must survive."""
        lead = lead_factory(
            email="sales@precisionalloys.co.uk",
            certifications_found=["AS9100D", "ITAR registered"],
        )
        result = verify_lead_across(lead, self._sources())

        assert result.certifications_found == ["AS9100D"]

    def test_handles_a_single_source(self, lead_factory, supplier_page):
        lead = lead_factory(email="sales@precisionalloys.co.uk")
        result = verify_lead_across(
            lead, {"https://precisionalloys.co.uk/custom-465": supplier_page}
        )
        assert result.email == "sales@precisionalloys.co.uk"

    def test_no_sources_withholds_everything(self, lead_factory):
        lead = lead_factory(email="sales@precisionalloys.co.uk")
        assert verify_lead_across(lead, {}).email is None

    def test_never_mutates_the_input(self, lead_factory):
        lead = lead_factory(email="fake@elsewhere.com")
        verify_lead_across(lead, self._sources())
        assert lead.email == "fake@elsewhere.com"


class TestEmailShape:
    """Grounding checks presence; without a shape check any string on the page
    qualifies as an email. A real run shipped a Cloudflare link blob as a
    contact address exactly this way."""

    def test_rejects_a_non_email_that_is_present_on_the_page(self):
        blob = "[email protected]](/cdn-cgi/l/email-protection#4231232e)"
        page = f"Contact {blob} for sales"
        assert not email_is_grounded(blob, page)

    def test_rejects_a_bare_word_present_on_the_page(self):
        assert not email_is_grounded("sales", "our sales team is here")

    def test_still_accepts_a_genuine_address(self):
        page = "Reach us at sales@vendor.example today"
        assert email_is_grounded("sales@vendor.example", page)
