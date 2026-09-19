"""Regex pre-extraction.

The precision bar here is high in both directions. Supplier pages are dense with
digit runs that look like phone numbers — alloy grade lists, dimension tables,
AMS numbers, company registration numbers — and a false positive means outreach
to a real stranger.
"""

from __future__ import annotations

from procurement_agent.extraction.patterns import (
    build_candidate_digest,
    deobfuscate,
    find_certifications,
    find_emails,
    find_phones,
)


class TestEmails:
    def test_finds_plain_address(self, supplier_page):
        assert "recruitment@precisionalloys.co.uk" in {
            c.value for c in find_emails(supplier_page)
        }

    def test_deobfuscates(self, supplier_page):
        assert "sales@precisionalloys.co.uk" in {
            c.value for c in find_emails(supplier_page)
        }

    def test_deobfuscate_handles_bracket_forms(self):
        assert deobfuscate("a [at] b [dot] com") == "a@b.com"
        assert deobfuscate("a (at) b (dot) com") == "a@b.com"

    def test_skips_image_and_asset_filenames(self):
        text = "logo@2x.png sprite@3x.png real@company.com"
        assert {c.value for c in find_emails(text)} == {"real@company.com"}

    def test_skips_placeholder_domains(self):
        text = "you@example.com someone@yourdomain.com actual@vendor.de"
        assert {c.value for c in find_emails(text)} == {"actual@vendor.de"}

    def test_deduplicates_case_insensitively(self):
        text = "Sales@Vendor.com and sales@vendor.com"
        assert len(find_emails(text)) == 1

    def test_attaches_context(self, supplier_page):
        candidate = next(
            c for c in find_emails(supplier_page) if c.value.startswith("recruitment")
        )
        assert "Careers" in candidate.context


class TestPhones:
    def test_finds_number_with_country_code(self, supplier_page):
        digits = {"".join(ch for ch in c.value if ch.isdigit()) for c in find_phones(supplier_page)}
        assert "4401215550147" in digits

    def test_rejects_alloy_grade_list(self):
        """The false positive that motivated keyword anchoring."""
        text = "We stock grades 304 316L 321 410 420 431 in bar and sheet."
        assert find_phones(text) == []

    def test_rejects_dimension_table(self):
        text = "Diameters 0.25 0.50 0.75 1.00 1.25 1.50 in stock."
        assert find_phones(text) == []

    def test_rejects_bare_company_registration_number(self):
        text = "Registered in England, company number 04412299."
        assert find_phones(text) == []

    def test_accepts_keyword_anchored_number(self):
        text = "Tel: 0121 555 0147 for enquiries"
        assert len(find_phones(text)) == 1

    def test_accepts_parenthesised_area_code(self):
        text = "Reach us on (0121) 555 0147 during office hours"
        assert len(find_phones(text)) == 1

    def test_rejects_repeated_digits(self):
        text = "Tel: 000 000 0000"
        assert find_phones(text) == []

    def test_deduplicates_by_digits(self):
        text = "Tel: +44 121 555 0147 or phone +44-121-555-0147"
        assert len(find_phones(text)) == 1


class TestCertifications:
    def test_finds_aerospace_and_quality_standards(self, supplier_page):
        found = {c.upper() for c in find_certifications(supplier_page)}
        assert "AS9100D" in found
        assert "ISO 9001:2015" in found
        assert "NADCAP" in found

    def test_returns_nothing_when_absent(self):
        assert find_certifications("We sell metal bars.") == []


class TestDigest:
    def test_includes_candidates_with_context(self, supplier_page):
        digest = build_candidate_digest(supplier_page)
        assert "EMAIL CANDIDATES" in digest
        assert "sales@precisionalloys.co.uk" in digest

    def test_reports_absence_explicitly(self):
        """The model must be told there were no candidates, not handed a blank."""
        digest = build_candidate_digest("Just some prose with no contacts at all.")
        assert "No contact candidates" in digest

    def test_is_far_smaller_than_the_page(self, supplier_page):
        page = supplier_page + ("Navigation boilerplate and product tables. " * 500)
        assert len(build_candidate_digest(page)) < len(page) / 10


class TestCloudflareEmails:
    """Cloudflare renders protected addresses as the literal text
    "[email protected]" with the real address XOR-encoded in the link. A real
    run emitted that raw blob as a contact address, because the string genuinely
    appears on the page — presence passed, shape was never checked."""

    RAW = (
        "Contact [email protected]](/cdn-cgi/l/email-protection"
        "#4231232e273102302321273627212a362b36232c2b372f6c212d2f) for sales"
    )

    def test_decodes_to_the_real_address(self):
        assert "sales@racetechtitanium.com" in deobfuscate(self.RAW)

    def test_extracts_the_decoded_address(self):
        assert "sales@racetechtitanium.com" in {c.value for c in find_emails(self.RAW)}

    def test_leaves_malformed_payloads_alone(self):
        text = "see /cdn-cgi/l/email-protection#zzzz for details"
        assert deobfuscate(text) == text

    def test_does_not_invent_an_address_from_undecodable_input(self):
        """A payload that decodes to junk must not become a contact."""
        assert find_emails("/cdn-cgi/l/email-protection#0102030405") == []
