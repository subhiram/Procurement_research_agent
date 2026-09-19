"""Verify that every emitted contact detail actually appears in its source page.

This is the last line of defence against the failure that matters most in this
app. A hallucinated email address does not look wrong: it is well-formed, it is
plausibly named after the company, and it flows straight into a procurement
workflow attached to a defense contract. Nobody notices until outreach silently
goes nowhere, or worse, reaches a real stranger.

The regex-first extraction in `extraction.patterns` already makes invention
structurally unlikely, so this module is a genuinely independent second check —
it re-verifies the model's output against the source text rather than trusting
the pipeline that produced it. Prompt instructions are not part of the defence;
they cannot be, because the thing being defended against is the model ignoring
them.

Anything that fails verification is dropped and recorded in `confidence_notes`.
A lead that keeps only its company name and source URL is still useful. A lead
with a fabricated email is worse than no lead at all.
"""

from __future__ import annotations

import logging
import re

from procurement_agent.extraction.patterns import EMAIL_RE, deobfuscate
from procurement_agent.graph.state import VendorLead

log = logging.getLogger(__name__)


def _norm_text(value: str) -> str:
    """Casefold and collapse whitespace for tolerant substring comparison."""
    return " ".join(value.split()).casefold()


def _norm_email(value: str) -> str:
    return deobfuscate(value).strip().strip(".<>()[]").casefold()


def _digits(value: str) -> str:
    return re.sub(r"\D", "", value)


def email_is_grounded(email: str, source: str) -> bool:
    """True when `email` is a real address AND appears in `source`.

    The shape check is not redundant. Grounding otherwise verifies *presence*
    only, so any string printed on the page passes — and a real run emitted
    `[email protected]](/cdn-cgi/l/email-protection#4231...)` as a contact
    address, because that literal text is on the page. Presence is necessary
    but not sufficient: it must also be an email.
    """
    if not email:
        return False
    if not EMAIL_RE.fullmatch(_norm_email(email)):
        return False
    return _norm_email(email) in _norm_email(source)


def phone_is_grounded(phone: str, source: str) -> bool:
    """True when `phone`'s digits appear in `source`'s digits.

    Compares digits only, so `+1 (555) 010-1234` matches a page that wrote
    `15550101234`. Requires at least 7 digits: shorter runs match incidentally
    against part numbers and dimensions, which would defeat the whole check.
    """
    wanted = _digits(phone)
    if len(wanted) < 7:
        return False
    return wanted in _digits(source)


def text_is_grounded(value: str, source: str) -> bool:
    """True when `value` appears in `source` up to whitespace and case."""
    if not value:
        return False
    return _norm_text(value) in _norm_text(source)


def verify_lead(lead: VendorLead, source: str) -> VendorLead:
    """Return a copy of `lead` with every ungrounded contact field removed.

    Never raises and never drops the lead itself — a real company with no
    verifiable contact details is still a useful result, as long as the record
    says so plainly.
    """
    notes = list(lead.confidence_notes)
    updates: dict[str, object] = {}

    if lead.email and not email_is_grounded(lead.email, source):
        notes.append(
            f"Dropped email {lead.email!r}: not found in the fetched page content."
        )
        log.warning(
            "grounding: dropped ungrounded email for %s (%s)",
            lead.company_name,
            lead.source_url,
        )
        updates["email"] = None

    if lead.phone and not phone_is_grounded(lead.phone, source):
        notes.append(
            f"Dropped phone {lead.phone!r}: not found in the fetched page content."
        )
        log.warning(
            "grounding: dropped ungrounded phone for %s (%s)",
            lead.company_name,
            lead.source_url,
        )
        updates["phone"] = None

    if lead.contact_name and not text_is_grounded(lead.contact_name, source):
        notes.append(
            f"Dropped contact name {lead.contact_name!r}: not found in the fetched "
            "page content."
        )
        updates["contact_name"] = None

    kept_certs = [c for c in lead.certifications_found if text_is_grounded(c, source)]
    if len(kept_certs) != len(lead.certifications_found):
        dropped = set(lead.certifications_found) - set(kept_certs)
        notes.append(f"Dropped unverified certifications: {sorted(dropped)}.")
        updates["certifications_found"] = kept_certs

    if not any(
        [
            updates.get("email", lead.email),
            updates.get("phone", lead.phone),
        ]
    ):
        notes.append(
            "No verifiable contact details on this page - check the source URL directly."
        )

    updates["confidence_notes"] = notes
    return lead.model_copy(update=updates)


def verify_lead_across(lead: VendorLead, sources: dict[str, str]) -> VendorLead:
    """Verify a lead whose contacts may come from any of several fetched pages.

    Contacts usually live on a separate contact page from the product page the
    search returned, so verification has to span every page fetched for this
    vendor. The risk in doing that naively is that the guarantee weakens from
    "this email is on this page" to "this email is on one of these pages
    somewhere" — so the page that actually matched is recorded in
    `contact_source_url` rather than discarded.

    Falls back to single-source behaviour when only one page was fetched.
    """
    if not sources:
        return verify_lead(lead, "")

    notes = list(lead.confidence_notes)
    updates: dict[str, object] = {}
    matched_url: str | None = None

    if lead.email:
        matched_url = _first_matching(
            sources, lambda text: email_is_grounded(lead.email or "", text)
        )
        if matched_url is None:
            notes.append(
                f"Dropped email {lead.email!r}: not found on any page fetched for "
                "this vendor."
            )
            log.warning(
                "grounding: dropped ungrounded email for %s (%s)",
                lead.company_name,
                lead.source_url,
            )
            updates["email"] = None

    if lead.phone:
        phone_url = _first_matching(
            sources, lambda text: phone_is_grounded(lead.phone or "", text)
        )
        if phone_url is None:
            notes.append(
                f"Dropped phone {lead.phone!r}: not found on any page fetched for "
                "this vendor."
            )
            updates["phone"] = None
        else:
            matched_url = matched_url or phone_url

    if lead.contact_name:
        name_url = _first_matching(
            sources, lambda text: text_is_grounded(lead.contact_name or "", text)
        )
        if name_url is None:
            notes.append(
                f"Dropped contact name {lead.contact_name!r}: not found on any page "
                "fetched for this vendor."
            )
            updates["contact_name"] = None

    combined = "\n".join(sources.values())
    kept_certs = [c for c in lead.certifications_found if text_is_grounded(c, combined)]
    if len(kept_certs) != len(lead.certifications_found):
        dropped = set(lead.certifications_found) - set(kept_certs)
        notes.append(f"Dropped unverified certifications: {sorted(dropped)}.")
        updates["certifications_found"] = kept_certs

    if not any([updates.get("email", lead.email), updates.get("phone", lead.phone)]):
        notes.append(
            "No verifiable contact details on the pages fetched - check the source "
            "URL directly."
        )
    elif matched_url:
        updates["contact_source_url"] = matched_url

    updates["confidence_notes"] = notes
    return lead.model_copy(update=updates)


def _first_matching(sources: dict[str, str], predicate) -> str | None:
    """URL of the first page satisfying `predicate`, or None."""
    for url, text in sources.items():
        if predicate(text):
            return url
    return None


def verify_leads(leads: list[VendorLead], sources: dict[str, str]) -> list[VendorLead]:
    """Verify each lead against the page text it was extracted from.

    A lead whose source text is missing gets all contact fields stripped: if we
    cannot check it, we do not publish it.
    """
    verified: list[VendorLead] = []
    for lead in leads:
        source = sources.get(lead.source_url)
        if source is None:
            verified.append(
                lead.model_copy(
                    update={
                        "email": None,
                        "phone": None,
                        "contact_name": None,
                        "certifications_found": [],
                        "confidence_notes": [
                            *lead.confidence_notes,
                            "Source page content unavailable, so contact details "
                            "could not be verified and were withheld.",
                        ],
                    }
                )
            )
            continue
        verified.append(verify_lead(lead, source))
    return verified
