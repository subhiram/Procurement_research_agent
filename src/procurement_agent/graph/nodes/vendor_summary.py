"""Deduplicate and rank the extracted leads into the final vendor list.

Ranking is deterministic. A global vendor search surfaces a lot of general
trading companies and directory listings alongside real manufacturers, and the
ordering rule for that is a stable business preference — manufacturer over
distributor over trader, verified contact over none — not a judgment call worth
spending free-tier tokens on per run.
"""

from __future__ import annotations

import logging
from urllib.parse import urlsplit

from procurement_agent.graph.state import SessionState, VendorLead

log = logging.getLogger(__name__)

#: Higher sorts first.
#:
#: `retail` and `not_a_supplier` sort below `unknown` deliberately: an
#: unclassified page might still be a real supplier, whereas a consumer shop or
#: a standards body is known not to serve an industrial quantity. They are kept
#: in the list, labelled, at the bottom.
KIND_RANK = {
    "manufacturer": 4,
    "distributor": 3,
    "trader": 2,
    "unknown": 1,
    "retail": 0,
    "not_a_supplier": -1,
}

DIRECTORY_DOMAINS = {
    "thomasnet.com",
    "alibaba.com",
    "indiamart.com",
    "made-in-china.com",
    "tradeindia.com",
    "exportersindia.com",
    "matmatch.com",
}


def _domain(url: str) -> str:
    host = urlsplit(url).netloc.casefold()
    return host[4:] if host.startswith("www.") else host


def _viability(lead: VendorLead) -> int:
    """How usable this vendor is for an industrial order. Higher is better.

    A directory listing is capped at the retail tier however it classifies
    itself: ThomasNet describing a page as a "manufacturer" does not make
    ThomasNet a mill, and a real vendor of unknown type is a better lead than
    an aggregator that merely lists one.
    """
    rank = KIND_RANK.get(lead.kind, 1)
    if _domain(lead.website) in DIRECTORY_DOMAINS:
        rank = min(rank, KIND_RANK["retail"])
    return rank


def _score(lead: VendorLead) -> tuple:
    """Sort key. Companies that can actually quote the order come first.

    A research citation outranks the commercial classification: a paper stating
    this company actually supplied the material is harder evidence than a page
    describing itself as a manufacturer.
    """
    return (
        # Viability first: retail, directories and non-suppliers sink, because
        # a buyer sourcing hundreds of kilograms cannot use any of them.
        _viability(lead),
        # A page that never names the material is weak evidence, whatever else
        # it claims about itself.
        1 if lead.mentions_material else 0,
        1 if lead.cited_in_research else 0,
        1 if lead.email else 0,
        1 if lead.phone else 0,
        len(lead.certifications_found),
    )


def _merge(primary: VendorLead, other: VendorLead) -> VendorLead:
    """Fold a duplicate into the better-ranked lead, filling only empty fields."""
    updates: dict[str, object] = {}
    if not primary.email and other.email:
        updates["email"] = other.email
    if not primary.phone and other.phone:
        updates["phone"] = other.phone
    if not primary.contact_name and other.contact_name:
        updates["contact_name"] = other.contact_name
    if not primary.country and other.country:
        updates["country"] = other.country
    if other.certifications_found:
        merged = list(dict.fromkeys([*primary.certifications_found, *other.certifications_found]))
        updates["certifications_found"] = merged
    # Provenance is additive: if either copy was cited in a paper, the vendor was.
    if other.cited_in_research and not primary.cited_in_research:
        updates["cited_in_research"] = True
    return primary.model_copy(update=updates) if updates else primary


async def vendor_summary(state: SessionState) -> dict:
    """Collapse duplicates by domain, rank, and publish the final list."""
    leads = list(state.get("extracted_leads", []))

    by_domain: dict[str, VendorLead] = {}
    for lead in sorted(leads, key=_score, reverse=True):
        domain = _domain(lead.website)
        if not domain:
            continue
        if domain in by_domain:
            by_domain[domain] = _merge(by_domain[domain], lead)
        else:
            by_domain[domain] = lead

    ranked = sorted(by_domain.values(), key=_score, reverse=True)

    reachable = sum(1 for lead in ranked if lead.email or lead.phone)
    log.info(
        "vendor_summary: %d leads -> %d unique vendors, %d with a verified contact",
        len(leads),
        len(ranked),
        reachable,
    )

    return {"vendor_leads": ranked, "status": "done"}
