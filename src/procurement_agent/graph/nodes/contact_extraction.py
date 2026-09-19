"""Turn one candidate page into one verified VendorLead.

This node is the token hog of the whole graph — it is the only one that touches
full page text, once per vendor — so it runs on the `bulk` chain, which puts
local Ollama first. Locally it is free and unmetered; on Groq's 6,000 tokens per
minute it would consume an entire minute's budget per vendor.

The page never reaches the model. `extraction.patterns` reduces it to a short
digest of regex-found candidates with surrounding context, and the model is
asked only to judge which candidate is the right procurement contact. That keeps
the payload small and makes fabrication structurally difficult — and
`grounding.verify_lead` then checks the output against the page anyway.
"""

from __future__ import annotations

import logging
import re
from typing import Any
from urllib.parse import urlsplit

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from procurement_agent.config import get_settings
from procurement_agent.crawl.contact_links import find_contact_links, guess_contact_urls
from procurement_agent.crawl.fetcher import get_fetcher
from procurement_agent.designations import is_broad_standard, normalise
from procurement_agent.extraction.patterns import (
    build_candidate_digest,
    find_certifications,
)
from procurement_agent.graph.context import allocation, thread_id_of
from procurement_agent.graph.state import (
    MaterialResearch,
    MaterialSpec,
    VendorCandidate,
    VendorKind,
    VendorLead,
)
from procurement_agent.grounding import verify_lead_across
from procurement_agent.llm.models import get_model
from procurement_agent.search.client import BudgetExhausted, result_text, search
from searchroute import Depth

log = logging.getLogger(__name__)

#: Hits for the confirmation search, which runs only for a candidate whose
#: page failed the material check. Its credits come from a single allocation
#: shared by the whole fan-out, not one per candidate.
CONFIRMATION_RESULTS = 5

#: Below this, the page has no usable text and we emit a bare lead rather than
#: spending a model call on nothing.
MIN_CONTENT = 200

#: Cap on page text used for certification scanning and grounding. Generous,
#: because this is regex work, not tokens.
MAX_SOURCE_CHARS = 120_000

#: Cap on certifications reported per vendor.
#:
#: Distributors publish their entire stock list — one real page yielded 52 AMS
#: numbers — which says what they carry, not what they are approved for. A long
#: dump buries the few standards that matter for this enquiry, so the list is
#: trimmed with the material's own standards preferred.
MAX_CERTIFICATIONS = 8

SYSTEM = """You are extracting a procurement contact for an industrial buyer.

You are given a company's page and a list of contact candidates that were found \
VERBATIM on that page by a regex scan, each with surrounding context.

Your job is selection and classification, NOT recall:

1. Choose the single best email for a raw-material purchasing enquiry. Prefer a \
sales, enquiry, or info address over careers, press, privacy, webmaster or \
support. If none of the candidates is suitable, return null.
2. Choose the best general or sales phone number, or null.
3. Choose the contact person's name ONLY if the page names one alongside the \
chosen contact. Otherwise null.
4. Classify the company:
   - "manufacturer": mills, forges or produces the material itself.
   - "distributor": stocks and resells industrial quantities.
   - "trader": general trading or brokerage company.
   - "retail": an online shop selling small quantities to consumers or \
hobbyists. Tell-tale signs are per-piece or per-inch pricing, "add to cart", \
shipping calculators, or offcuts and sample sizes. A buyer sourcing hundreds of \
kilograms cannot use these.
   - "not_a_supplier": the page does not sell the material at all. Standards \
bodies, universities, research institutes, government sites, encyclopaedias, \
directories, news articles and datasheet libraries all belong here.
   - "unknown": the page genuinely does not say.
5. country: the company's country if the page states it, else null.

CRITICAL: you may only return an email or phone that appears EXACTLY in the \
candidate list above. Do not correct, complete, normalise, or infer one — not \
even if a candidate looks like a typo, and not even if you can guess the \
company's address format from its domain. If the right contact is not in the \
list, return null. A null field is a correct answer; an invented one is not.
"""


#: Named without a leading underscore deliberately. LangChain derives the tool
#: name from the class, and Mistral rejected the result with "Unknown tool type:
#: 'Extraction'. Available tools: _Extraction" - the underscore is stripped
#: somewhere in the round trip and the name then fails to match. It cost a
#: ladder step on every Mistral call before this was found, which is invisible
#: unless you read the recorded attempts on a trace.
class Extraction(BaseModel):
    email: str | None = Field(default=None)
    phone: str | None = Field(default=None)
    contact_name: str | None = Field(default=None)
    kind: VendorKind = "unknown"
    country: str | None = Field(default=None)


def _domain(url: str) -> str:
    host = urlsplit(url).netloc.casefold()
    return host[4:] if host.startswith("www.") else host


def _website(url: str) -> str:
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}" if parts.netloc else url


#: Company-level approvals. Always relevant to a defense buyer, whatever the
#: material, because they describe how the vendor operates rather than what
#: they stock.
_QUALITY_MARKERS = (
    "as9100", "iso 9001", "iso9001", "iso 14001", "nadcap", "itar", "dfars",
    "10204",
)


def _rank_certifications(
    found: list[str],
    spec: MaterialSpec | None,
    research: MaterialResearch | None = None,
) -> list[str]:
    """Keep company approvals and this material's own standards. Drop the rest.

    Distributors publish their entire catalogue of specifications. On a live
    titanium enquiry one vendor listed eight AMS numbers, none of which was
    AMS 4928 — the titanium spec from the research step. Listing another
    material's standards under this material is not merely noisy, it implies an
    approval the vendor may not hold for what is being bought.

    So material specs are kept only when they match the designations the
    research step established for *this* material.
    """
    relevant = {s.casefold() for s in (spec.standards if spec else []) if s}
    if spec and spec.grade:
        relevant.add(spec.grade.casefold())
    if research:
        relevant.update(d.casefold() for d in research.designations if d)

    def is_quality(cert: str) -> bool:
        lowered = cert.casefold()
        return any(marker in lowered for marker in _QUALITY_MARKERS)

    def is_for_this_material(cert: str) -> bool:
        lowered = " ".join(cert.casefold().split())
        return any(
            lowered in wanted or wanted in lowered
            for wanted in (" ".join(w.split()) for w in relevant)
        )

    kept = [
        cert
        for cert in dict.fromkeys(found)
        if is_quality(cert) or is_for_this_material(cert)
    ]
    # Quality approvals first, then the material's own standards.
    return sorted(kept, key=lambda c: 0 if is_quality(c) else 1)[:MAX_CERTIFICATIONS]


#: Host suffixes that are never a commercial supplier.
_NON_COMMERCIAL_SUFFIXES = (".gov", ".edu", ".ac.uk", ".gov.uk", ".edu.au", ".mil")

#: Hosts that publish material data or aggregate suppliers, but sell nothing.
_NON_SUPPLIER_HOSTS = frozenset(
    {
        "nist.gov", "shop.nist.gov", "matweb.com", "azom.com", "wikipedia.org",
        "en.wikipedia.org", "makeitfrom.com", "matmatch.com", "efunda.com",
        "engineeringtoolbox.com", "totalmateria.com", "suppliersonline.com",
        # Video and social platforms. A product video is not a supplier, and a
        # live run returned a YouTube page as a vendor.
        "youtube.com", "m.youtube.com", "youtu.be", "vimeo.com", "facebook.com",
        "linkedin.com", "twitter.com", "x.com", "instagram.com", "pinterest.com",
        "reddit.com", "quora.com", "amazon.com", "ebay.com", "aliexpress.com",
    }
)

#: Unambiguous retail: per-unit pricing or a storefront. These override even a
#: confident "manufacturer" from the model — a page selling bar *by the inch*
#: cannot fill a 150 kg order, whatever the company also does. Observed live on
#: a page titled `3/16" Titanium Bar (Price/inch)` that was classed a
#: manufacturer and ranked second.
_STRONG_RETAIL_MARKERS = re.compile(
    r"(?:^|\.)(?:shop|store)\.|sold-by-the-|price-?per-|per-inch|"
    r"/(?:cart|checkout|add-to-cart)|/collections/",
    re.I,
)

#: Weaker signals. A real distributor often has /product/ URLs too, so these
#: only settle the question when the model had no opinion.
_WEAK_RETAIL_MARKERS = re.compile(r"/products?/|/buy/", re.I)


def _classify_by_url(url: str, model_kind: VendorKind) -> tuple[VendorKind, str | None]:
    """Deterministic backstop for the model's supplier classification.

    The model reads the page and is usually right, but a single classification
    is a thin guard for something a buyer acts on. Host-level facts are certain
    in a way a judgement call is not: a `.gov` domain does not sell titanium bar,
    whatever the page says.

    Only ever downgrades, never promotes — a real distributor is not turned into
    retail by having a `/products/` URL, but it can be flagged as retail if the
    model already thought so.
    """
    host = urlsplit(url).netloc.casefold()
    bare = host[4:] if host.startswith("www.") else host

    if bare in _NON_SUPPLIER_HOSTS or host.endswith(_NON_COMMERCIAL_SUFFIXES):
        return "not_a_supplier", (
            f"{bare} is a standards, reference or non-commercial site, not a "
            "supplier - listed last for reference only."
        )

    if (
        model_kind == "retail"
        or _STRONG_RETAIL_MARKERS.search(url)
        or (model_kind == "unknown" and _WEAK_RETAIL_MARKERS.search(url))
    ):
        return "retail", (
            "This looks like a consumer or per-piece retail listing rather than "
            "an industrial supplier - check they can quote your quantity."
        )

    return model_kind, None


#: Form and size words that are part of the *order*, not the material's
#: identity. Intake sometimes captures the whole phrase — one live run parsed
#: material_name as "Titanium Grade 5 round bar" — and that exact string appears
#: on no vendor page, so every single vendor got flagged as irrelevant and the
#: warning became noise.
_FORM_WORDS = re.compile(
    r"\b(?:round|square|hex|flat|solid|hollow|cold[\s-]?drawn|hot[\s-]?rolled|"
    r"bar|bars|rod|rods|stock|sheet|plate|tube|tubing|pipe|wire|forging|billet|"
    r"powder|strip|coil|section|profile|material|alloy)\b",
    re.I,
)


def _identifying_terms(
    spec: MaterialSpec | None, research: MaterialResearch | None
) -> list[str]:
    """Terms that would identify this material on a vendor's page.

    Prefers precise designations (UNS R56400, AMS 4928, Ti-6Al-4V) over the
    buyer's phrasing, and strips form words from the material name so that
    "Titanium Grade 5 round bar" is matched as "titanium grade 5".
    """
    if spec is None:
        return []

    candidates: list[str] = []
    if research:
        candidates.extend(research.designations)
        candidates.append(research.canonical_name)
    candidates.extend(spec.standards or [])
    if spec.grade:
        candidates.append(spec.grade)

    stripped = " ".join(_FORM_WORDS.sub(" ", spec.material_name).split())
    if stripped:
        candidates.append(stripped)

    # Family standards are dropped. ASTM B348 covers every titanium bar grade,
    # so matching on it passed Grade 7 pages for a Grade 5 enquiry — the page
    # cites the same standard while selling a different alloy.
    terms = [t for t in candidates if not is_broad_standard(t)]

    return [t for t in terms if t and len(t.strip()) >= 3]


def _tokens(value: str) -> list[str]:
    """Split into comparable tokens: "Ti-6Al-4V" -> ["ti", "6al", "4v"]."""
    return [t for t in re.split(r"[^a-z0-9]+", value.casefold()) if t]


def _term_present(term: str, page_tokens: set[str], page_flat: str) -> bool:
    """Whether `term` appears on the page, allowing for word order and spacing.

    Token-based rather than substring-based. Vendors write the same alloy as
    "Grade 5 Titanium" and "Titanium Grade 5", so a concatenated comparison
    misses genuine matches; and matching bare substrings would let the "5" in
    "R52400" satisfy a search for grade 5.

    A single-token designation is also checked against the page with its
    punctuation removed, because Ti-6Al-4V, Ti 6Al 4V and Ti6Al4V are all the
    same alloy.
    """
    tokens = _tokens(term)
    if not tokens:
        return False
    if all(token in page_tokens for token in tokens):
        return True
    joined = "".join(tokens)
    return len(joined) >= 5 and joined in page_flat


def _mentions_material(
    source: str, spec: MaterialSpec | None, research: MaterialResearch | None = None
) -> bool:
    """Whether the page actually names the material being sourced.

    A search for a specialty alloy still returns generic pages, and worse,
    neighbouring grades — a Grade 5 titanium search returned Grade 7 pages,
    which is a genuinely different alloy. This does not drop them, since the
    vendor may still stock the right grade, but it records the doubt so the
    ranking can prefer pages that name the material outright.
    """
    terms = _identifying_terms(spec, research)
    if not terms:
        return True
    page_tokens = set(_tokens(source))
    page_flat = re.sub(r"[^a-z0-9]", "", source.casefold())
    return any(_term_present(t, page_tokens, page_flat) for t in terms)


async def contact_extraction(state: dict[str, Any], config: RunnableConfig = None) -> dict:
    """Extract and verify contact details for one candidate.

    Receives a single-candidate payload from `Send()`, not the full session
    state, so it reads `candidate` directly.
    """
    candidate: VendorCandidate = state["candidate"]
    source = (candidate.raw_content or candidate.snippet or "")[:MAX_SOURCE_CHARS]

    # Follow the vendor's contact pages. This is the main quality lever: the
    # search returns a product page, but the sales address almost always lives
    # on a separate contact page. crawl4ai makes those fetches free, so there
    # is no budget check here.
    sources: dict[str, str] = {candidate.url: source} if source else {}
    try:
        sources.update(await _fetch_contact_pages(candidate.url))
    except Exception as exc:  # noqa: BLE001 - enrichment, never fatal to the branch
        log.warning("contact_extraction: contact-page step failed: %s", exc)

    if not sources or max(len(t) for t in sources.values()) < MIN_CONTENT:
        log.info("contact_extraction: %s had no usable content", candidate.url)
        return {
            "extracted_leads": [
                VendorLead(
                    company_name=candidate.company_name,
                    website=_website(candidate.url),
                    source_url=candidate.url,
                    cited_in_research=candidate.from_research,
                    confidence_notes=[
                        "Page content could not be retrieved, so no contact details "
                        "were extracted. Check the source URL directly."
                    ],
                )
            ]
        }

    # One digest per page, labelled by URL. Still regex-found candidates only,
    # so the phase-1 token budget holds: a few hundred tokens, not a page.
    digest = "\n\n".join(
        f"--- from {url} ---\n{build_candidate_digest(text)}"
        for url, text in sources.items()
    )
    model = get_model("bulk", schema=Extraction, session_id=thread_id_of(config))

    try:
        extracted: Extraction = await model.ainvoke(
            [
                SystemMessage(content=SYSTEM),
                HumanMessage(
                    content=(
                        f"Company: {candidate.company_name}\n"
                        f"URL: {candidate.url}\n"
                        f"Domain: {_domain(candidate.url)}\n\n"
                        f"{digest}"
                    )
                ),
            ]
        )
    except Exception as exc:  # noqa: BLE001 - one bad page must not fail the run
        log.warning("contact_extraction failed for %s: %s", candidate.url, exc)
        return {
            "extracted_leads": [
                VendorLead(
                    company_name=candidate.company_name,
                    website=_website(candidate.url),
                    source_url=candidate.url,
                    cited_in_research=candidate.from_research,
                    confidence_notes=[f"Contact extraction failed: {exc}"],
                )
            ]
        }

    spec: MaterialSpec | None = state.get("material_spec")
    # Relevance is judged on the product page only. A contact page never names
    # the material, so including it would make every vendor look irrelevant.
    mentions_material = _mentions_material(source, spec, state.get("research"))

    notes: list[str] = []
    if not mentions_material and spec is not None:
        # Only the doubtful candidates get a confirmation search. Doing it for
        # every vendor would roughly double the run's credit spend - twelve
        # candidates against a twelve-credit budget - to re-confirm the ones
        # whose page already names the material.
        confirmed, note = await _confirm_supplies_material(candidate, spec, config)
        mentions_material = confirmed
        notes.append(note)

    kind, kind_note = _classify_by_url(candidate.url, extracted.kind)
    if kind_note:
        notes.append(kind_note)

    lead = VendorLead(
        company_name=candidate.company_name,
        website=_website(candidate.url),
        country=extracted.country,
        kind=kind,
        contact_name=extracted.contact_name,
        email=extracted.email,
        phone=extracted.phone,
        certifications_found=_rank_certifications(
            find_certifications(source), spec, state.get("research")
        ),
        source_url=candidate.url,
        cited_in_research=candidate.from_research,
        mentions_material=mentions_material,
        confidence_notes=notes,
    )

    # Independent second check: everything above is re-verified against the
    # pages actually fetched, and the matching page recorded on the lead.
    return {"extracted_leads": [verify_lead_across(lead, sources)]}


async def _confirm_supplies_material(
    candidate: VendorCandidate, spec: MaterialSpec, config: RunnableConfig = None
) -> tuple[bool, str]:
    """Ask the web whether this company supplies this material after all.

    Reached only when the fetched page does not name the material. That happens
    for two very different reasons and the page alone cannot tell them apart: a
    real stockist whose catalogue is behind a search box, or a page about a
    neighbouring grade. One targeted query separates them, and it is worth a
    credit precisely because it is not run for every vendor.

    Returns (confirmed, note). The note is written either way, so a buyer can
    see what was checked rather than being handed a bare boolean.
    """
    settings = get_settings()
    # ONE allocation shared by every branch of the fan-out, not one each.
    # `allocation()` memoises by name on the run budget, so twelve concurrent
    # candidates draw from the same few credits rather than twelve times over.
    budget = await allocation(config, "contact_extraction", settings)
    query = f"{candidate.company_name} {spec.material_name}"
    domain = _website(candidate.url)

    try:
        results = await search(
            query,
            budget,
            depth=Depth.SNIPPETS,
            max_results=CONFIRMATION_RESULTS,
            settings=settings,
        )
    except BudgetExhausted:
        return False, (
            f"This page does not name {spec.material_name}, and the search budget "
            "was spent before it could be confirmed elsewhere - it may be a "
            "neighbouring grade. Confirm before enquiring."
        )

    # Only this vendor's own pages count. A directory listing that pairs the
    # company with the material is exactly the weak evidence this check exists
    # to avoid trusting.
    material = normalise(spec.material_name)
    for result in results:
        if domain not in _website(result.url):
            continue
        if material in normalise(f"{result.title} {result_text(result)}"):
            return True, (
                f"The page fetched does not name {spec.material_name}, but "
                f"{result.url} on the same site does. Confirm the exact grade "
                "when enquiring."
            )

    return False, (
        f"This page does not name {spec.material_name} or any of its "
        "designations, and a search of this vendor's own site did not find it "
        "either - it may be a neighbouring grade. Confirm before enquiring."
    )


async def _fetch_contact_pages(product_url: str) -> dict[str, str]:
    """Fetch the vendor's contact pages. Free, best-effort, never raises.

    Uses the links crawl4ai already extracted from the product page. When a
    page exposed no usable links — some sites build navigation in JavaScript —
    falls back to trying the conventional `/contact` URLs, since one
    speculative free fetch is cheap.
    """
    try:
        settings = get_settings()
        fetcher = get_fetcher(settings)
        if not fetcher.is_enabled() or settings.crawl_max_contact_pages <= 0:
            return {}

        product = await fetcher.fetch(product_url)
        urls = find_contact_links(
            product.links, product_url, limit=settings.crawl_max_contact_pages
        )
        if not urls:
            urls = guess_contact_urls(product_url, limit=1)

        pages = await fetcher.fetch_many(urls)
    except Exception as exc:  # noqa: BLE001 - enrichment must never fail the run
        log.warning("contact_extraction: contact-page fetch failed for %s: %s", product_url, exc)
        return {}

    found = {p.url: p.content[:MAX_SOURCE_CHARS] for p in pages if p.is_useful}
    if found:
        log.info(
            "contact_extraction: %s -> %d contact page(s): %s",
            _domain(product_url),
            len(found),
            list(found),
        )
    return found
