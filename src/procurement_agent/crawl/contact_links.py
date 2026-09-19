"""Pick out the pages on a vendor's site that are likely to carry contact details.

The gap this closes: a search returns a vendor's *product* page, but the sales
email and phone number almost always live on a separate contact page. Phase 1
only ever read the product page, which is why roughly half the leads came back
with no way to reach the company.

Pure functions, no network, so the matching rules are cheap to test — including
the false positives, which matter more than they look. `/contact-lens-alloys`
is a product page, not a contact page, and following it wastes a fetch.
"""

from __future__ import annotations

import re
from urllib.parse import urljoin, urlsplit, urlunsplit

#: Path segments that mark a contact page. Matched against path segments rather
#: than the raw URL so that `/contact` hits and `/contact-lens` does not.
_CONTACT_SEGMENTS = frozenset(
    {
        "contact",
        "contacts",
        "contact-us",
        "contactus",
        "contact-me",
        "kontakt",
        "contacto",
        "contatti",
        "nous-contacter",
        "enquiry",
        "enquiries",
        "inquiry",
        "inquiries",
        "get-a-quote",
        "request-a-quote",
        "quote-request",
        "rfq",
        "reach-us",
        "reachus",
        "get-in-touch",
        "about",
        "about-us",
        "aboutus",
        "company",
        "imprint",
        "impressum",
        "legal-notice",
    }
)

#: Anchor text that signals a contact page even when the URL does not.
_CONTACT_TEXT = re.compile(
    r"\b(contact|enquir|inquir|get in touch|reach us|request a quote|"
    r"kontakt|contacto|impressum)\b",
    re.I,
)

#: Ranked best-first. A dedicated contact page beats an "about" page, which
#: beats a quote form, because that is the order in which a direct sales
#: address is likely to appear.
_PRIORITY = (
    ("contact", 0),
    ("kontakt", 0),
    ("contacto", 0),
    ("contatti", 0),
    ("reach", 1),
    ("touch", 1),
    ("enquir", 2),
    ("inquir", 2),
    ("quote", 3),
    ("rfq", 3),
    ("about", 4),
    ("company", 4),
    ("imprint", 5),
    ("impressum", 5),
)

#: Never worth a fetch even though the path can look contact-ish.
_SKIP = re.compile(
    r"\.(pdf|jpe?g|png|gif|svg|zip|docx?|xlsx?)$|"
    r"/(cart|checkout|login|signin|register|account|privacy|terms|cookie)",
    re.I,
)


def _domain(url: str) -> str:
    host = urlsplit(url).netloc.casefold()
    return host[4:] if host.startswith("www.") else host


def _canonical(url: str) -> str:
    """Drop the fragment and any trailing slash so near-duplicates collapse."""
    parts = urlsplit(url)
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((parts.scheme, parts.netloc, path, parts.query, ""))


def _segments(url: str) -> list[str]:
    return [s for s in urlsplit(url).path.casefold().split("/") if s]


def _rank(url: str) -> int:
    """Lower sorts first. Shallow paths win ties: /contact beats /a/b/contact."""
    lowered = urlsplit(url).path.casefold()
    score = 9
    for marker, value in _PRIORITY:
        if marker in lowered:
            score = min(score, value)
    return score * 10 + min(len(_segments(url)), 9)


def looks_like_contact_page(url: str, text: str = "") -> bool:
    """Whether `url` (optionally with its anchor `text`) is a contact page."""
    if _SKIP.search(url):
        return False
    segments = _segments(url)
    if any(segment in _CONTACT_SEGMENTS for segment in segments):
        return True
    # Anchor text catches sites that route contact pages through opaque URLs.
    return bool(text and _CONTACT_TEXT.search(text) and len(segments) <= 3)


def find_contact_links(
    links: list[dict], base_url: str, *, limit: int = 3
) -> list[str]:
    """Best contact-page URLs on the same site as `base_url`, best first.

    `links` is crawl4ai's internal-link list: dicts carrying `href` and `text`.
    Off-site links are dropped — a link to a marketplace profile is not this
    vendor's contact page, and following it would attribute someone else's
    email to them.
    """
    base_domain = _domain(base_url)
    ranked: dict[str, int] = {}

    for link in links:
        href = (link.get("href") or "").strip()
        if not href or href.startswith(("mailto:", "tel:", "javascript:", "#")):
            continue

        absolute = urljoin(base_url, href)
        if not absolute.startswith(("http://", "https://")):
            continue
        if _domain(absolute) != base_domain:
            continue
        if not looks_like_contact_page(absolute, link.get("text") or ""):
            continue

        canonical = _canonical(absolute)
        if canonical == _canonical(base_url):
            continue  # the page we already have
        ranked.setdefault(canonical, _rank(canonical))

    return sorted(ranked, key=lambda u: (ranked[u], u))[:limit]


def guess_contact_urls(base_url: str, *, limit: int = 2) -> list[str]:
    """Conventional contact URLs, for when a page exposed no usable links.

    A fallback, not the main path: several vendor sites render navigation with
    JavaScript that the crawler does not always resolve, and `/contact` is a
    strong enough convention to be worth one speculative fetch.
    """
    parts = urlsplit(base_url)
    root = f"{parts.scheme}://{parts.netloc}"
    return [f"{root}/{segment}" for segment in ("contact", "contact-us")][:limit]
