"""Find contact candidates with regex, so the model never has to.

Two reasons this is regex-first rather than "hand the page to an LLM".

Budget: a scraped vendor page runs 5,000-15,000 tokens, and Groq's free tier
allows 6,000 tokens per minute. Asking a model to *find* an email address that a
regex finds for free is the single most expensive mistake this app could make.

Correctness: anything regex returns is by construction a literal substring of
the source page, so it cannot be hallucinated. The model is left with the
judgment call it is actually needed for — which of these candidates is the sales
contact — over a payload small enough to be cheap.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

EMAIL_RE = re.compile(
    r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}",
)

#: Deliberately conservative. Loose phone patterns match part numbers,
#: dimensions and dates, which is worse than missing a phone number: a wrong
#: number reaches a real stranger.
PHONE_RE = re.compile(
    r"""
    (?<![\w.])
    (?:\+\d{1,3}[\s.\-]?)?      # optional country code, e.g. +44
    (?:\(\d{1,4}\)[\s.\-]?)?    # optional parenthesised area code, e.g. (0)
    \d{1,4}(?:[\s.\-]\d{2,6}){1,4}
    (?![\w.])
    """,
    re.VERBOSE,
)

#: A digit run only counts as a phone number if something corroborates it.
#:
#: Supplier pages in this domain are dense with digit sequences that the pattern
#: above matches perfectly well: alloy grade lists ("304 316L 321 410 420 431"),
#: dimension tables, AMS numbers and part codes. Requiring a `+` prefix,
#: parentheses, or a nearby phone keyword is what separates a real number from a
#: row of stainless grades.
#: Searched anywhere in the lookbehind window, NOT anchored to its end.
#:
#: Real contact pages break the anchored version constantly: "CALL TOLL FREE
#: 1-800-500-2141" puts the keyword four words before the number, and a second
#: number listed after the first has no keyword of its own at all.
_PHONE_KEYWORD_RE = re.compile(
    r"(?:tel|telephone|phone|call|fax|mobile|cell|whatsapp|contact|office|"
    r"toll[\s\-]?free|hotline|direct|ph|mob)\b",
    re.I,
)

#: Immediately-preceding words that mean this is a materials designation, not a
#: phone number. Supplier pages are dense with these, and "Contact us for
#: grades 304 316 321 410" would otherwise sail through the keyword check.
_MATERIAL_CONTEXT_RE = re.compile(
    r"(?:grades?|alloys?|types?|uns|ams|astm|aisi|sae|din|en|iso|no\.?|"
    r"size|dia|diameter|thickness|width|length|qty|quantity|part)"
    r"[\s:#\-]*$",
    re.I,
)

#: How far back to look. Wide enough to reach a heading like "CALL TOLL FREE",
#: narrow enough that an unrelated mention of "contact" elsewhere on the page
#: does not legitimise a random digit run.
_PHONE_LOOKBEHIND = 40

#: Obfuscations vendors use to slow down scrapers. Normalised before matching so
#: a real address written "sales [at] acme [dot] com" is still found.
#: Cloudflare's email protection: `/cdn-cgi/l/email-protection#<hex>`, where the
#: first hex byte is an XOR key for the rest. Sites using it render the address
#: as the literal text "[email protected]", so the real address is invisible to
#: a plain regex — but it is fully recoverable, and decoding it turns a dead
#: lead into a reachable one.
_CF_EMAIL_RE = re.compile(r"/cdn-cgi/l/email-protection#([0-9a-fA-F]{6,})")


def decode_cloudflare_emails(text: str) -> str:
    """Replace Cloudflare-protected email links with the addresses they encode."""

    def _decode(match: re.Match) -> str:
        payload = match.group(1)
        try:
            key = int(payload[:2], 16)
            decoded = "".join(
                chr(int(payload[i : i + 2], 16) ^ key) for i in range(2, len(payload), 2)
            )
        except ValueError:
            return match.group(0)
        # Only substitute when it actually decoded to an address.
        return f" {decoded} " if EMAIL_RE.fullmatch(decoded) else match.group(0)

    return _CF_EMAIL_RE.sub(_decode, text)


_OBFUSCATIONS = [
    (re.compile(r"\s*[\[\(]\s*at\s*[\]\)]\s*", re.I), "@"),
    (re.compile(r"\s+at\s+(?=[A-Za-z0-9.\-]+\.[A-Za-z]{2,})", re.I), "@"),
    (re.compile(r"\s*[\[\(]\s*dot\s*[\]\)]\s*", re.I), "."),
    (re.compile(r"\s+dot\s+(?=[A-Za-z]{2,}\b)", re.I), "."),
]

#: Addresses that are never a useful procurement contact.
_JUNK_EMAIL_HINTS = (
    "example.com",
    "yourdomain",
    "domain.com",
    "sentry.io",
    "wixpress.com",
    "@2x.png",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".svg",
    ".css",
    ".js",
)

CERTIFICATION_RE = re.compile(
    r"\b(?:"
    r"AS\s?9100[A-D]?|ISO\s?9001(?::\d{4})?|ISO\s?14001|NADCAP|ITAR|"
    r"AMS\s?\d{4}[A-Z]?|ASTM\s?[A-Z]\s?\d{1,4}|DFARS|"
    r"EN\s?10204(?:[\s.\-]*3\.[12])?"
    r")\b",
    re.I,
)

#: Characters of surrounding text kept with each candidate. Enough for the model
#: to tell a sales address from a careers one, small enough to stay cheap.
CONTEXT_CHARS = 160


@dataclass(frozen=True)
class ContactCandidate:
    """One regex hit plus the text around it."""

    kind: str  # "email" | "phone"
    value: str
    context: str


def deobfuscate(text: str) -> str:
    """Turn common anti-scraper spellings back into real addresses."""
    text = decode_cloudflare_emails(text)
    for pattern, replacement in _OBFUSCATIONS:
        text = pattern.sub(replacement, text)
    return text


def _looks_like_junk_email(email: str) -> bool:
    lowered = email.lower()
    return any(hint in lowered for hint in _JUNK_EMAIL_HINTS)


def _plausible_phone(raw: str, text: str, start: int) -> bool:
    """Reject anything that is really a dimension, year, grade, or part number."""
    digits = re.sub(r"\D", "", raw)
    if not 7 <= len(digits) <= 15:
        return False
    # A run of identical digits is placeholder text, not a phone number.
    if len(set(digits)) <= 1:
        return False

    preceding = text[max(0, start - _PHONE_LOOKBEHIND) : start]

    # Checked before any acceptance rule: on a supplier page a digit run
    # directly after "grade" or "AMS" is a designation, whatever else is nearby.
    if _MATERIAL_CONTEXT_RE.search(preceding):
        return False

    stripped = raw.strip()
    # A country code or a parenthesised area code is corroboration in itself.
    if stripped.startswith("+") or ("(" in stripped and ")" in stripped):
        return True

    # Otherwise something nearby has to say this is a phone number.
    return bool(_PHONE_KEYWORD_RE.search(preceding))


def _context_for(text: str, start: int, end: int) -> str:
    lo = max(0, start - CONTEXT_CHARS)
    hi = min(len(text), end + CONTEXT_CHARS)
    return " ".join(text[lo:hi].split())


def find_emails(text: str) -> list[ContactCandidate]:
    """Every plausible email in `text`, de-duplicated, with context."""
    normalised = deobfuscate(text)
    seen: set[str] = set()
    out: list[ContactCandidate] = []
    for match in EMAIL_RE.finditer(normalised):
        value = match.group(0).rstrip(".")
        key = value.lower()
        if key in seen or _looks_like_junk_email(value):
            continue
        seen.add(key)
        out.append(
            ContactCandidate(
                kind="email",
                value=value,
                context=_context_for(normalised, match.start(), match.end()),
            )
        )
    return out


def find_phones(text: str) -> list[ContactCandidate]:
    """Every plausible phone number in `text`, de-duplicated by digits."""
    seen: set[str] = set()
    out: list[ContactCandidate] = []
    for match in PHONE_RE.finditer(text):
        raw = match.group(0).strip()
        if not _plausible_phone(raw, text, match.start()):
            continue
        key = re.sub(r"\D", "", raw)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            ContactCandidate(
                kind="phone",
                value=raw,
                context=_context_for(text, match.start(), match.end()),
            )
        )
    return out


def find_certifications(text: str) -> list[str]:
    """Quality/compliance standards named on the page."""
    seen: dict[str, str] = {}
    for match in CERTIFICATION_RE.finditer(text):
        value = " ".join(match.group(0).split())
        seen.setdefault(value.upper(), value)
    return list(seen.values())


#: Phrases that introduce a material's supplier in academic prose.
#:
#: Papers name their sources in the methods section and acknowledgements —
#: "the bar stock was supplied by X", "purchased from Y" — and those companies
#: are frequently specialist mills that commercial search results bury under
#: marketplaces. The capture group takes the text following the phrase; a model
#: then decides which part of it is actually a company name, because the
#: grammar here is far too varied for a regex to finish the job.
#: `manufactured by` and `produced by` are deliberately absent. In materials
#: science they overwhelmingly introduce a *process* ("manufactured by laser
#: powder bed fusion"), not a company, and including them produced far more
#: noise than signal when tested against real arXiv abstracts.
SOURCING_RE = re.compile(
    r"(?:"
    r"suppl(?:ied|y|ier)\s+by|purchased\s+from|obtained\s+from|sourced\s+from|"
    r"procured\s+from|provided\s+by|donated\s+by|received\s+from|"
    r"courtesy\s+of|acquired\s+from|bought\s+from|"
    r"kindly\s+(?:supplied|provided|donated)\s+by"
    r")\s+([^.;]{3,120})",
    re.I,
)

#: Words that mark a match as describing a process or method rather than a
#: company, so the candidate is discarded before it reaches the model.
_PROCESS_WORDS = re.compile(
    r"^(?:a|an|the)?\s*(?:laser|powder|suction|vacuum|arc|electron|additive|"
    r"selective|direct|hot|cold|thermal|chemical|mechanical|plasma|induction|"
    r"casting|milling|forging|sintering|melting|deposition|extrusion|welding|"
    r"annealing|quenching|machining|means|use|using|applying|combining)\b",
    re.I,
)

#: Characters of context kept around a sourcing mention.
SOURCING_CONTEXT = 200


@dataclass(frozen=True)
class SourcingMention:
    """A phrase in a paper that appears to name a material supplier."""

    phrase: str
    candidate_text: str
    context: str


def find_sourcing_mentions(text: str, *, limit: int = 10) -> list[SourcingMention]:
    """Find passages that look like they name where a material came from."""
    out: list[SourcingMention] = []
    seen: set[str] = set()
    for match in SOURCING_RE.finditer(text):
        candidate = " ".join(match.group(1).split())
        if _PROCESS_WORDS.match(candidate):
            continue
        key = candidate.casefold()
        if key in seen:
            continue
        seen.add(key)
        lo = max(0, match.start() - SOURCING_CONTEXT)
        hi = min(len(text), match.end() + SOURCING_CONTEXT)
        out.append(
            SourcingMention(
                phrase=" ".join(match.group(0).split())[:80],
                candidate_text=candidate,
                context=" ".join(text[lo:hi].split()),
            )
        )
        if len(out) >= limit:
            break
    return out


def build_sourcing_digest(mentions: list[SourcingMention]) -> str:
    """Compact prompt payload for turning mentions into company names."""
    if not mentions:
        return "No sourcing statements were found in these papers."
    lines = ["SOURCING STATEMENTS FOUND VERBATIM IN RESEARCH PAPERS:"]
    for mention in mentions:
        lines.append(f"  statement: {mention.phrase}")
        lines.append(f"    context: ...{mention.context}...")
    return "\n".join(lines)


def build_candidate_digest(text: str, *, max_emails: int = 12, max_phones: int = 8) -> str:
    """Render candidates as a compact prompt payload.

    This replaces sending the page itself, which is the whole token saving: a
    15,000-token page becomes a few hundred tokens of candidates-with-context.
    """
    emails = find_emails(text)[:max_emails]
    phones = find_phones(text)[:max_phones]

    lines: list[str] = []
    if emails:
        lines.append("EMAIL CANDIDATES (found verbatim on the page):")
        lines += [f"  {c.value}\n    context: ...{c.context}..." for c in emails]
    if phones:
        lines.append("PHONE CANDIDATES (found verbatim on the page):")
        lines += [f"  {c.value}\n    context: ...{c.context}..." for c in phones]
    if not lines:
        lines.append("No contact candidates were found on this page.")
    return "\n".join(lines)
