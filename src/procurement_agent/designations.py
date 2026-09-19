"""Which material designations actually identify one grade.

Shared by the vendor cache (deciding what may become a lookup alias) and by
contact extraction (deciding whether a page is about the right material). Both
were bitten by the same mistake, so the rule lives in one place.

The distinction that matters: **specification families are not grades.**
ASTM B348 covers every titanium bar grade, ASTM A276 every stainless bar. AMS
numbers, UNS numbers and Werkstoff numbers name a single alloy. Treating a
family standard as identifying makes a Grade 7 page look like a Grade 5 match,
and makes an unrelated alloy look like a cache hit.
"""

from __future__ import annotations

import re

#: Material families and form words. Alone, none identifies a material:
#: "titanium" is thousands of alloys, "bar stock" is a shape.
GENERIC_TERMS = frozenset(
    {
        "titanium", "steel", "stainless", "stainless steel", "carbon steel",
        "alloy", "alloy steel", "nickel", "nickel alloy", "aluminium", "aluminum",
        "copper", "brass", "bronze", "inconel", "monel", "hastelloy", "duplex",
        "super duplex", "tool steel", "mild steel", "metal", "superalloy",
        "bar", "bar stock", "rod", "round bar", "round rod", "sheet", "plate",
        "tube", "pipe", "wire", "forging", "billet", "powder", "strip", "coil",
        "material", "stock", "product",
    }
)

#: An ordinal with no material attached. "Grade 5" is Ti-6Al-4V in titanium and
#: something entirely different in fasteners or cast iron.
_BARE_ORDINAL_RE = re.compile(r"^(?:grade|gr|type|class|cl)\s*[a-z]?\d{1,3}[a-z]?$")

#: Specification families covering many grades, and handbooks that are not
#: designations at all.
#:
#: AMS, UNS and Werkstoff are deliberately absent — those are grade-specific
#: and are exactly the designations worth trusting.
#:
#: `mmpds` is the odd one out: it is a properties handbook (Metallic Materials
#: Properties Development and Standardization), so "MMPDS-01" names a document
#: rather than an alloy. It was returned as a designation on a real run and
#: passed every other test here, because it has digits and two words.
#:
#: SAE J-numbers get their own alternative without a trailing `\b`: normalising
#: "SAE J404" gives "sae j404", where `j` runs straight into `4` with no word
#: boundary between them, so the shared `\b` below never matches it.
_BROAD_STANDARD_RE = re.compile(
    r"^(?:astm|asme|en|din|jis|bs|iso|gost|gb|mmpds|mil|nas)\b|^sae\s*j"
)

_PUNCT = re.compile(r"[^a-z0-9]+")


def normalise(value: str | None) -> str:
    """Collapse a name to a comparable key: "Custom-465" -> "custom 465"."""
    if not value:
        return ""
    return _PUNCT.sub(" ", value.casefold()).strip()


def is_broad_standard(value: str) -> bool:
    """True for a specification family that spans many grades."""
    return bool(_BROAD_STANDARD_RE.match(normalise(value)))


def is_grade_specific(value: str) -> bool:
    """Whether `value` identifies one material rather than a whole family.

    Guards two distinct failures, both observed live: a cache that returned
    Ti-6Al-4V vendors for a "Titanium Grade 2" query, and a relevance check
    that passed Grade 7 pages for a Grade 5 enquiry because both cite ASTM
    B348. Anything not clearly specific is rejected — a missed match costs far
    less than a confidently wrong one.
    """
    value = normalise(value)
    if not value or len(value) < 3:
        return False
    if value in GENERIC_TERMS:
        return False
    if _BARE_ORDINAL_RE.match(value):
        return False
    if is_broad_standard(value):
        return False
    # A single word with no digits is a family name, not a designation.
    # noqa: SIM103 - the linter would fold this into `return not (...)`, but the
    # value of this function is the readable list of *reasons* something is
    # rejected. Collapsing the last one breaks the pattern for no gain.
    if len(value.split()) < 2 and not any(ch.isdigit() for ch in value):  # noqa: SIM103
        return False
    return True
