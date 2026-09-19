"""Session state and the domain models threaded through the graph.

The Pydantic models here serve double duty: they are `.with_structured_output()`
targets for the LLM calls and the response schemas for the FastAPI layer.
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, BeforeValidator, Field
from typing_extensions import TypedDict

Status = Literal["clarifying", "researching", "done"]

#: `retail` and `not_a_supplier` were added after a live run returned NIST
#: (a standards body selling reference chips) and three retail sites among
#: twelve results for a 150 kg enquiry. They are ranked last and labelled
#: rather than dropped: a small-quantity source is occasionally useful, and a
#: misjudged vendor should never vanish silently.
VendorKind = Literal[
    "manufacturer", "distributor", "trader", "retail", "not_a_supplier", "unknown"
]


def _empty_if_none(value: object) -> object:
    """Coerce a null list back to an empty one."""
    return [] if value is None else value


#: A list field that tolerates the model sending `null` instead of `[]`.
#:
#: Models routinely emit null for an empty list, and Groq validates tool calls
#: against the JSON schema *before* the response reaches Pydantic — so a plain
#: `list[str]` field is rejected upstream with `expected array, but got null`
#: and no validator of ours ever runs. Including null in the declared type makes
#: the schema accept it; the BeforeValidator then normalises it away, so every
#: consumer still sees a list.
def _nullable_list(item_type):
    return Annotated[list[item_type] | None, BeforeValidator(_empty_if_none)]


StrList = _nullable_list(str)


class MaterialSpec(BaseModel):
    """What the buyer actually wants, normalised."""

    material_name: str = Field(description="Common or trade name, e.g. 'Custom 465'")
    grade: str | None = Field(
        default=None, description="Grade / UNS / AMS / ASTM designation if stated"
    )
    form: str | None = Field(
        default=None, description="bar, rod, sheet, plate, tube, wire, forging, powder"
    )
    dimensions: str | None = Field(default=None, description="As stated, e.g. 'Dia 2 inch'")
    quantity: float | None = None
    unit: str | None = Field(default=None, description="KG, LB, PCS, M, FT")
    condition: str | None = Field(
        default=None, description="Temper / heat treatment condition, e.g. 'H900'"
    )
    standards: StrList = Field(
        default_factory=list, description="Standards explicitly called out"
    )

    def search_label(self) -> str:
        """Compact identifier used to build search queries."""
        parts = [self.material_name]
        if self.grade and self.grade.lower() not in self.material_name.lower():
            parts.append(self.grade)
        if self.form:
            parts.append(self.form)
        return " ".join(parts)


class MaterialResearch(BaseModel):
    """Findings about the material itself, ahead of any vendor search."""

    canonical_name: str
    designations: StrList = Field(
        default_factory=list, description="UNS / AMS / ASTM / EN equivalents"
    )
    synonyms: StrList = Field(default_factory=list, description="Trade names and aliases")
    common_forms: StrList = Field(default_factory=list)
    # Surfaced rather than silently resolved: a trade name can map to several
    # different materials across industries.
    ambiguities: StrList = Field(
        default_factory=list,
        description="Competing interpretations the buyer may need to disambiguate",
    )
    notes: str | None = None


class ClarificationQuestion(BaseModel):
    field: str = Field(description="Which MaterialSpec field this resolves")
    question: str
    why: str = Field(description="Why this matters for finding the right vendor")


class VendorCandidate(BaseModel):
    """A search hit, before contact extraction and verification."""

    company_name: str
    url: str
    snippet: str = ""
    raw_content: str | None = None
    query: str = Field(default="", description="Which search query surfaced this")
    #: True when this vendor was found via a research paper's sourcing
    #: statement rather than open-web search. Worth surfacing: it is a stronger
    #: signal that the company genuinely supplies the material.
    from_research: bool = False


class VendorLead(BaseModel):
    """A verified vendor. Every contact field here has passed the grounding check."""

    company_name: str
    website: str
    country: str | None = None
    kind: VendorKind = "unknown"
    contact_name: str | None = None
    email: str | None = None
    phone: str | None = None
    certifications_found: list[str] = Field(default_factory=list)
    #: The product page the vendor was found on.
    source_url: str
    #: The page the email/phone were actually found on, which is often a
    #: separate contact page. Kept distinct from `source_url` so the grounding
    #: guarantee stays precise: without it the record would only say the
    #: contact appeared on *one of* the pages fetched for this vendor.
    contact_source_url: str | None = None
    #: Named as a material supplier in a research paper, which is independent
    #: corroboration that they really do supply it.
    cited_in_research: bool = False
    #: Whether the source page actually names the material. A search for a
    #: specialty alloy still returns generic stainless pages; those are ranked
    #: down rather than dropped, since the vendor may still stock it.
    mentions_material: bool = True
    # Records what was dropped and why, so a thin lead is legible rather than
    # looking like a silent extraction failure.
    confidence_notes: list[str] = Field(default_factory=list)


class EmailDraft(BaseModel):
    vendor_company: str
    to_email: str | None = None
    subject: str
    body: str


class SessionState(TypedDict, total=False):
    """One checkpointed state object threaded through every node."""

    messages: Annotated[list[AnyMessage], add_messages]
    raw_input: str
    material_spec: MaterialSpec | None
    pending_questions: list[ClarificationQuestion]
    clarification_history: list[dict]
    research: MaterialResearch | None
    #: Supplier names mined from research papers. These become extra vendor
    #: search queries rather than leads in their own right, because a paper
    #: names a company but never how to reach it.
    sourcing_companies: list[str]

    # Fan-in fields: concurrent Send() branches merge instead of clobbering.
    search_queries: Annotated[list[str], operator.add]
    vendor_candidates: Annotated[list[VendorCandidate], operator.add]
    #: Written concurrently by contact_extraction branches, one per candidate.
    extracted_leads: Annotated[list[VendorLead], operator.add]

    #: Final ranked, deduplicated list produced by vendor_summary.
    vendor_leads: list[VendorLead]
    email_drafts: list[EmailDraft]

    search_credits_remaining: int
    truncated: bool
    status: Status

    #: Where this run was archived to, set by the terminal `save_run` node.
    archived_to: str | None
