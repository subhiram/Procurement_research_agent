"""Draft vendor outreach emails. Optional, and only on request.

Drafts only — nothing is sent from here. Quoting and pricing are out of scope
for this app, so these emails ask for availability and a quote; they never make
a commercial commitment.
"""

from __future__ import annotations

import asyncio
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from procurement_agent.config import get_settings
from procurement_agent.graph.state import EmailDraft, MaterialSpec, SessionState, VendorLead
from procurement_agent.llm.models import get_model

log = logging.getLogger(__name__)

#: Drafting one email per vendor costs a model call each, so cap it.
MAX_DRAFTS = 10

EXPORT_NOTE = (
    "Note: this material may be subject to export control (ITAR/EAR). Please "
    "confirm your ability to supply to our jurisdiction."
)

SYSTEM = """You write short B2B procurement enquiry emails for an industrial \
contracting company.

Write a plain, professional enquiry to a supplier asking about availability of a \
specific material. Requirements:

- Subject line: material, form and quantity. No marketing language.
- Body: 4-6 sentences. State exactly what is needed, ask about availability, lead \
time, minimum order quantity, and material certification.
- Ask them to quote. Do NOT state, estimate, or negotiate any price, and do not \
commit to any quantity beyond what is stated.
- Do not invent company details, contract references, project names, or \
certifications. Use only what you are given.
- Address the named contact if one is provided; otherwise open with "Hello".
- Sign off as "Procurement Team" with no invented company name or signature block.

Keep it direct. Suppliers respond better to a clear specification than to a \
polished pitch.
"""


async def draft_outreach_email(state: SessionState) -> dict:
    """Draft one email per reachable vendor, in parallel within the fan-out cap."""
    spec = state.get("material_spec")
    leads = state.get("vendor_leads", [])
    if spec is None or not leads:
        log.info("draft_outreach_email: nothing to draft")
        return {"email_drafts": []}

    targets = [lead for lead in leads if lead.email][:MAX_DRAFTS]
    if not targets:
        log.info("draft_outreach_email: no vendor has a verified email address")
        return {"email_drafts": []}

    semaphore = asyncio.Semaphore(get_settings().max_fanout)

    async def draft(lead: VendorLead) -> EmailDraft | None:
        async with semaphore:
            try:
                return await _draft_one(spec, lead)
            except Exception as exc:  # noqa: BLE001 - one failure must not sink the batch
                log.warning("draft failed for %s: %s", lead.company_name, exc)
                return None

    drafts = [d for d in await asyncio.gather(*(draft(t) for t in targets)) if d]
    log.info("draft_outreach_email: drafted %d emails", len(drafts))
    return {"email_drafts": drafts}


async def _draft_one(spec: MaterialSpec, lead: VendorLead) -> EmailDraft:
    model = get_model("drafting", schema=EmailDraft)
    draft: EmailDraft = await model.ainvoke(
        [
            SystemMessage(content=SYSTEM),
            HumanMessage(
                content=(
                    f"Material required:\n{spec.model_dump_json(indent=2)}\n\n"
                    f"Supplier: {lead.company_name}\n"
                    f"Contact name: {lead.contact_name or '(none given)'}\n"
                    f"Country: {lead.country or '(unknown)'}"
                )
            ),
        ]
    )
    # Set authoritatively rather than trusting the model to echo them back.
    return draft.model_copy(
        update={
            "vendor_company": lead.company_name,
            "to_email": lead.email,
            "body": f"{draft.body.rstrip()}\n\n{EXPORT_NOTE}",
        }
    )
