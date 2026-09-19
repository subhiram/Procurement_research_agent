"""Identify what the material actually is, before looking for anyone selling it.

This node used to run on model memory alone, and that was the single largest
source of wrong output in the graph. A model asked for the UNS equivalent of a
trade name will supply one whether or not it knows it, and a fabricated
designation is undetectable by inspection: "UNS S46500" is exactly as plausible
as "UNS S45500". The damage is not contained here either -
`vendor_search.build_queries()` builds its query set from these designations, so
one invented number sends the entire vendor search after a different alloy and
the buyer has no way to notice.

So the node now works from evidence. It searches first, hands the model the
retrieved text, and then **drops any designation or synonym that does not appear
in that text**. That last step is the one that matters: supplying sources
without checking the output against them just gives the model more rope. It is
the same verify-against-corpus rule `research_sourcing` already applies to
supplier names, for the same reason.

Ambiguities are deliberately exempt from the check. That field is the model
reporting its own uncertainty, and there is no corpus a doubt could be verified
against - filtering it would silently delete exactly the warning the buyer most
needs.

Runs on the `reasoning` tier: low token volume, highest judgment, so this is the
one call worth spending the scarcest quota on.
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from procurement_agent.config import get_settings
from procurement_agent.designations import is_grade_specific, normalise
from procurement_agent.graph.context import allocation, thread_id_of
from procurement_agent.graph.state import MaterialResearch, SessionState
from procurement_agent.llm.models import get_model
from procurement_agent.search.client import BudgetExhausted, result_text, search
from searchroute import Capability, Depth

log = logging.getLogger(__name__)

#: Hits per evidence query.
EVIDENCE_RESULTS = 5

#: Cap on the evidence text handed to the model. This runs on the highest tier,
#: where the token budget is tightest, and designations appear in the first few
#: hundred characters of a datasheet or an encyclopedia entry rather than deep
#: inside one.
MAX_EVIDENCE_CHARS = 6000

SYSTEM = """You are a materials engineer supporting an industrial procurement team.

You are given a material specification and REFERENCE EXTRACTS retrieved from the \
web. Identify the material precisely so that a vendor search can target the right \
product:

- canonical_name: the name the industry actually uses.
- designations: formal equivalents — UNS, AMS, ASTM, EN, werkstoff numbers.
- synonyms: trade names and aliases the material is sold under, since different \
vendors list it differently and the search needs all of them.
- common_forms: the forms this material is actually supplied in.
- ambiguities: THIS FIELD MATTERS MOST. If the name could refer to more than one \
material — a proprietary trade name reused across industries, a regional naming \
difference, a designation that maps to several alloys — say so explicitly and \
describe each candidate interpretation. Do NOT silently pick the one you think \
is most likely. A wrong guess here sends the entire vendor search after the wrong \
product, and the buyer has no way to notice.

CRITICAL RULE FOR designations AND synonyms: take them from the reference \
extracts. Do not add a designation from your own knowledge, however confident you \
are, and do not correct or complete one that appears there. Anything you return \
that is not in the extracts will be discarded before it is used, so inventing one \
does not help the buyer — it just loses you a slot.

If the extracts do not establish a designation, return fewer of them. An \
incomplete answer is recoverable; a confidently wrong one is not.

`ambiguities`, `canonical_name`, `common_forms` and `notes` are your own judgment \
and are not checked against the extracts. Use them to say what you actually think, \
including that the extracts were thin or contradictory.
"""


async def _gather_evidence(spec, settings, config=None) -> str:
    """Retrieved text about this material, or empty when nothing was found.

    Two angles, because they surface different things: an open web search finds
    vendor datasheets and cross-reference tables, which is where equivalents are
    actually listed, while a reference lookup finds encyclopedic descriptions,
    which is where a reused trade name gets disambiguated.

    Best-effort throughout. No evidence means the verification step below drops
    every designation, which is the correct outcome - an unsourced designation is
    exactly what this node exists to stop.
    """
    budget = await allocation(config, "material_research", settings)
    label = spec.search_label()
    results: list = []

    try:
        results += await search(
            f"{spec.material_name} UNS AMS ASTM equivalent designation specification",
            budget,
            capability=Capability.SEARCH,
            depth=Depth.SNIPPETS,
            max_results=EVIDENCE_RESULTS,
            settings=settings,
        )
        results += await search(
            label,
            budget,
            capability=Capability.REFERENCE,
            depth=Depth.CONTENT,
            max_results=2,
            settings=settings,
        )
    except BudgetExhausted:
        log.info("material_research: evidence budget exhausted")

    if not results:
        log.warning(
            "material_research: no reference evidence found for %r; designations "
            "will be dropped rather than trusted",
            spec.material_name,
        )
        return ""

    extracts = "\n\n".join(
        f"--- {r.title} ({r.url})\n{result_text(r)[:2000]}" for r in results
    )
    return extracts[:MAX_EVIDENCE_CHARS]


def _verify(values: list[str] | None, corpus: str, kind: str) -> tuple[list[str], list[str]]:
    """Split `values` into those present in `corpus` and those that are not.

    Two independent tests, both required. Presence proves the term came from a
    source rather than from the model. `is_grade_specific` proves it identifies
    one alloy: "ASTM B348" covers every titanium bar grade, so it can be
    genuinely present in the corpus and still be useless as a search term - and
    worse than useless as a designation, because it makes a Grade 7 page look
    like a Grade 5 match.
    """
    haystack = normalise(corpus)
    kept: list[str] = []
    dropped: list[str] = []
    for value in values or []:
        value = (value or "").strip()
        if not value:
            continue
        if not is_grade_specific(value):
            dropped.append(f"{value} (not specific to one grade)")
        elif normalise(value) not in haystack:
            dropped.append(f"{value} (not found in the sources)")
        else:
            kept.append(value)
    if dropped:
        log.warning(
            "material_research: dropped %d unverified %s: %s",
            len(dropped), kind, "; ".join(dropped),
        )
    return kept, dropped


async def material_research(state: SessionState, config: RunnableConfig = None) -> dict:
    """Resolve designations, synonyms, and — critically — ambiguities."""
    spec = state.get("material_spec")
    if spec is None:
        raise ValueError("material_research ran before a spec was available")

    settings = get_settings()
    evidence = await _gather_evidence(spec, settings, config)

    model = get_model(
        "reasoning", schema=MaterialResearch, settings=settings,
        session_id=thread_id_of(config),
    )
    research: MaterialResearch = await model.ainvoke(
        [
            SystemMessage(content=SYSTEM),
            HumanMessage(
                content=(
                    f"Material specification:\n{spec.model_dump_json(indent=2)}\n\n"
                    f"REFERENCE EXTRACTS:\n{evidence or '(no sources were retrieved)'}"
                )
            ),
        ]
    )

    designations, dropped_designations = _verify(
        research.designations, evidence, "designation(s)"
    )
    synonyms, dropped_synonyms = _verify(research.synonyms, evidence, "synonym(s)")

    # Recorded on the result rather than only logged: a thin designation list
    # should be legible as "we could not source these" rather than looking like
    # the material simply has no equivalents.
    notes = [research.notes] if research.notes else []
    if dropped_designations or dropped_synonyms:
        notes.append(
            "Dropped as unverified against the retrieved sources: "
            + "; ".join(dropped_designations + dropped_synonyms)
        )
    if not evidence:
        notes.append(
            "No reference sources could be retrieved, so no designation or "
            "synonym could be verified."
        )

    research = research.model_copy(
        update={
            "designations": designations,
            "synonyms": synonyms,
            "notes": " ".join(notes) or None,
        }
    )

    if research.ambiguities:
        log.warning(
            "material_research: %d ambiguity(ies) for %s - %s",
            len(research.ambiguities),
            spec.material_name,
            research.ambiguities,
        )
    log.info(
        "material_research: %s -> %d verified designation(s), %d synonym(s)",
        research.canonical_name,
        len(designations),
        len(synonyms),
    )

    return {"research": research, "status": "researching"}
