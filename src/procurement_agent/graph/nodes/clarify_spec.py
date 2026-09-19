"""Ask only what is genuinely ambiguous, then resume.

Deliberately not a fixed question checklist. The model decides what is missing
given the parsed spec, so a fully specified request passes straight through
without interrogating the buyer about things they already told us.

One reference lookup runs before the questions are written. Without it the model
is guessing at what an unfamiliar trade name even refers to, and questions built
on a wrong guess are worse than no questions - they read as authoritative and
lead the buyer into confirming the wrong material. The lookup is deliberately
minimal: this is the only node a person is sitting waiting on, so it is a single
snippet-depth call, it never blocks on failure, and `enable_spec_lookup` turns it
off entirely.
"""

from __future__ import annotations

import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from procurement_agent.config import get_settings
from procurement_agent.graph.context import allocation, thread_id_of
from procurement_agent.graph.state import (
    ClarificationQuestion,
    MaterialSpec,
    SessionState,
    _nullable_list,
)
from procurement_agent.llm.models import get_model
from procurement_agent.search.client import BudgetExhausted, result_text, search
from searchroute import Capability, Depth

log = logging.getLogger(__name__)

#: Cap on the looked-up text. This is context for writing three questions, not
#: research material - `material_research` does the real work later.
MAX_LOOKUP_CHARS = 2000

#: Never ask more than this at once, however much is missing — a wall of
#: questions reads as an interrogation and buyers abandon it.
MAX_QUESTIONS = 3

SYSTEM = """You are helping an industrial procurement team pin down a raw-material \
specification before searching for vendors.

You are given a partially parsed MaterialSpec. Decide what genuinely MUST be \
clarified before a vendor search would return useful results, and ask at most \
{max_questions} questions.

Ask about a field ONLY when not knowing it would materially change which vendors \
or which product you would search for. Typically decisive:
- form (bar vs tube vs sheet) — different vendors entirely
- condition/temper — often a different product line or price
- the applicable standard, when a buyer's industry implies several

Typically NOT worth asking:
- anything already present in the spec
- commercial details such as delivery date, budget, or payment terms — out of scope here
- precision beyond what a vendor search needs

If the spec is already good enough to search on, return an EMPTY question list. \
Being able to proceed silently is a success, not a failure.

You may be given a REFERENCE LOOKUP for the material name. Use it only to \
understand what the buyer is asking for — especially whether the name is \
ambiguous and therefore worth asking about. Do not treat it as established fact \
about this order, and do not quote designations from it back to the buyer as \
though they were confirmed.
"""

APPLY_SYSTEM = """Update the MaterialSpec using the buyer's answers to the \
clarifying questions.

Apply only what the answers actually establish. If an answer is vague or the \
buyer says they do not know, leave that field null rather than guessing — a wrong \
value here sends the vendor search after the wrong product.

Two rules that matter more than they look:

1. NEVER replace material_name with a broader category. If the buyer said \
"Custom 465" and then explained it is a stainless steel, material_name stays \
"Custom 465" — that is the term vendors actually list it under. Generalising it \
to "Stainless Steel" destroys the vendor search. Put the formal designation in \
`grade` instead, and leave the trade name where it is.

2. Never drop or overwrite a field the buyer already gave you and did not \
change. You are adding detail, not rewriting the spec.
"""


#: Named without a leading underscore deliberately. LangChain derives the tool
#: name from the class, and Mistral rejected the result with "Unknown tool type:
#: 'Extraction'. Available tools: _Extraction" - the underscore is stripped
#: somewhere in the round trip and the name then fails to match. It cost a
#: ladder step on every Mistral call before this was found, which is invisible
#: unless you read the recorded attempts on a trace.
class Questions(BaseModel):
    questions: _nullable_list(ClarificationQuestion) = Field(default_factory=list)


async def clarify_spec(state: SessionState, config: RunnableConfig = None) -> dict:
    """Decide what to ask, and commit that decision before anyone is asked.

    Deliberately does **not** interrupt. Deciding and asking are separate nodes
    because a node containing `interrupt()` is re-executed from the top when the
    answers arrive, and this half is not idempotent: it calls a model, and the
    model may reach a different conclusion the second time.

    That is not hypothetical. A resume that had failed once was retried, the
    retry landed on a different provider, that provider decided no clarification
    was needed, and the early return below discarded the buyer's answer without
    a word — the run then searched for "Hastelloy", a family of dozens of
    alloys, instead of the "Hastelloy C-276" they had just typed.

    With the decision committed to `pending_questions` first, the re-executed
    half reads it back rather than asking again, so the answer cannot be
    stranded by a model changing its mind.
    """
    spec = state.get("material_spec")
    if spec is None:
        raise ValueError("clarify_spec ran before intake_parser produced a spec")

    lookup = await _lookup_material(spec, config)

    asked = get_model("clarification", schema=Questions, session_id=thread_id_of(config))
    result: Questions = await asked.ainvoke(
        [
            SystemMessage(content=SYSTEM.format(max_questions=MAX_QUESTIONS)),
            HumanMessage(
                content=(
                    f"Buyer's original request:\n{state.get('raw_input', '')}\n\n"
                    f"Parsed so far:\n{spec.model_dump_json(indent=2)}"
                    + (f"\n\nREFERENCE LOOKUP:\n{lookup}" if lookup else "")
                )
            ),
        ]
    )
    questions = result.questions[:MAX_QUESTIONS]

    if not questions:
        log.info("clarify_spec: spec is sufficient, skipping clarification")
        return {"status": "researching", "pending_questions": []}

    log.info("clarify_spec: asking %d question(s)", len(questions))
    return {"pending_questions": questions, "status": "clarifying"}


def route_after_clarify(state: SessionState) -> str:
    """Ask the buyer, or get on with the research."""
    return "ask_clarification" if state.get("pending_questions") else "material_research"


async def ask_clarification(
    state: SessionState, config: RunnableConfig = None
) -> dict:
    """Suspend for the buyer's answers, then fold them into the spec.

    The interrupt persists the checkpoint, so the answers can arrive minutes
    later from a different process via `Command(resume=...)`.

    Everything this node needs is already in state, which is what makes it safe
    to re-execute: it reads the questions rather than generating them, so a
    replay asks the same thing and applies the answer to the same spec.
    """
    spec = state.get("material_spec")
    questions = list(state.get("pending_questions") or [])
    if spec is None or not questions:
        # Nothing to ask. Reachable only if state was manipulated directly.
        return {"pending_questions": [], "status": "researching"}

    # Suspends here. Everything above has already been checkpointed.
    answers = interrupt(
        {
            "kind": "clarification",
            "questions": [q.model_dump() for q in questions],
            "spec": spec.model_dump(),
        }
    )

    updated = _protect_spec(spec, await _apply_answers(spec, questions, answers))
    log.info("ask_clarification: resumed, spec now %s", updated.search_label())

    return {
        "material_spec": updated,
        "pending_questions": [],
        "clarification_history": [
            *state.get("clarification_history", []),
            {
                "questions": [q.model_dump() for q in questions],
                "answers": answers,
            },
        ],
        "messages": [
            AIMessage(content="\n".join(f"- {q.question}" for q in questions)),
            HumanMessage(content=_render_answers(answers)),
        ],
        "status": "researching",
    }


async def _lookup_material(spec: MaterialSpec, config: RunnableConfig = None) -> str:
    """One reference lookup on the material name, or "" if anything goes wrong.

    Entirely optional by construction. The buyer is waiting on this node, so a
    slow or failing lookup must cost them nothing beyond the time already spent:
    every failure path returns an empty string and the questions are written
    without it, exactly as they were before.
    """
    settings = get_settings()
    if not settings.enable_spec_lookup:
        return ""

    budget = await allocation(config, "clarify_spec", settings)
    try:
        results = await search(
            spec.material_name,
            budget,
            capability=Capability.REFERENCE,
            depth=Depth.SNIPPETS,
            max_results=2,
            settings=settings,
        )
    except BudgetExhausted:
        return ""

    if not results:
        log.info("clarify_spec: no reference entry for %r", spec.material_name)
        return ""
    return "\n".join(
        f"{r.title}: {result_text(r)[:MAX_LOOKUP_CHARS]}" for r in results
    )[:MAX_LOOKUP_CHARS]


def _protect_spec(original: MaterialSpec, updated: MaterialSpec) -> MaterialSpec:
    """Stop a clarification pass from degrading the spec it was meant to sharpen.

    Two failure modes seen in practice, both from weaker models: generalising the
    trade name ("Custom 465" -> "Stainless Steel"), which wrecks the vendor
    search because vendors list the trade name; and nulling out fields the buyer
    already supplied. Clarification may only add detail, so both are reverted
    here rather than left to the prompt.
    """
    updates: dict[str, object] = {}

    old_name = original.material_name.strip()
    new_name = updated.material_name.strip()
    # A refinement contains the original ("Custom 465" -> "Custom 465 bar");
    # anything else is a substitution and gets reverted.
    if old_name and old_name.casefold() not in new_name.casefold():
        log.warning(
            "clarify_spec: reverting material_name %r -> %r (clarification must "
            "not rename the material)",
            new_name,
            old_name,
        )
        updates["material_name"] = old_name

    # Never let a clarification erase something the buyer already told us.
    for field in ("grade", "form", "dimensions", "quantity", "unit", "condition"):
        if getattr(updated, field) is None and getattr(original, field) is not None:
            updates[field] = getattr(original, field)
    if not updated.standards and original.standards:
        updates["standards"] = original.standards

    return updated.model_copy(update=updates) if updates else updated


async def _apply_answers(
    spec: MaterialSpec, questions: list[ClarificationQuestion], answers: object
) -> MaterialSpec:
    """Fold the buyer's answers back into the spec."""
    model = get_model("clarification", schema=MaterialSpec)
    qa = "\n".join(f"Q: {q.question}" for q in questions)
    return await model.ainvoke(
        [
            SystemMessage(content=APPLY_SYSTEM),
            HumanMessage(
                content=(
                    f"Current spec:\n{spec.model_dump_json(indent=2)}\n\n"
                    f"Questions asked:\n{qa}\n\n"
                    f"Buyer's answers:\n{_render_answers(answers)}"
                )
            ),
        ]
    )


def _render_answers(answers: object) -> str:
    """Accept whatever shape the caller resumed with: str, list, or dict."""
    if isinstance(answers, str):
        return answers
    if isinstance(answers, dict):
        return "\n".join(f"{k}: {v}" for k, v in answers.items())
    if isinstance(answers, list):
        return "\n".join(str(a) for a in answers)
    return str(answers)
