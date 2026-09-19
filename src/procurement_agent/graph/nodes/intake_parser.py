"""Parse the buyer's free-text request into a structured MaterialSpec.

Short input and a small schema, so this runs on the `extraction` tier: the
cheapest models handle it, and a measured bake-off found the 20b model matched
the 120b on exactly this task.
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from procurement_agent.graph.state import MaterialSpec, SessionState
from procurement_agent.llm.models import get_model

log = logging.getLogger(__name__)

SYSTEM = """You extract raw-material purchase specifications for an industrial \
procurement team.

Given a buyer's free-text request, populate the MaterialSpec fields from what is \
ACTUALLY STATED. Leave a field null when the request does not state it — do not \
infer a plausible value, and do not fill in a typical or default value. A null \
field triggers a clarifying question later, which is the correct outcome; a \
guessed value silently corrupts the vendor search.

Notes on specific fields:
- material_name: the common or trade name as written, e.g. "Custom 465", "Ti-6Al-4V".
- grade: only a formal designation (UNS, AMS, ASTM, EN). If the buyer wrote only \
a trade name, leave grade null.
- form: bar, rod, sheet, plate, tube, wire, forging, powder. Note that "Dia 2 inch" \
implies a round cross-section but NOT whether it is bar or tube — leave form null \
in that case.
- condition: temper or heat-treatment condition such as H900, annealed, solution treated.
"""


async def intake_parser(state: SessionState) -> dict:
    """Extract a MaterialSpec from `raw_input`."""
    raw = state.get("raw_input", "")
    model = get_model("extraction", schema=MaterialSpec)

    spec: MaterialSpec = await model.ainvoke(
        [SystemMessage(content=SYSTEM), HumanMessage(content=raw)]
    )
    log.info("intake_parser: parsed %s", spec.search_label())

    return {"material_spec": spec, "status": "clarifying"}
