"""Mine academic papers for the suppliers they name.

Researchers state where their material came from — in the methods section or
the acknowledgements — and those companies are often specialist mills and
stockists that an open-web search buries under marketplaces and SEO pages. That
makes this a genuinely different pool of vendors, not more of the same.

This node produces *company names*, not contacts. The names then become extra
queries in `vendor_search`, which is what finds the actual contact page. That
split matters: a paper will tell you "supplied by Ronald Britton Ltd" and
nothing else, so the web search still has to do the reachability work.

Searching is one `capability=ACADEMIC` call. That capability is a filter, not a
hint: it is what routes the query to arXiv, PubMed and Crossref, all of which
are keyless and free, and it is also what keeps a plain web search from
answering instead. Broadening from arXiv-alone to all three matters here,
because arXiv is physics-and-CS-heavy while the materials and metallurgy work
that names a mill is mostly in PubMed and Crossref's index.

The known weakness is depth, not breadth: these APIs return abstracts, and
sourcing statements live in the methods section and the acknowledgements. So the
hit rate is modest by construction, and content depth is requested to pull the
full text wherever a provider can supply it.

Coverage is uneven regardless: a specialist research-grade alloy does far better
here than a common structural steel, and a proprietary trade name may return
nothing at all. This node is enrichment, never a dependency — every failure path
returns an empty list rather than breaking the run.
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from procurement_agent.config import get_settings
from procurement_agent.extraction.patterns import (
    build_sourcing_digest,
    find_sourcing_mentions,
)
from procurement_agent.graph.context import allocation, thread_id_of
from procurement_agent.graph.state import SessionState, StrList
from procurement_agent.llm.models import get_model
from procurement_agent.search.client import BudgetExhausted, result_text, search
from searchroute import Capability, Depth

log = logging.getLogger(__name__)

MAX_PAPERS = 8
MAX_SUPPLIERS = 6

SYSTEM = """You are reading sourcing statements taken verbatim from research \
papers, and extracting the names of companies that supplied a material.

For each statement, decide whether it names a COMMERCIAL SUPPLIER of the \
material. Return only real company names.

Include: mills, stockists, distributors, specialty alloy producers, chemical \
and materials suppliers.

Exclude, because they are not places a buyer can purchase from:
- universities, laboratories, research institutes and their departments
- funding bodies, grant agencies and government programmes
- individual people's names
- the authors' own institution
- equipment or instrument manufacturers, when the statement is about apparatus \
rather than the material itself

Normalise each name to how the company trades (for example "Carpenter \
Technology Corporation" rather than "Carpenter Technology Corp., Philadelphia, \
PA, USA"), and drop any trailing address or location.

You may only return a company that appears in the statements given to you. Do \
not add well-known suppliers from your own knowledge, however plausible — the \
value here is that these names came from a real sourcing record. If no \
statement names a commercial supplier, return an empty list.
"""


#: Named without a leading underscore deliberately. LangChain derives the tool
#: name from the class, and Mistral rejected the result with "Unknown tool type:
#: 'Extraction'. Available tools: _Extraction" - the underscore is stripped
#: somewhere in the round trip and the name then fails to match. It cost a
#: ladder step on every Mistral call before this was found, which is invisible
#: unless you read the recorded attempts on a trace.
class Suppliers(BaseModel):
    company_names: StrList = Field(default_factory=list)


async def research_sourcing(state: SessionState, config: RunnableConfig = None) -> dict:
    """Search papers for the material, and extract supplier names they mention."""
    settings = get_settings()
    spec = state.get("material_spec")
    if spec is None:
        raise ValueError("research_sourcing ran before a spec was available")

    if not settings.enable_research_search:
        log.info("research_sourcing: disabled by configuration")
        return {"sourcing_companies": []}

    research = state.get("research")
    queries = [spec.material_name]
    if research:
        queries += [d for d in research.designations[:1] if d]

    # Its own budget, separate from the vendor search's: this pass is optional
    # enrichment and must never be able to eat the credits the vendor search
    # needs to do the job the run was actually asked for.
    budget = await allocation(
        config, "research_sourcing", settings, credits=settings.research_search_credits
    )

    papers: list = []
    for query in queries[:2]:
        try:
            papers += await search(
                f"{query} material supplied by",
                budget,
                capability=Capability.ACADEMIC,
                # Sourcing statements are in the methods section, so ask for
                # full text; the academic providers mostly return abstracts and
                # the response degrades to those on its own.
                depth=Depth.CONTENT,
                max_results=MAX_PAPERS,
                max_hydrate=MAX_PAPERS,
            )
        except BudgetExhausted:
            log.info("research_sourcing: budget exhausted after %d paper(s)", len(papers))
            break

    if not papers:
        log.info("research_sourcing: no papers found for %s", spec.material_name)
        return {"sourcing_companies": []}

    corpus = "\n\n".join(f"{p.title}\n{result_text(p)}" for p in papers)
    mentions = find_sourcing_mentions(corpus)
    if not mentions:
        log.info("research_sourcing: %d papers, no sourcing statements", len(papers))
        return {"sourcing_companies": []}

    model = get_model("extraction", schema=Suppliers, session_id=thread_id_of(config))
    try:
        result: Suppliers = await model.ainvoke(
            [
                SystemMessage(content=SYSTEM),
                HumanMessage(
                    content=(
                        f"Material: {spec.material_name}\n\n"
                        f"{build_sourcing_digest(mentions)}"
                    )
                ),
            ]
        )
    except Exception as exc:  # noqa: BLE001 - optional enrichment, never fatal
        log.warning("research_sourcing: supplier extraction failed: %s", exc)
        return {"sourcing_companies": []}

    # Only keep names that actually appeared in the papers. The prompt forbids
    # inventing well-known suppliers, but the whole value of this path is that
    # the names came from a real sourcing record, so it is verified rather than
    # trusted.
    corpus_lower = corpus.casefold()
    verified = [
        name
        for name in dict.fromkeys(result.company_names)
        if name.strip() and name.strip().casefold() in corpus_lower
    ]
    dropped = len(result.company_names) - len(verified)
    if dropped:
        log.warning(
            "research_sourcing: dropped %d supplier name(s) not present in the papers",
            dropped,
        )

    log.info(
        "research_sourcing: %d papers -> %d sourcing statements -> %d suppliers %s",
        len(papers),
        len(mentions),
        len(verified),
        verified,
    )
    return {"sourcing_companies": verified[:MAX_SUPPLIERS]}
