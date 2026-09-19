"""Graph wiring and checkpointer lifecycle.

`open_graph` currently runs on `InMemorySaver`, chosen for Streamlit Cloud
deployment where a Postgres instance isn't available. This means `interrupt()`
state does not survive a process restart - a clarification question must be
answered in the same process that asked it. `open_checkpointer` below still
builds the Postgres-backed saver; swap `open_graph` back to it (see the
comment inline) once a Postgres instance is available.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph

from procurement_agent.config import Settings, get_settings
from procurement_agent.crawl.fetcher import close_fetcher
from procurement_agent.graph.nodes.clarify_spec import (
    ask_clarification,
    clarify_spec,
    route_after_clarify,
)
from procurement_agent.graph.nodes.contact_extraction import contact_extraction
from procurement_agent.graph.nodes.draft_outreach_email import draft_outreach_email
from procurement_agent.graph.nodes.intake_parser import intake_parser
from procurement_agent.graph.nodes.material_research import material_research
from procurement_agent.graph.nodes.research_sourcing import research_sourcing
from procurement_agent.graph.nodes.save_run import save_run
from procurement_agent.graph.nodes.vendor_search import (
    fan_out_to_extraction,
    vendor_search,
)
from procurement_agent.graph.nodes.vendor_summary import vendor_summary
from procurement_agent.graph.state import SessionState
from procurement_agent.search.client import SessionBudget
from procurement_agent.search.client import close_client as close_search_client
from procurement_agent.trace import TRACED_NODES, RunTrace, build_callbacks

log = logging.getLogger(__name__)

#: Our own Pydantic models get written into checkpoints and must be explicitly
#: allow-listed for deserialisation. LangGraph warns about unregistered types
#: today and will refuse them outright in a future version, so this is declared
#: rather than left to the default.
_STATE_MODULE = "procurement_agent.graph.state"
ALLOWED_MSGPACK_MODULES = (
    (_STATE_MODULE, "MaterialSpec"),
    (_STATE_MODULE, "MaterialResearch"),
    (_STATE_MODULE, "ClarificationQuestion"),
    (_STATE_MODULE, "VendorCandidate"),
    (_STATE_MODULE, "VendorLead"),
    (_STATE_MODULE, "EmailDraft"),
)


def make_serde() -> JsonPlusSerializer:
    """Checkpoint serializer that knows about this app's state models."""
    return JsonPlusSerializer(allowed_msgpack_modules=ALLOWED_MSGPACK_MODULES)


def build_graph() -> StateGraph:
    """Wire the nodes. Compilation (and the checkpointer) happens separately."""
    graph = StateGraph(SessionState)

    graph.add_node("intake_parser", intake_parser)
    graph.add_node("clarify_spec", clarify_spec)
    graph.add_node("ask_clarification", ask_clarification)
    graph.add_node("material_research", material_research)
    graph.add_node("research_sourcing", research_sourcing)
    graph.add_node("save_run", save_run)
    graph.add_node("vendor_search", vendor_search)
    graph.add_node("contact_extraction", contact_extraction)
    graph.add_node("vendor_summary", vendor_summary)
    graph.add_node("draft_outreach_email", draft_outreach_email)

    graph.add_edge(START, "intake_parser")
    graph.add_edge("intake_parser", "clarify_spec")

    # Deciding what to ask and actually asking are separate nodes. A node that
    # calls `interrupt()` is re-executed from the top when the answer arrives,
    # so the model call that decides the questions must not live inside it -
    # it can reach a different conclusion on the replay and strand the answer.
    graph.add_conditional_edges(
        "clarify_spec",
        route_after_clarify,
        ["ask_clarification", "material_research"],
    )
    graph.add_edge("ask_clarification", "material_research")

    graph.add_edge("material_research", "research_sourcing")

    # Sequential, not parallel: research_sourcing produces supplier *names*,
    # which vendor_search then turns into queries to find contact pages. A
    # paper names a company but never how to reach it.
    graph.add_edge("research_sourcing", "vendor_search")

    # Dynamic fan-out: one extraction branch per candidate, or straight to the
    # summary when the search found nothing.
    graph.add_conditional_edges(
        "vendor_search",
        fan_out_to_extraction,
        ["contact_extraction", "vendor_summary"],
    )
    graph.add_edge("contact_extraction", "vendor_summary")
    # The run is archived after the results are final. This is a record, never
    # a cache: nothing in the graph reads it back, so an archived run cannot
    # silently stand in for a fresh search.
    graph.add_edge("vendor_summary", "save_run")
    graph.add_edge("save_run", END)

    # Email drafting is a separate entry point, invoked on request against an
    # already-completed session rather than run as part of the research pass.
    graph.add_edge("draft_outreach_email", END)

    # The trace records an allowlist of node names, which would silently stop
    # recording a node renamed here. Cheap to check, and the failure it prevents
    # is a gap in a record nobody notices until they need it.
    missing = set(graph.nodes) - TRACED_NODES
    if missing:
        log.warning(
            "these graph nodes are absent from trace.TRACED_NODES and will not "
            "appear in any run record: %s",
            ", ".join(sorted(missing)),
        )

    return graph


@asynccontextmanager
async def open_checkpointer(
    settings: Settings | None = None,
) -> AsyncIterator[AsyncPostgresSaver]:
    """Open the Postgres checkpointer and ensure its tables exist."""
    settings = settings or get_settings()
    async with AsyncPostgresSaver.from_conn_string(
        settings.database_url, serde=make_serde()
    ) as saver:
        # Idempotent; safe to call on every start.
        await saver.setup()
        yield saver


@asynccontextmanager
async def open_graph(settings: Settings | None = None) -> AsyncIterator:
    """Compiled graph bound to an in-memory checkpointer.

    To swap to Postgres, replace the body with:
        settings = settings or get_settings()
        async with open_checkpointer(settings) as saver:
            ...
    """
    async with InMemorySaver() as saver:
        graph = build_graph().compile(checkpointer=saver)
        try:
            yield graph
        finally:
            await close_fetcher()
            await close_search_client()

def initial_state(raw_input: str, settings: Settings | None = None) -> dict:
    """Fresh session state for a new research request."""
    settings = settings or get_settings()
    return {
        "raw_input": raw_input,
        "messages": [],
        "clarification_history": [],
        "search_queries": [],
        "sourcing_companies": [],
        "vendor_candidates": [],
        "extracted_leads": [],
        "vendor_leads": [],
        "email_drafts": [],
        "search_credits_remaining": settings.search_credits_per_session,
        "truncated": False,
        "status": "clarifying",
    }


def run_config(
    thread_id: str,
    settings: Settings | None = None,
    graph=None,
    *,
    credits_remaining: int | None = None,
) -> dict:
    """Run configuration for one session.

    `max_concurrency` caps fan-out graph-wide. It is deliberately low: on these
    free tiers, wide parallelism produces simultaneous 429s rather than speed.

    The search budget and the trace both live here rather than in state, for the
    same reason the vendor cache used to: each holds a lock, and a lock cannot
    be serialised into a checkpoint. One budget per run is what makes
    `search_credits_per_session` mean the run rather than one node - nodes take
    sub-allocations from it instead of opening pools of their own.

    `credits_remaining` seeds the budget when resuming, so a session continued
    in a fresh process picks up where it left off rather than starting again
    with a full allowance.

    `graph` is accepted and unused, so existing callers keep working now that
    there is no connection pool to thread through.
    """
    settings = settings or get_settings()
    trace = RunTrace(thread_id=thread_id)
    return {
        "configurable": {
            "thread_id": thread_id,
            "search_budget": SessionBudget(credits=credits_remaining, settings=settings),
            "run_trace": trace,
        },
        "callbacks": build_callbacks(trace, thread_id, settings),
        "max_concurrency": settings.max_fanout,
    }
