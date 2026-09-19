"""What a node needs from the run config: its credit allocation and the trace.

Both live in the run config rather than in graph state because each holds a
lock, and a lock cannot be serialised into a checkpoint. This module is the one
place that knows that, so a node asks for what it needs and never reaches into
`config["configurable"]` itself.

The allocations below are the run's budget split between the nodes that search.
They are shares of one pool, not separate pools — `SessionBudget.sub()` takes
them from the run budget atomically and returns whatever goes unspent.
"""

from __future__ import annotations

import logging

from langchain_core.runnables import RunnableConfig

from procurement_agent.config import Settings, get_settings
from procurement_agent.search.client import SessionBudget, bind_trace, budget_of
from procurement_agent.trace import RunTrace

log = logging.getLogger(__name__)

#: Credits each searching node may draw from the run's pool.
#:
#: Most of these are nominal in practice, because the providers they reach are
#: keyless and therefore free: the spec lookup and half of the material research
#: go to Wikipedia, and the whole academic pass goes to arXiv, PubMed and
#: Crossref. They are allocated anyway so that a run which *does* fall through
#: to a metered provider still cannot quietly outspend the vendor search, which
#: is what the run actually exists to do and which gets the remainder.
ALLOCATIONS: dict[str, int] = {
    # One reference lookup, on the only node a person is waiting on.
    "clarify_spec": 1,
    # One metered web search plus one free reference lookup.
    "material_research": 3,
    # Shared across the entire contact_extraction fan-out, not per candidate.
    "contact_extraction": 4,
}


def trace_of(config: RunnableConfig | None) -> RunTrace | None:
    """The run's trace, if this run has one."""
    return ((config or {}).get("configurable") or {}).get("run_trace")


def thread_id_of(config: RunnableConfig | None) -> str | None:
    return ((config or {}).get("configurable") or {}).get("thread_id")


def run_budget(
    config: RunnableConfig | None, settings: Settings | None = None
) -> SessionBudget:
    """The whole run's budget. For `vendor_search`, which gets the remainder."""
    _attach_trace(config)
    return budget_of(config, settings)


async def allocation(
    config: RunnableConfig | None,
    name: str,
    settings: Settings | None = None,
    credits: int | None = None,
) -> SessionBudget:
    """This node's share of the run budget.

    Named allocations are memoised on the parent, so every branch of a fan-out
    calling this with the same name receives the *same* pool rather than one
    each.
    """
    _attach_trace(config)
    settings = settings or get_settings()
    if credits is None:
        credits = ALLOCATIONS.get(name, 1)
    return await run_budget(config, settings).sub(credits, name)


def _attach_trace(config: RunnableConfig | None) -> None:
    """Point search-event recording at this run.

    Called on the way in to every node because contextvars are per task and
    LangGraph runs each node — and each fan-out branch — as its own task, so a
    value set in one does not carry into the next.
    """
    bind_trace(trace_of(config))
