"""Archive the finished run, then end.

Takes the place the cache write used to occupy, and inherits its one important
property: this runs after the results are already good, so a storage failure
must never turn a successful run into a failed one.

What it does *not* inherit is any influence on future runs. Nothing reads these
files back except a person asking for them.
"""

from __future__ import annotations

import logging

from langchain_core.runnables import RunnableConfig

from procurement_agent.archive import save_run as write_run
from procurement_agent.config import get_settings
from procurement_agent.graph.context import run_budget, thread_id_of, trace_of
from procurement_agent.graph.state import SessionState

log = logging.getLogger(__name__)


async def save_run(state: SessionState, config: RunnableConfig = None) -> dict:
    """Write the run to disk. Best-effort: never fails the run."""
    settings = get_settings()
    if not settings.enable_run_archive:
        return {}

    thread_id = thread_id_of(config) or "unknown"
    trace = trace_of(config)
    budget = run_budget(config, settings)

    # The budget lives in run config, so the credit count in state is only as
    # fresh as the last node that wrote it. Take the real remainder here, where
    # the run is actually over.
    state = {
        **dict(state),
        "search_credits_remaining": budget.credits_remaining,
        "truncated": bool(state.get("truncated")) or budget.truncated,
    }

    try:
        path = write_run(state, thread_id, settings, trace.as_dict() if trace else None)
    except Exception as exc:  # noqa: BLE001 - the results already stand
        log.warning("save_run failed (results are unaffected): %s", exc)
        return {}

    log.info(
        "save_run: archived %d vendor(s) to %s (%d credit(s) left)",
        len(state.get("vendor_leads", [])),
        path,
        budget.credits_remaining,
    )
    return {
        "archived_to": str(path),
        "search_credits_remaining": budget.credits_remaining,
        "truncated": state["truncated"],
    }
