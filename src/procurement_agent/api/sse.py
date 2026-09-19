"""Translate a graph run into a server-sent event stream.

The client needs to distinguish three outcomes: the run finished, the run
suspended for clarification, or it failed. An `interrupt` event carries the
questions so the client knows to prompt and call `/resume`, which is what makes
the conversational front half work over HTTP.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from procurement_agent.graph.state import VendorLead

log = logging.getLogger(__name__)

EXPORT_DISCLAIMER = (
    "Contact details are extracted from the linked source pages and verified "
    "against them, but should be confirmed before use. Material sourced through "
    "this tool may be subject to export control (ITAR/EAR); this tool does not "
    "assess or enforce export eligibility."
)


def _event(event: str, data: dict[str, Any]) -> dict[str, str]:
    return {"event": event, "data": json.dumps(data, default=str)}


def _serialise_leads(leads: list[VendorLead]) -> list[dict]:
    return [lead.model_dump() for lead in leads]


async def stream_run(graph, payload: Any, config: dict) -> AsyncIterator[dict[str, str]]:
    """Run the graph, emitting node progress and a terminal event.

    Always ends with exactly one of `interrupt`, `final`, or `error`, so a
    client can rely on the stream terminating in a known state.
    """
    try:
        async for chunk in graph.astream(payload, config, stream_mode="updates"):
            for node_name, update in chunk.items():
                # LangGraph reports a suspension as `__interrupt__`, whose value
                # is a tuple of Interrupt objects rather than a state update.
                # The interrupt itself is emitted below, from the resolved
                # state, so it is only skipped here.
                if node_name.startswith("__") or not isinstance(update, dict):
                    continue
                yield _event(
                    "node_end",
                    {"node": node_name, "status": update.get("status")},
                )

        state = await graph.aget_state(config)

        if state.interrupts:
            interrupt_value = state.interrupts[0].value
            yield _event(
                "interrupt",
                {
                    "kind": interrupt_value.get("kind", "clarification"),
                    "questions": interrupt_value.get("questions", []),
                    "thread_id": config["configurable"]["thread_id"],
                },
            )
            return

        values = state.values
        yield _event(
            "final",
            {
                "thread_id": config["configurable"]["thread_id"],
                "status": values.get("status"),
                "material_spec": (
                    values["material_spec"].model_dump()
                    if values.get("material_spec")
                    else None
                ),
                "research": (
                    values["research"].model_dump() if values.get("research") else None
                ),
                "vendor_leads": _serialise_leads(values.get("vendor_leads", [])),
                "truncated": values.get("truncated", False),
                "search_credits_remaining": values.get("search_credits_remaining"),
                # Provenance, so a client can show that results were reused and
                # how confidently they were matched to this material.
                "archived_to": values.get("archived_to"),
                "disclaimer": EXPORT_DISCLAIMER,
            },
        )
    except Exception as exc:  # noqa: BLE001 - the client needs to hear about it
        log.exception("graph run failed")
        yield _event("error", {"detail": str(exc), "type": type(exc).__name__})
