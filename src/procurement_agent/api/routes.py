"""Session endpoints."""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request, status
from langgraph.types import Command
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from procurement_agent import archive
from procurement_agent.api.sse import EXPORT_DISCLAIMER, stream_run
from procurement_agent.config import get_settings
from procurement_agent.graph.build import initial_state, run_config
from procurement_agent.security import require_api_key

log = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(require_api_key)])


class CreateSession(BaseModel):
    request: str = Field(description="The buyer's free-text material request")


class SessionCreated(BaseModel):
    thread_id: str


class MessageIn(BaseModel):
    content: str


class ResumeIn(BaseModel):
    answers: str = Field(description="Answers to the outstanding clarifying questions")


def _graph(request: Request):
    graph = getattr(request.app.state, "graph", None)
    if graph is None:  # pragma: no cover - only if lifespan failed
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="graph is not ready",
        )
    return graph


@router.post("/sessions", response_model=SessionCreated, status_code=201)
async def create_session() -> SessionCreated:
    """Allocate a thread id. Nothing runs until the first message."""
    return SessionCreated(thread_id=str(uuid.uuid4()))


@router.post("/sessions/{thread_id}/messages")
async def send_message(thread_id: str, body: MessageIn, request: Request):
    """Start (or continue) research on this thread, streamed as SSE.

    A thread that is already waiting on an interrupt treats an incoming message
    as the answer, so a client that reconnects and simply sends text does the
    right thing without having to call `/resume` explicitly.
    """
    graph = _graph(request)
    state = await graph.aget_state(run_config(thread_id, graph=graph))

    # An existing thread treats the message as a resume value, so a client that
    # reconnects and simply sends text does the right thing.
    resuming = bool(state.created_at)
    payload = Command(resume=body.content) if resuming else initial_state(body.content)

    # A resumed thread continues its budget; a new one starts with a full
    # allowance. Otherwise the per-run ceiling resets every time a client
    # reconnects, which is exactly when it should not.
    config = run_config(
        thread_id,
        graph=graph,
        credits_remaining=(
            state.values.get("search_credits_remaining") if resuming else None
        ),
    )
    return EventSourceResponse(stream_run(graph, payload, config))


@router.post("/sessions/{thread_id}/resume")
async def resume_session(thread_id: str, body: ResumeIn, request: Request):
    """Resume a suspended session with clarification answers."""
    graph = _graph(request)
    state = await graph.aget_state(run_config(thread_id, graph=graph))
    if not state.created_at:
        raise HTTPException(status_code=404, detail=f"no session {thread_id!r}")
    if not state.interrupts:
        raise HTTPException(
            status_code=409,
            detail="session is not waiting for clarification",
        )

    config = run_config(
        thread_id,
        graph=graph,
        credits_remaining=state.values.get("search_credits_remaining"),
    )
    return EventSourceResponse(stream_run(graph, Command(resume=body.answers), config))


@router.get("/sessions/{thread_id}")
async def get_session(thread_id: str, request: Request) -> dict:
    """Full session state, for reconnecting after an interrupt."""
    graph = _graph(request)
    state = await graph.aget_state(run_config(thread_id, graph=graph))
    if not state.created_at:
        raise HTTPException(status_code=404, detail=f"no session {thread_id!r}")

    values = state.values
    pending = (
        state.interrupts[0].value.get("questions", []) if state.interrupts else []
    )

    return {
        "thread_id": thread_id,
        "status": values.get("status"),
        "awaiting_clarification": bool(state.interrupts),
        "pending_questions": pending,
        "material_spec": (
            values["material_spec"].model_dump() if values.get("material_spec") else None
        ),
        "research": values["research"].model_dump() if values.get("research") else None,
        "search_queries": values.get("search_queries", []),
        "vendor_leads": [lead.model_dump() for lead in values.get("vendor_leads", [])],
        "email_drafts": [d.model_dump() for d in values.get("email_drafts", [])],
        "truncated": values.get("truncated", False),
        "search_credits_remaining": values.get("search_credits_remaining"),
        "archived_to": values.get("archived_to"),
        "disclaimer": EXPORT_DISCLAIMER,
    }


@router.post("/sessions/{thread_id}/emails")
async def draft_emails(thread_id: str, request: Request) -> dict:
    """Draft outreach emails for a completed session.

    Deliberately a separate call rather than part of the research run: it costs
    one model call per vendor, and most sessions never need it.
    """
    from procurement_agent.graph.nodes.draft_outreach_email import draft_outreach_email

    graph = _graph(request)
    config = run_config(thread_id, graph=graph)
    state = await graph.aget_state(config)
    if not state.created_at:
        raise HTTPException(status_code=404, detail=f"no session {thread_id!r}")

    leads = state.values.get("vendor_leads", [])
    if not leads:
        raise HTTPException(
            status_code=409, detail="session has no vendor leads to write to"
        )

    result = await draft_outreach_email(state.values)
    await graph.aupdate_state(config, result)

    return {
        "thread_id": thread_id,
        "email_drafts": [d.model_dump() for d in result["email_drafts"]],
        "disclaimer": EXPORT_DISCLAIMER,
    }


@router.get("/runs")
async def list_archived_runs(q: str | None = None, limit: int = 50) -> dict:
    """Archived runs, newest first.

    Index entries only — enough to find the run you want. `GET /runs/{id}`
    returns the whole thing.

    Nothing in the graph reads this; it exists solely because someone asked.
    That is the difference between an archive and the cache it replaced.
    """
    entries = archive.search_runs(q) if q else archive.list_runs()
    return {"count": len(entries), "runs": entries[: max(1, limit)]}


@router.get("/runs/{run_id}")
async def get_archived_run(run_id: str) -> dict:
    """One archived run, complete.

    Everything recorded: the request and clarifications, the material research,
    every query and search result, every vendor lead, and the trace of what the
    run actually did — which model served each node, what the routing ladder
    tried first, which search provider answered and what it cost.
    """
    record = archive.load_run(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"no archived run {run_id!r}")
    return record


@router.get("/health")
async def health() -> dict:
    settings = get_settings()
    return {
        "status": "ok",
        "search_credits_per_session": settings.search_credits_per_session,
        "max_fanout": settings.max_fanout,
        "archived_runs": len(archive.list_runs(settings)),
    }
