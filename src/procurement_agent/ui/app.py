"""A chat UI that calls the graph directly — no FastAPI in the loop.

Run with:

    uv run streamlit run src/procurement_agent/ui/app.py

Needs exactly what the CLI needs: `.env` loaded and Postgres reachable
(`docker compose up -d postgres`). Nothing else — it opens the graph the same
way `cli.py` does, so it inherits the "works with no provider API keys, on a
local Ollama daemon" property for free.

Streamlit reruns this whole script on every interaction and has no native
async support in the script body, so each turn is driven exactly like
`cmd_new`/`cmd_resume` in `cli.py`: open the graph, stream it, read the final
state, close the graph — via `asyncio.run(...)` called from a synchronous
Streamlit callback. `open_graph()` also closes the crawl4ai browser and the
search client at the end of every call, so a long-lived Streamlit server pays
that startup cost once per chat turn rather than once per process, the way the
CLI does. Accepted here rather than fixed: it is the already-tested pattern,
and the fix if turn latency ever becomes the bottleneck (rather than the
unavoidable 2-3 minute graph run) is `st.cache_resource` holding one long-lived
graph, not a rewrite of this file.

The one real design point: a clarification interrupt renders as a normal
assistant chat bubble, tagged but not boxed into a form, and the *same*
`st.chat_input()` the user types a material request into is what they type
their answer into. There is no second widget and no mode the user has to
notice — matching the free-text `answers` shape `API_CONTRACT.md` already
documents for the HTTP client.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import uuid
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from langgraph.types import Command

load_dotenv()

from procurement_agent.config import get_settings  # noqa: E402
from procurement_agent.graph.build import (  # noqa: E402
    initial_state,
    open_graph,
    run_config,
)

log = logging.getLogger(__name__)

#: Shown once under every result. Deliberately this app's own copy rather than
#: importing either of the two disclaimer strings already in the codebase
#: (`api.sse.EXPORT_DISCLAIMER`, `archive.DISCLAIMER`): they are not
#: interchangeable — sse.py's is also served by two live FastAPI endpoints and
#: carries a sentence archive's does not — so reusing either would mean
#: pulling FastAPI into a script that has nothing to do with it, or quietly
#: changing what a live API response says. Not worth it for one string.
DISCLAIMER = (
    "Contact details are extracted from the linked source pages and verified "
    "against them, but should be confirmed before use. Material sourced through "
    "this tool may be subject to export control (ITAR/EAR); this tool does not "
    "assess or enforce export eligibility."
)

_KIND_COLOR = {
    "manufacturer": "green",
    "distributor": "blue",
    "trader": "orange",
    "retail": "gray",
    "not_a_supplier": "red",
    "unknown": "gray",
}


def _setup_logging() -> None:
    """Mirror the CLI's logging so node-level log lines land in the terminal
    running `streamlit run`, not just in Streamlit's own console."""
    if logging.getLogger().handlers:
        return  # basicConfig is a no-op with handlers present; guard anyway
    logging.basicConfig(
        level=get_settings().log_level.upper(),
        format="%(levelname)-7s %(name)-45s %(message)s",
        stream=sys.stderr,
    )


# --------------------------------------------------------------------------- #
# Running one turn
# --------------------------------------------------------------------------- #


async def _run_turn(payload: Any, thread_id: str, credits_remaining: int | None, on_event):
    """Open the graph, stream one turn, return the resulting checkpoint state.

    `payload` is `initial_state(text)` for a fresh thread or `Command(resume=
    text)` for an existing one. `credits_remaining` seeds the run's search
    budget from the last known value — omitting this on a resume was a real
    bug earlier in this project: the per-run ceiling silently reset to the full
    session default every time a conversation continued.
    """
    async with open_graph() as graph:
        config = run_config(thread_id, graph=graph, credits_remaining=credits_remaining)
        async for chunk in graph.astream(payload, config, stream_mode="updates"):
            for node_name, update in chunk.items():
                # LangGraph reports a suspension as `__interrupt__`, whose value
                # is a tuple of Interrupt objects rather than a state update.
                # The interrupt itself is read back below via aget_state, once
                # the stream has ended, so it is only skipped here.
                if node_name.startswith("__") or not isinstance(update, dict):
                    continue
                on_event(node_name, update)
        return await graph.aget_state(config)


def _progress_line(node_name: str, update: dict) -> str | None:
    """A human-readable line for a node's own return value.

    Calling the graph in-process rather than through the SSE API means `update`
    is the *full* dict a node returns, not the `{node, status}` shape
    `api/sse.py` forwards over the wire — so this can say something real about
    what happened, not just which node ran.
    """
    if node_name == "intake_parser":
        spec = update.get("material_spec")
        return f"📋 Parsed: **{spec.search_label()}**" if spec else None
    if node_name == "material_research":
        research = update.get("research")
        if research is None:
            return None
        n = len(research.designations or [])
        return f"🔬 Identified as **{research.canonical_name}** ({n} designation(s))"
    if node_name == "ask_clarification":
        spec = update.get("material_spec")
        return f"✏️ Updated: **{spec.search_label()}**" if spec else None
    if node_name == "research_sourcing":
        companies = update.get("sourcing_companies") or []
        if not companies:
            return None
        return f"📚 {len(companies)} supplier(s) named in the research literature"
    if node_name == "vendor_summary":
        leads = update.get("vendor_leads") or []
        return f"📊 Ranked {len(leads)} vendor(s)"
    # clarify_spec and save_run are deliberately silent: clarify_spec's
    # decision is covered by the interrupt bubble that follows it, and
    # save_run has nothing worth telling the buyer mid-run.
    return None


def _make_on_event(status_box) -> Any:
    """A progress callback bound to one `st.status` container.

    `vendor_search` and `contact_extraction` get special handling rather than
    going through `_progress_line`: `vendor_search` reports how many
    candidates it found, and `contact_extraction` fires once per candidate, so
    counting them against that total turns the run's longest, quietest stretch
    into a visible checklist instead of a silent wait.
    """
    counters = {"total_candidates": None, "verified": 0}

    def on_event(node_name: str, update: dict) -> None:
        if node_name == "vendor_search":
            candidates = update.get("vendor_candidates") or []
            counters["total_candidates"] = len(candidates)
            status_box.write(f"🔎 {len(candidates)} candidate vendor(s) found")
            return
        if node_name == "contact_extraction":
            leads = update.get("extracted_leads") or []
            if not leads:
                return
            counters["verified"] += 1
            total = counters["total_candidates"]
            suffix = f" ({counters['verified']}/{total})" if total else ""
            status_box.write(f"✓ {leads[0].company_name}{suffix}")
            return
        line = _progress_line(node_name, update)
        if line:
            status_box.write(line)

    return on_event


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def render_interrupt(payload: dict) -> None:
    """The clarification moment, as a normal chat bubble.

    `payload["questions"]` is already plain dicts — `ask_clarification` calls
    `interrupt({..., "questions": [q.model_dump() for q in questions], ...})`
    because an interrupt payload has to survive being checkpointed. No widgets
    per question: the reply is free text in the same input every other message
    goes through, not a structured per-field form.
    """
    st.badge("Needs your input", color="orange")
    st.write("I need a bit more detail before searching:")
    for q in payload.get("questions", []):
        st.markdown(f"**{q['question']}**")
        st.caption(q["why"])


def render_vendor(lead) -> None:
    with st.container(border=True):
        cols = st.columns([5, 2])
        with cols[0]:
            st.markdown(f"**{lead.company_name}**")
            st.write(lead.website)
        with cols[1]:
            st.badge(lead.kind, color=_KIND_COLOR.get(lead.kind, "gray"))

        details = []
        if lead.country:
            details.append(f"🌍 {lead.country}")
        if lead.contact_name:
            details.append(f"👤 {lead.contact_name}")
        if lead.email:
            details.append(f"✉️ {lead.email}")
        if lead.phone:
            details.append(f"📞 {lead.phone}")
        if details:
            st.write(" · ".join(details))

        if lead.certifications_found:
            st.caption("Certifications: " + ", ".join(lead.certifications_found))
        # A visible marker, not a silent drop: the vendor may still stock the
        # material, the page just did not say so outright.
        if not lead.mentions_material:
            st.caption("⚠️ This page did not explicitly name the material.")
        # These are warnings meant to be read, not debug output — never hidden
        # in a collapsed expander.
        for note in lead.confidence_notes:
            st.caption(f"⚠️ {note}")
        if lead.cited_in_research:
            st.caption("📄 Cited as a supplier in research literature")

        st.caption(f"Source: {lead.source_url}")
        if lead.contact_source_url and lead.contact_source_url != lead.source_url:
            st.caption(f"Contact verified on: {lead.contact_source_url}")


def render_final(values: dict) -> None:
    """The completed run, following the display rules in `API_CONTRACT.md`.

    These are domain rules, not styling preferences: `vendor_leads` arrives
    already ranked by whether a vendor can actually fill an industrial order,
    so it is rendered in that order and never re-sorted here.
    """
    spec = values.get("material_spec")
    research = values.get("research")
    leads = values.get("vendor_leads") or []
    truncated = values.get("truncated", False)
    credits = values.get("search_credits_remaining")

    if spec is not None:
        wanted = spec.search_label()
        if spec.quantity and spec.unit:
            wanted += f", {spec.quantity:g} {spec.unit}"
        st.markdown(f"**Searched for:** {wanted}")

    # This field matters most: it means the name could refer to more than one
    # material, and if it is wrong the whole list below is for the wrong
    # product — so it renders above the results, not in a details pane.
    if research is not None and research.ambiguities:
        st.warning(
            "**This material name may be ambiguous** — confirm before relying "
            "on the results below:\n\n"
            + "\n".join(f"- {a}" for a in research.ambiguities)
        )

    if truncated:
        st.info(
            "The search budget ran out before every angle was tried — this "
            "list may be incomplete."
        )

    if not leads:
        st.write("No vendors found.")
    else:
        st.markdown(f"**{len(leads)} vendor(s) found**")
        for lead in leads:
            render_vendor(lead)
        st.caption(DISCLAIMER)

    if credits is not None:
        st.caption(f"Search credits remaining this session: {credits}")


def render_error(content: dict) -> None:
    st.error(f"**{content.get('type', 'Error')}:** {content.get('detail', '')}")


def render_message(msg: dict) -> None:
    kind = msg["kind"]
    if kind == "text":
        st.write(msg["content"])
    elif kind == "interrupt":
        render_interrupt(msg["content"])
    elif kind == "final":
        render_final(msg["content"])
    elif kind == "error":
        render_error(msg["content"])


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #

st.set_page_config(page_title="Procurement Research Agent", page_icon="🔩")
_setup_logging()

st.session_state.setdefault("thread_id", None)
st.session_state.setdefault("awaiting_answer", False)
st.session_state.setdefault("credits_remaining", None)
st.session_state.setdefault("messages", [])

with st.sidebar:
    st.header("🔩 Procurement Research Agent")
    st.caption("Talks to the graph directly — no API server in the loop.")
    if st.button("🆕 New search", use_container_width=True):
        st.session_state.thread_id = None
        st.session_state.awaiting_answer = False
        st.session_state.credits_remaining = None
        st.session_state.messages = []
    st.divider()
    st.caption(
        "Runs the same graph the CLI and API use, against the same Postgres "
        "checkpoints. Works with no provider API keys at all, on a local "
        "Ollama daemon plus keyless search providers."
    )

st.title("Source a material")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        render_message(msg)

placeholder = (
    "Answer the question(s) above…"
    if st.session_state.awaiting_answer
    else "Describe the material you need — grade, form, quantity…"
)

if prompt := st.chat_input(placeholder):
    st.session_state.messages.append({"role": "user", "kind": "text", "content": prompt})

    resuming = st.session_state.awaiting_answer
    if resuming:
        thread_id = st.session_state.thread_id
        payload = Command(resume=prompt)
    else:
        # A brand-new thread, not only on the very first message: a thread
        # that already reached `final` cannot be resumed — sending it another
        # message would call Command(resume=...) on a graph that is not
        # suspended. `awaiting_answer` is only ever True right after an
        # interrupt, so this branch is correct for both "the very first
        # message" and "a new material request after a completed one".
        thread_id = str(uuid.uuid4())
        payload = initial_state(prompt)

    # The live status log below is rendered once, while this turn is running,
    # and then discarded by the `st.rerun()` at the end — it is not the
    # result. `render_interrupt`/`render_final` are deliberately NOT called in
    # here: content written inside `with status:` renders inside that
    # container, and a completed `st.status` visually collapses to a one-line
    # pill regardless of `expanded=True` at construction, which silently hid
    # the whole result the first time this was tried. Keeping this block to
    # progress lines only, and letting the rerun below draw the actual
    # interrupt/final message at the top level via the normal replay loop, is
    # what makes it land as a normal chat bubble rather than something
    # tucked inside a widget that looks finished and empty.
    with st.chat_message("assistant"):
        status = st.status("Researching…", expanded=True)
        with status:
            try:
                state = asyncio.run(
                    _run_turn(
                        payload,
                        thread_id,
                        st.session_state.credits_remaining,
                        _make_on_event(status),
                    )
                )
            except Exception as exc:  # noqa: BLE001 - shown to the user, not swallowed
                log.exception("turn failed")
                status.update(label="Something went wrong", state="error")
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "kind": "error",
                        "content": {"type": type(exc).__name__, "detail": str(exc)},
                    }
                )
                # thread_id/awaiting_answer are left exactly as they were: the
                # checkpoint on Postgres is unaffected by a client-side
                # exception, so the safest recovery is "try again", not
                # silently discarding state the user did not ask to lose.
            else:
                st.session_state.credits_remaining = state.values.get(
                    "search_credits_remaining"
                )
                if state.interrupts:
                    status.update(label="Waiting for your answer", state="complete")
                    st.session_state.thread_id = thread_id
                    st.session_state.awaiting_answer = True
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "kind": "interrupt",
                            "content": state.interrupts[0].value,
                        }
                    )
                else:
                    status.update(label="Done", state="complete")
                    st.session_state.thread_id = None  # see the comment above
                    st.session_state.awaiting_answer = False
                    st.session_state.messages.append(
                        {"role": "assistant", "kind": "final", "content": dict(state.values)}
                    )

    st.rerun()
