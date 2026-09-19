"""Terminal driver for the graph.

Exists mainly to prove the persistence story: `--new` runs until the graph
interrupts for clarification and prints the thread id, and `--resume` continues
that session from a *different process*. If that works, the checkpointer is
doing its job.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import uuid

from dotenv import load_dotenv
from langgraph.types import Command

from procurement_agent.config import get_settings
from procurement_agent.graph.build import initial_state, open_graph, run_config
from procurement_agent.graph.state import VendorLead


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(levelname)-7s %(name)-45s %(message)s",
        stream=sys.stderr,
    )


def _print_interrupt(payload: dict) -> None:
    print("\n\033[1mI need a few details before searching:\033[0m\n")
    for i, question in enumerate(payload.get("questions", []), 1):
        print(f"  {i}. {question['question']}")
        print(f"     \033[2mwhy: {question['why']}\033[0m")


def _print_archive_line(values: dict) -> None:
    """Say where the run was filed, so it can be found again."""
    path = values.get("archived_to")
    if path:
        print(f"\033[2mArchived to {path}\033[0m")


def _print_leads(leads: list[VendorLead], truncated: bool) -> None:
    if not leads:
        print("\nNo vendors found.")
        return

    print(f"\n\033[1m{len(leads)} vendor(s) found\033[0m\n")
    for i, lead in enumerate(leads, 1):
        print(f"\033[1m{i}. {lead.company_name}\033[0m  [{lead.kind}]")
        print(f"   {lead.website}")
        if lead.country:
            print(f"   country: {lead.country}")
        if lead.contact_name:
            print(f"   contact: {lead.contact_name}")
        if lead.email:
            print(f"   email:   {lead.email}")
        if lead.phone:
            print(f"   phone:   {lead.phone}")
        if lead.certifications_found:
            print(f"   certs:   {', '.join(lead.certifications_found)}")
        print(f"   source:  {lead.source_url}")
        # Shown separately when the contact came from a different page, since
        # that page is what the email and phone were actually verified against.
        if lead.contact_source_url and lead.contact_source_url != lead.source_url:
            print(f"   contact verified on: {lead.contact_source_url}")
        for note in lead.confidence_notes:
            print(f"   \033[33m! {note}\033[0m")
        print()

    if truncated:
        print(
            "\033[33mNote: the search budget was exhausted, so this list is "
            "incomplete.\033[0m"
        )
    print(
        "\033[2mContact details are extracted from the linked source pages and "
        "verified against them; always confirm before use. Material may be "
        "subject to export control (ITAR/EAR).\033[0m"
    )


async def _drain(graph, payload, config) -> dict:
    """Run the graph to completion or to its next interrupt."""
    async for _ in graph.astream(payload, config, stream_mode="updates"):
        pass
    return await graph.aget_state(config)


async def cmd_new(request: str) -> int:
    thread_id = str(uuid.uuid4())
    async with open_graph() as graph:
        config = run_config(thread_id, graph=graph)
        state = await _drain(graph, initial_state(request), config)

        if state.interrupts:
            _print_interrupt(state.interrupts[0].value)
            print(f"\n\033[1mthread id:\033[0m {thread_id}")
            print(
                "resume with:\n"
                f'  uv run python -m procurement_agent.cli --resume {thread_id} '
                f'--answer "your answers here"'
            )
            return 0

        values = state.values
        _print_archive_line(values)
        _print_leads(values.get("vendor_leads", []), values.get("truncated", False))
    return 0


async def cmd_resume(thread_id: str, answer: str) -> int:
    async with open_graph() as graph:
        current = await graph.aget_state(run_config(thread_id, graph=graph))
        if not current.created_at:
            print(f"no session found for thread id {thread_id!r}", file=sys.stderr)
            return 1

        # Continue the run's budget rather than handing a resumed session a
        # fresh allowance. Without this the ceiling only holds within one
        # process, and answering a clarification would reset it.
        config = run_config(
            thread_id,
            graph=graph,
            credits_remaining=current.values.get("search_credits_remaining"),
        )

        state = await _drain(graph, Command(resume=answer), config)
        if state.interrupts:
            _print_interrupt(state.interrupts[0].value)
            print(f"\n\033[1mthread id:\033[0m {thread_id}")
            return 0

        values = state.values
        _print_archive_line(values)
        _print_leads(values.get("vendor_leads", []), values.get("truncated", False))
    return 0


async def cmd_show(thread_id: str) -> int:
    async with open_graph() as graph:
        state = await graph.aget_state(run_config(thread_id, graph=graph))
        if not state.created_at:
            print(f"no session found for thread id {thread_id!r}", file=sys.stderr)
            return 1
        values = state.values
        spec = values.get("material_spec")
        print(f"status: {values.get('status')}")
        print(f"credits left: {values.get('search_credits_remaining')}")
        _print_archive_line(values)
        if spec:
            print(f"spec: {spec.model_dump_json(indent=2)}")
        _print_leads(values.get("vendor_leads", []), values.get("truncated", False))
    return 0


async def cmd_validate() -> int:
    from procurement_agent.llm.models import describe_routing, validate_models

    problems = await validate_models()
    for line in describe_routing():
        print(f"  {line}")
    if not problems:
        print("\nevery task has a usable endpoint")
        print(
            "\033[2mTo check the model IDs themselves against each provider's live "
            "catalogue: uv run python LLMRoute/scripts/verify_models.py\033[0m"
        )
        return 0
    print("\n\033[33mrouting warnings:\033[0m")
    for problem in problems:
        print(f"  - {problem}")
    return 0


def cmd_list_runs() -> int:
    """Archived runs, newest first. Never consulted by the graph."""
    from procurement_agent.archive import list_runs

    entries = list_runs()
    if not entries:
        print("no archived runs yet")
        return 0
    for entry in entries:
        print(f"\033[1m{entry['run_id']}\033[0m  ({entry['vendor_count']} vendor(s))")
        print(f"   {entry['saved_at']}")
        print(f"   request: {entry['request']}")
        if entry["keywords"]:
            print(f"   keywords: {', '.join(entry['keywords'])}")
        print()
    return 0


def cmd_show_run(identifier: str) -> int:
    from procurement_agent.archive import load_run

    record = load_run(identifier)
    if record is None:
        print(f"no archived run matching {identifier!r}", file=sys.stderr)
        return 1

    print(f"\033[1m{record['run_id']}\033[0m  saved {record['saved_at']}")
    print(f"\nrequest: {record['request']}")
    print(f"\n{record['summary']}")
    if record.get("keywords"):
        print(f"\nkeywords: {', '.join(record['keywords'])}")
    if record.get("search_queries"):
        print("\nqueries:")
        for query in record["search_queries"]:
            print(f"  - {query}")
    leads = [VendorLead.model_validate(lead) for lead in record.get("vendor_leads", [])]
    _print_leads(leads, record.get("truncated", False))
    return 0


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="procurement-agent")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--new", metavar="REQUEST", help="start a new research session")
    group.add_argument("--resume", metavar="THREAD_ID", help="resume after clarification")
    group.add_argument("--show", metavar="THREAD_ID", help="print a session's state")
    group.add_argument(
        "--validate-models",
        action="store_true",
        help="check that every task has a usable provider endpoint",
    )
    group.add_argument(
        "--list-runs", action="store_true", help="list archived runs, newest first"
    )
    group.add_argument(
        "--show-run", metavar="RUN_ID", help="print one archived run"
    )
    parser.add_argument("--answer", help="answers to the clarifying questions")
    args = parser.parse_args(argv)

    _setup_logging(get_settings().log_level)

    if args.new:
        return asyncio.run(cmd_new(args.new))
    if args.resume:
        if not args.answer:
            parser.error("--resume requires --answer")
        return asyncio.run(cmd_resume(args.resume, args.answer))
    if args.show:
        return asyncio.run(cmd_show(args.show))
    if args.list_runs:
        return cmd_list_runs()
    if args.show_run:
        return cmd_show_run(args.show_run)
    return asyncio.run(cmd_validate())


if __name__ == "__main__":
    raise SystemExit(main())
