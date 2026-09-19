"""Force the degradation paths against real providers, end to end.

The unit tests prove the *mapping* - that a 429 becomes a RateLimited, that a
keyless provider costs nothing. Only a live run proves the *wiring*: that the
ladder actually reaches the endpoint it is supposed to, that a container without
Chromium still produces leads, that the run really does complete with no API
keys at all.

That last one is the point of this script. "It works with no keys" is claimed
throughout the design and, until now, had never been executed.

Each scenario runs the graph with the environment deliberately broken, then
asserts on the **trace** rather than on stdout - the trace is what says which
provider actually served each call, and a run can print a perfectly good answer
while having quietly reached it the wrong way.

    uv run python scripts/fallback_drill.py                # every scenario
    uv run python scripts/fallback_drill.py --only no-keys
    uv run python scripts/fallback_drill.py --list

Costs real search credits and real tokens. `--only` exists so a single path can
be re-checked without paying for all of them.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv  # noqa: E402

#: Short and unambiguous, so a thin result means the fallback path is thin
#: rather than the material being obscure.
REQUEST = "Inconel 625 round bar, 25mm diameter, 40 KG"

HOSTED_LLM_KEYS = (
    "GROQ_API_KEY",
    "MISTRAL_API_KEY",
    "NVIDIA_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "OPENROUTER_API_KEY",
)
SEARCH_KEYS = ("TAVILY_API_KEY", "FIRECRAWL_API_KEY", "EXA_API_KEY", "BRAVE_API_KEY")

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"


@dataclass
class Scenario:
    name: str
    why: str
    env: dict[str, str | None]
    #: Given the archived record, return a list of failure strings. Empty passes.
    check: object
    notes: list[str] = field(default_factory=list)


def _served_by(record: dict) -> set[str]:
    return set(record["totals"].get("llm_calls_by_provider", {}))


def _search_providers(record: dict) -> set[str]:
    return set(record["totals"].get("search_calls_by_provider", {}))


def warnings_for(record: dict) -> list[str]:
    """Things worth saying that are not failures."""
    notes = []
    if not record.get("vendor_leads"):
        notes.append(
            "zero vendor leads - the run wired up correctly but its search "
            "providers returned nothing"
        )
    elif record.get("truncated"):
        notes.append("results are truncated (budget spent)")
    return notes


def _completed(record: dict) -> list[str]:
    """Every scenario shares this floor: the run finished and found something."""
    problems = []
    if not record.get("vendor_leads"):
        problems.append("no vendor leads at all")
    if "vendor_summary" not in record["totals"].get("nodes_run", []):
        problems.append("run did not reach vendor_summary")
    return problems


def check_no_keys(record: dict) -> list[str]:
    """Assert the *routing*, not the yield.

    Deliberately does not require vendor leads. With no keys there is exactly
    one web-search provider - DuckDuckGo - and it is IP rate-limited and prone
    to transient failure; a run of this scenario genuinely returned zero
    candidates because `ddgs` rotated onto a backup backend that then failed
    DNS. That is a property of the free tier, not a defect in the fallback.

    What must hold is that nothing reached for a credential it does not have
    and nothing was charged. If those hold and the yield is zero, the path is
    correct and the free provider was simply unavailable - reported below as a
    warning rather than a failure, because failing here would mean the drill
    goes red for someone else's outage.
    """
    problems = []
    if "vendor_summary" not in record["totals"].get("nodes_run", []):
        problems.append("run did not reach vendor_summary")
    served = _served_by(record)
    if served - {"ollama"}:
        problems.append(f"a hosted provider served a call with no keys set: {served}")
    metered = _search_providers(record) - {
        "arxiv", "pubmed", "crossref", "wikipedia", "duckduckgo", "hackernews",
        "jina", "http_extract", "searxng",
    }
    if metered:
        problems.append(f"a metered search provider was used with no keys: {metered}")
    if record["totals"].get("searches_charged", 0) != 0:
        problems.append("keyless searches were charged against the run budget")
    if not served:
        problems.append("no LLM call was served at all - is Ollama running?")
    return problems


def check_bad_groq_key(record: dict) -> list[str]:
    problems = _completed(record)
    if "groq" in _served_by(record):
        problems.append("groq served a call despite an invalid key")
    return problems


def check_no_ollama(record: dict) -> list[str]:
    problems = _completed(record)
    if "ollama" in _served_by(record):
        problems.append("ollama served a call with the daemon unreachable")
    return problems


def check_tiny_budget(record: dict) -> list[str]:
    problems = _completed(record)
    if not record.get("truncated"):
        problems.append("a 2-credit run was not marked truncated")
    # Grounding must survive a starved budget: a designation with no evidence
    # behind it is exactly what the verification step exists to remove.
    research = record.get("research") or {}
    if research.get("designations") and not (research.get("notes") or ""):
        problems.append("designations kept with no note about the evidence")
    return problems


def check_no_crawler(record: dict) -> list[str]:
    problems = _completed(record)
    reachable = sum(
        1 for lead in record["vendor_leads"] if lead.get("email") or lead.get("phone")
    )
    if reachable == 0:
        problems.append("no contact details at all without the crawler")
    return problems


SCENARIOS = [
    Scenario(
        "no-keys",
        "The headline claim: runs with no API keys at all, on Ollama plus the "
        "keyless search providers.",
        {k: None for k in HOSTED_LLM_KEYS + SEARCH_KEYS},
        check_no_keys,
        ["needs a local Ollama daemon with gemma4:e4b pulled"],
    ),
    Scenario(
        "bad-groq-key",
        "A 401 on the provider that leads every tier must park it and let the "
        "rest of the ladder serve, not fail the run.",
        {"GROQ_API_KEY": "sk-invalid-on-purpose"},
        check_bad_groq_key,
    ),
    Scenario(
        "no-ollama",
        "The local backstop absent. Hosted providers carry the run and nothing "
        "waits on a daemon that is not there.",
        {"OLLAMA_BASE_URL": "http://127.0.0.1:59999"},
        check_no_ollama,
    ),
    Scenario(
        "tiny-budget",
        "Two credits. The run completes, says it is truncated, and the results "
        "it did get are still grounded.",
        {"SEARCH_CREDITS_PER_SESSION": "2"},
        check_tiny_budget,
    ),
    Scenario(
        "no-crawler",
        "No Chromium - what a stripped container looks like. Metered extraction "
        "takes over and contacts are still found.",
        {"ENABLE_CRAWL4AI": "false"},
        check_no_crawler,
    ),
]


def _reset_caches() -> None:
    """Make the routers re-read the environment.

    Settings, the model registry and the search client are all cached for the
    process, which is right in production and wrong here: a scenario that edits
    the environment after they are built would test the previous scenario's
    configuration.
    """
    import llm_router

    import procurement_agent.llm.models as models
    import procurement_agent.search.client as search_client
    from procurement_agent.config import get_settings

    get_settings.cache_clear()
    llm_router.reset_default_registry()
    llm_router.reset_state()
    from llm_router.providers import reset_providers

    reset_providers()
    models._configured = False
    search_client._client = None


async def run_scenario(scenario: Scenario) -> tuple[bool, list[str], str, dict | None]:
    """Run the graph under this scenario. Returns (ok, problems, run_id, record)."""
    original = {k: os.environ.get(k) for k in scenario.env}
    for key, value in scenario.env.items():
        # Blanked, never deleted. `import llm_router` calls `load_dotenv()`,
        # which repopulates anything *absent* from os.environ straight out of
        # .env - so a deleted key comes back and the scenario silently tests
        # nothing. It skips keys that are present, and an empty value reads as
        # "no credential" to both routers, so blanking holds.
        #
        # The first run of this drill failed exactly here: with every key
        # "unset", Groq, Mistral, Tavily and Exa all still served the run.
        os.environ[key] = value if value is not None else ""

    try:
        _reset_caches()

        from procurement_agent.archive import load_run
        from procurement_agent.graph.build import initial_state, open_graph, run_config

        thread_id = str(uuid.uuid4())
        async with open_graph() as graph:
            config = run_config(thread_id, graph=graph)
            async for _ in graph.astream(
                initial_state(REQUEST), config, stream_mode="updates"
            ):
                pass
            state = await graph.aget_state(config)

            # A scenario that interrupts is answered so the run completes; the
            # point is the degradation path, not the conversation.
            if state.interrupts:
                from langgraph.types import Command

                async for _ in graph.astream(
                    Command(resume="Annealed, standard mill lengths."),
                    config,
                    stream_mode="updates",
                ):
                    pass

        record = load_run(thread_id)
        if record is None:
            return False, ["the run produced no archive record"], "", None
        problems = scenario.check(record)
        return (not problems), problems, record["run_id"], record
    except Exception as exc:  # noqa: BLE001 - a crash is the finding
        return False, [f"raised {type(exc).__name__}: {exc}"], "", None
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        # Leave the caches cleared too, so the next scenario re-reads the
        # restored environment rather than the one it just finished breaking.
        _reset_caches()


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fallback_drill")
    parser.add_argument("--only", help="run one scenario by name")
    parser.add_argument("--list", action="store_true", help="list the scenarios")
    args = parser.parse_args(argv)

    if args.list:
        for scenario in SCENARIOS:
            print(f"{scenario.name:<16} {scenario.why}")
        return 0

    load_dotenv()
    chosen = [s for s in SCENARIOS if not args.only or s.name == args.only]
    if not chosen:
        print(f"no scenario named {args.only!r}", file=sys.stderr)
        return 2

    failures = 0
    for scenario in chosen:
        print(f"\n{'=' * 72}\n{scenario.name}\n{DIM}{scenario.why}{RESET}")
        for note in scenario.notes:
            print(f"{DIM}  note: {note}{RESET}")

        ok, problems, run_id, record = await run_scenario(scenario)
        if ok:
            print(f"  {GREEN}PASS{RESET}  archived as {run_id}")
            for note in warnings_for(record or {}):
                print(f"    {YELLOW}!{RESET} {note}")
        else:
            failures += 1
            print(f"  {RED}FAIL{RESET}")
            for problem in problems:
                print(f"    {RED}-{RESET} {problem}")

    print(f"\n{'=' * 72}")
    if failures:
        print(f"{RED}{failures} of {len(chosen)} scenario(s) failed.{RESET}")
    else:
        print(f"{GREEN}all {len(chosen)} scenario(s) passed.{RESET}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
