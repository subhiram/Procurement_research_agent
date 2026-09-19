#!/usr/bin/env python3
"""Live end-to-end check against whichever providers have an API key set.

The unit tests use fakes, so this is the script that proves the real thing
works: every configured endpoint is called once directly, then the ladder and
the LangChain wrapper are exercised, then the ledger is printed so you can see
what the run actually cost.

    export GROQ_API_KEY=...  NVIDIA_API_KEY=...  OPENROUTER_API_KEY=...
    python scripts/smoke_test.py
    python scripts/smoke_test.py --provider groq     # just one
    python scripts/smoke_test.py --skip-endpoints    # ladder + wrapper only

Each endpoint costs one small request, so a full run spends a handful of your
free tier. Endpoints with no key are skipped, not failed.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_router import (  # noqa: E402
    AllCandidatesExhausted,
    LLMRouter,
    ProviderError,
    RateLimited,
    default_ledger,
    load_registry,
    route,
)

PROMPT = "Reply with exactly one word: ok"
GREEN, RED, YELLOW, DIM, BOLD, RESET = (
    "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[1m", "\033[0m"
)


def heading(text: str) -> None:
    print(f"\n{BOLD}{text}{RESET}\n" + "-" * len(text))


def check_endpoints(registry, only: str | None) -> tuple[int, int]:
    """Call every configured endpoint directly, bypassing the ladder."""
    heading("1. every configured endpoint, called directly")
    passed = failed = 0

    for model in registry.models():
        for endpoint in registry.get_endpoints(model):
            if only and endpoint.provider != only:
                continue
            label = f"{endpoint.provider}/{endpoint.model_id}"
            if not endpoint.has_credentials:
                print(f"  {DIM}skip    {label} - no API key{RESET}")
                continue

            started = time.monotonic()
            try:
                response = route(
                    PROMPT, provider=endpoint.provider, model=model,
                    max_tokens=16, return_route=True,
                )
            except RateLimited as exc:
                print(f"  {YELLOW}limited {label} - {exc}{RESET}")
                continue
            except AllCandidatesExhausted as exc:
                print(f"  {YELLOW}busy    {label} - {exc.args[0].splitlines()[0]}{RESET}")
                continue
            except ProviderError as exc:
                failed += 1
                print(f"  {RED}FAIL    {label} - {exc}{RESET}")
                continue

            passed += 1
            elapsed = time.monotonic() - started
            text = str(response.message.content).strip().replace("\n", " ")[:40]
            print(f"  {GREEN}ok{RESET}      {label}  {elapsed:5.2f}s  "
                  f"{response.tokens or '?'} tok  {DIM}{text!r}{RESET}")
    return passed, failed


def check_ladder() -> None:
    heading("2. the ladder, tiers and strategies")
    for description, kwargs in [
        ("free_first, no preference", {}),
        ("tier S floor", {"tier": "S"}),
        ("tier B floor", {"tier": "B"}),
        ("sticky, first call", {"strategy": "sticky", "session_id": "smoke"}),
        ("sticky, second call", {"strategy": "sticky", "session_id": "smoke"}),
    ]:
        try:
            result = route(PROMPT, max_tokens=16, return_route=True, **kwargs)
        except AllCandidatesExhausted as exc:
            print(f"  {YELLOW}busy{RESET}    {description}: {exc.args[0].splitlines()[0]}")
            continue
        flag = f" {YELLOW}(TIER DOWNGRADE){RESET}" if result.tier_downgraded else ""
        print(f"  {GREEN}ok{RESET}      {description}: {result.endpoint.provider}/"
              f"{result.endpoint.model_id} [tier {result.endpoint.tier}, "
              f"step {result.candidate.step}]{flag}")


def check_wrapper() -> None:
    heading("3. the LangChain wrapper")
    router = LLMRouter(strategy="free_first", max_wait=5.0)
    try:
        response = router.invoke(PROMPT)
        routing = response.response_metadata["llm_router"]
        print(f"  {GREEN}ok{RESET}      invoke  -> {routing['provider']}/{routing['model_id']}")
    except AllCandidatesExhausted as exc:
        print(f"  {YELLOW}busy{RESET}    invoke: {exc.args[0].splitlines()[0]}")
        return

    try:
        chunks = list(router.stream(PROMPT))
        text = "".join(str(c.content) for c in chunks).strip()[:40]
        print(f"  {GREEN}ok{RESET}      stream  -> {len(chunks)} chunks {DIM}{text!r}{RESET}")
    except Exception as exc:
        print(f"  {RED}FAIL{RESET}    stream: {type(exc).__name__}: {exc}")

    try:
        from langchain_core.tools import tool

        @tool
        def add(a: int, b: int) -> int:
            """Add two integers."""
            return a + b

        result = router.bind_tools([add]).invoke("What is 17 + 25? Use the add tool.")
        calls = getattr(result, "tool_calls", [])
        status = f"{len(calls)} tool call(s): {calls}" if calls else "no tool call made"
        print(f"  {GREEN}ok{RESET}      tools   -> {status}")
    except Exception as exc:
        print(f"  {RED}FAIL{RESET}    tools: {type(exc).__name__}: {exc}")


def show_ledger() -> None:
    heading("4. what this run spent")
    snapshot = default_ledger().snapshot()
    if not snapshot:
        print(f"  {DIM}nothing recorded{RESET}")
        return
    for key, bucket in sorted(snapshot.items()):
        counters = ", ".join(
            f"{name} {values['used']}/{values['limit']}"
            for name, values in sorted(bucket["counters"].items())
        )
        cooling = bucket.get("cooling_down_for")
        suffix = f"  {YELLOW}cooling down {cooling:.0f}s{RESET}" if cooling else ""
        print(f"  {key}: {counters}{suffix}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", help="only test this provider")
    parser.add_argument("--skip-endpoints", action="store_true",
                        help="skip the per-endpoint pass (saves quota)")
    parser.add_argument("--verbose", action="store_true", help="show router logs")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    registry = load_registry()
    live = [p for p in registry.providers() if registry.is_usable(p)]
    if not live:
        print(f"{RED}No provider has an API key set.{RESET} Set at least one of "
              "GROQ_API_KEY, MISTRAL_API_KEY, GOOGLE_API_KEY, NVIDIA_API_KEY, "
              "OPENROUTER_API_KEY.")
        return 2
    print(f"{BOLD}Live providers:{RESET} {', '.join(live)}")

    failed = 0
    if not args.skip_endpoints:
        _, failed = check_endpoints(registry, args.provider)
    check_ladder()
    check_wrapper()
    show_ledger()

    print()
    if failed:
        print(f"{RED}{failed} endpoint(s) failed outright.{RESET} "
              f"Run `python scripts/verify_models.py` to check the model ids.")
        return 1
    print(f"{GREEN}Smoke test finished.{RESET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
