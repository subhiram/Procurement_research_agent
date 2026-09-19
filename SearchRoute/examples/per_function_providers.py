"""Different providers for different functions in one app.

The common real requirement: one code path wants Tavily with an Exa fallback,
another wants DuckDuckGo and nothing else.

The pattern is a single module-level client holding every provider the app will
ever pin, then pinning per call. One client means one quota ledger, one set of
circuit breakers, and one connection pool across the whole app — build two and a
provider that just failed in one function gets retried from scratch in the next.

Run it with whatever keys you have; the functions degrade independently.

    python examples/per_function_providers.py
"""

from __future__ import annotations

from searchroute import SearchRoute, discover_providers

# Everything any function might pin has to be on the client. Pinning a provider
# the client wasn't built with raises NoProviderAvailable.
#
# Note the extractors in this list. Pinning `providers=[...]` restricts the pool
# for *extraction* as well as search, so a client built only from search-only
# providers cannot serve depth="content" at all — every result comes back FAILED.
# Including a keyless extractor costs nothing and makes content depth work.
AVAILABLE = discover_providers()
WANTED = [
    p
    for p in ("tavily", "exa", "duckduckgo", "jina", "http_extract")
    if p in AVAILABLE
]

search = SearchRoute(providers=WANTED or ["duckduckgo"])


def research(query: str):
    """Tavily first, Exa only if Tavily fails or returns nothing.

    The list order is the fallback order — it is not reordered behind your back.
    """
    providers = [p for p in ("tavily", "exa") if p in search.providers]
    return search.search(query, providers=providers or None, depth="content", max_hydrate=3)


def quick_lookup(query: str):
    """Explicitly DuckDuckGo. No fallback, no spend."""
    return search.search(query, providers=["duckduckgo"], max_results=5)


def show(label: str, response) -> None:
    print(f"\n=== {label} ===")
    print(f"providers used : {response.providers_used}")
    print(f"depth          : {response.depth.name} (asked {response.requested_depth.name})")
    print(f"degraded       : {response.degraded}")
    print(f"cost           : {dict(response.cost.by_provider) or '{}'}")

    print("attempts       :")
    for attempt in response.attempts:
        status = "ok" if attempt.ok else f"FAILED ({attempt.error_kind})"
        print(f"    {attempt.provider:<12} {status:<22} {attempt.latency_ms:>6.0f}ms")

    print("results        :")
    for hit in response.results[:3]:
        chars = len(hit.content) if hit.content else 0
        print(f"    [{hit.content_status.value:<9}] {chars:>6} chars  {hit.title[:44]}")

    for note in response.notes:
        print(f"note           : {note}")


def main() -> int:
    print(f"configured: {search.providers}")

    with search:
        show("research('vector database indexing')", research("vector database indexing"))
        show("quick_lookup('python asyncio')", quick_lookup("python asyncio"))

        print("\n--- shared state across both functions ---")
        status = search.status()
        for name, info in status["breakers"].items():
            print(f"  breaker {name:<12} {info['state']} ({info['latency_ms']:.0f}ms avg)")
        for name, info in status["quota"].items():
            print(f"  quota   {name:<12} {info['remaining']} {info['unit']} left")
        if not status["quota"]:
            print("  quota   (no metered providers configured)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
