"""End-to-end demo: the research shape.

Runs a real query at CONTENT depth and prints the per-result content status
matrix plus what the query actually spent. This is the acceptance check for the
depth axis — run it with real API keys set.

    export EXA_API_KEY=...        # or TAVILY_API_KEY, FIRECRAWL_API_KEY, ...
    python examples/research_agent.py "your question here"

With no keys at all it still works, degraded, via keyless DuckDuckGo plus local
extraction — install those with: pip install 'searchroute[all]'
"""

from __future__ import annotations

import sys

from searchroute import ContentStatus, SearchRoute, discover_providers

QUERY = "retrieval augmented generation benchmarks 2026"


def main() -> int:
    query = " ".join(sys.argv[1:]) or QUERY

    found = discover_providers()
    print(f"providers available: {', '.join(found) or 'none'}\n")
    if not found:
        print("No providers configured. Set an API key or install searchroute[ddg].")
        return 1

    with SearchRoute(profile="research", max_results=8, max_hydrate=5) as sr:
        response = sr.search(query)

        print(f"query      : {response.query}")
        print(f"providers  : {', '.join(response.providers_used)}")
        print(f"depth      : {response.depth.name} (asked for {response.requested_depth.name})")
        print(f"degraded   : {response.degraded}")
        print(f"cost       : {dict(response.cost.by_provider)}\n")

        print(f"{'status':<10} {'chars':>7}  {'via':<12} url")
        print("-" * 92)
        for hit in response.results:
            chars = len(hit.content) if hit.content else 0
            via = hit.content_provider or "-"
            print(f"{hit.content_status.value:<10} {chars:>7}  {via:<12} {hit.url[:55]}")

        hydrated = sum(1 for r in response if r.content_status is ContentStatus.HYDRATED)
        native = sum(1 for r in response if r.content_status is ContentStatus.NATIVE)
        failed = sum(1 for r in response if r.content_status is ContentStatus.FAILED)
        print(
            f"\n{native} native, {hydrated} hydrated, {failed} failed, "
            f"{len(response.results)} total"
        )

        print("\n--- quota after this query ---")
        for name, info in sr.status()["quota"].items():
            print(
                f"  {name:<12} {info['remaining']} {info['unit']} left, "
                f"resets {info['resets_at']}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
