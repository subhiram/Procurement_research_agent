"""Live smoke test: one real query per configured provider.

Prints an ok/latency/results matrix plus remaining quota, so you can see at a
glance which providers actually work with the keys you have. Excluded from CI —
it spends real credits.

    python scripts/smoke.py
    python scripts/smoke.py --depth content     # also exercise extraction
"""

from __future__ import annotations

import argparse
import asyncio
import time

from searchroute import AsyncSearchRoute, Capability, Depth
from searchroute.config import discover_providers
from searchroute.providers import registry
from searchroute.types import SearchQuery

QUERY = "what is a vector database"
EXTRACT_URL = "https://en.wikipedia.org/wiki/Vector_database"


async def probe_search(
    provider, depth: Depth, capability: Capability = Capability.SEARCH
) -> dict:
    query = SearchQuery(query=QUERY, max_results=3, depth=depth, capability=capability)
    started = time.perf_counter()
    try:
        response = await provider.search(query)
    except Exception as exc:
        return {"ok": False, "ms": (time.perf_counter() - started) * 1000, "note": str(exc)[:60]}
    elapsed = (time.perf_counter() - started) * 1000
    with_content = sum(1 for r in response.results if r.content)
    return {
        "ok": True,
        "ms": elapsed,
        "n": len(response.results),
        "note": f"{with_content} with content" if depth >= Depth.CONTENT else "",
    }


async def probe_extract(provider) -> dict:
    started = time.perf_counter()
    try:
        docs = await provider.extract([EXTRACT_URL])
    except Exception as exc:
        return {"ok": False, "ms": (time.perf_counter() - started) * 1000, "note": str(exc)[:60]}
    elapsed = (time.perf_counter() - started) * 1000
    doc = docs[0]
    return {
        "ok": doc.ok,
        "ms": elapsed,
        "n": len(doc.content or ""),
        "note": doc.error[:60] if doc.error else "chars",
    }


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", default="snippets", choices=[d.name.lower() for d in Depth])
    args = parser.parse_args()
    depth = Depth[args.depth.upper()]

    names = discover_providers()
    if not names:
        print("No providers configured. Set an API key, e.g. EXA_API_KEY.")
        return 1

    print(f"query: {QUERY!r}   depth: {depth.name}")
    print(f"configured: {', '.join(names)}\n")

    sr = AsyncSearchRoute(quota_store="memory")
    print(f"{'provider':<14} {'op':<8} {'ok':<4} {'ms':>7} {'n':>6}  note")
    print("-" * 78)

    for name in names:
        provider = sr.engine.providers.get(name)
        if provider is None:
            continue
        if provider.can_serve(Capability.SEARCH):
            r = await probe_search(provider, depth)
            print(
                f"{name:<14} {'search':<8} {'yes' if r['ok'] else 'NO':<4} "
                f"{r['ms']:>7.0f} {r.get('n', 0):>6}  {r.get('note', '')}"
            )
        # The specialized sources answer only their own capability, so
        # probing them with SEARCH would report a false negative.
        for capability, label in (
            (Capability.ACADEMIC, "academic"),
            (Capability.REFERENCE, "reference"),
            (Capability.DISCUSSION, "discuss"),
        ):
            if provider.can_serve(capability):
                r = await probe_search(provider, depth, capability)
                print(
                    f"{name:<14} {label:<8} {'yes' if r['ok'] else 'NO':<4} "
                    f"{r['ms']:>7.0f} {r.get('n', 0):>6}  {r.get('note', '')}"
                )
        if provider.can_serve(Capability.EXTRACT):
            r = await probe_extract(provider)
            print(
                f"{name:<14} {'extract':<8} {'yes' if r['ok'] else 'NO':<4} "
                f"{r['ms']:>7.0f} {r.get('n', 0):>6}  {r.get('note', '')}"
            )

    print("\n--- quota ---")
    snapshot = sr.status()["quota"]
    if not snapshot:
        print("  (no metered providers configured)")
    for name, info in snapshot.items():
        print(f"  {name:<14} {info['remaining']} {info['unit']} left, resets {info['resets_at']}")

    # Providers that exist but aren't usable, and why.
    missing = sorted(set(registry.available()) - set(names))
    if missing:
        print(f"\nnot configured: {', '.join(missing)}")

    await sr.aclose()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
