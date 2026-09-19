"""Fan-out across the free scholarly sources.

Shows the payoff of putting the specialized sources on the capability axis:
arXiv, PubMed and Crossref are all unmetered, so fanning out across three of them
and fusing the rankings costs **nothing**.

No API keys needed. Run it anywhere:

    python examples/academic_search.py
    python examples/academic_search.py "your research question"
"""

from __future__ import annotations

import asyncio
import sys

from searchroute import AsyncSearchRoute, Capability

QUESTION = "retrieval augmented generation chunking strategies"


def _authors(raw: dict) -> list[str]:
    """Pull author names out of whichever shape the source used."""
    if raw.get("authors_flat"):  # PubMed: pre-flattened alongside the raw dicts
        return [str(a) for a in raw["authors_flat"]]

    authors = raw.get("author") or raw.get("authors") or []
    names = []
    for entry in authors:
        if isinstance(entry, dict):  # Crossref: {"given": ..., "family": ...}
            name = " ".join(x for x in (entry.get("given"), entry.get("family")) if x)
            names.append(name or entry.get("name", ""))
        else:  # arXiv: plain strings
            names.append(str(entry))
    return [n for n in names if n]


async def main() -> int:
    query = " ".join(sys.argv[1:]) or QUESTION

    async with AsyncSearchRoute(
        strategy="fanout",
        # Crossref and PubMed give better service to identified callers, and
        # their terms ask for it. Costs nothing.
        contact="you@example.com",
        quota_store="memory",
    ) as sr:
        academic = [
            p.name
            for p in sr.engine.providers.values()
            if p.can_serve(Capability.ACADEMIC)
        ]
        print(f"query   : {query}")
        print(f"sources : {', '.join(academic)}\n")

        response = await sr.search(
            query, capability=Capability.ACADEMIC, max_results=8, n=3
        )

        print(f"{'rank':<5} {'source':<10} title")
        print("-" * 88)
        for hit in response.results:
            found_by = hit.raw.get("_searchroute", {}).get("found_by")
            marker = "+".join(found_by) if found_by else hit.provider
            print(f"{hit.rank:<5} {marker[:10]:<10} {hit.title[:64]}")

            # Each source shapes authors differently — arXiv gives plain strings,
            # PubMed gives dicts (and a flattened list alongside), Crossref gives
            # given/family pairs. `raw` keeps all of it; normalizing is the
            # caller's job, which is the point of `raw` existing.
            authors = _authors(hit.raw)
            if authors:
                shown = ", ".join(authors[:3])
                print(f"{'':<16} {shown}{' et al.' if len(authors) > 3 else ''}")
            print(f"{'':<16} {hit.url}")

        print(f"\nproviders used : {', '.join(response.providers_used)}")
        print(f"total cost     : {response.cost.total} credits")
        print("\nThree scholarly sources, fused by reciprocal rank fusion, for free.")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
