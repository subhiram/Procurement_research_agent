"""Crossref — DOI metadata across essentially every academic publisher.

Keyless and unmetered, ``ACADEMIC`` only. Best for resolving citations and
finding the canonical publication record for a work, which is exactly what a
research agent needs when it has a title and wants the real source.

Crossref runs a "polite pool" with better service for requests that identify
themselves and supply a contact address. Set ``SEARCHROUTE_CONTACT`` (or pass
``contact=``) and it is sent as both a ``mailto`` parameter and part of the
User-Agent, per their guidance.

API: https://api.crossref.org
"""

from __future__ import annotations

from typing import Any

from ..types import (
    Capability,
    Depth,
    SearchQuery,
    SearchResponse,
    SearchResult,
    Usage,
)
from .base import Provider


class CrossrefProvider(Provider):
    name = "crossref"
    capabilities = frozenset({Capability.ACADEMIC})
    native_depths = frozenset({Depth.LINKS})
    """Metadata only — no abstract on most records, so it honestly claims LINKS.
    The router will still hydrate it if the caller asks for content."""
    requires_key = False
    quota = None
    quality_hint = 0.75
    default_priority = 16
    min_interval = 0.1
    base_url = "https://api.crossref.org/works"

    def __init__(self, *args, contact: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.contact = contact

    def cost_of(self, query: SearchQuery) -> int:
        return 0

    def _headers(self) -> dict[str, str]:
        agent = "SearchRoute/0.1.0 (+https://github.com/subhiram/searchroute)"
        if self.contact:
            agent = f"{agent} mailto:{self.contact}"
        return {"User-Agent": agent, "Accept": "application/json"}

    async def search(self, query: SearchQuery) -> SearchResponse:
        params: dict[str, Any] = {
            "query": query.query,
            "rows": min(query.max_results, 100),
            "select": (
                "DOI,title,author,abstract,issued,container-title,URL,"
                "is-referenced-by-count,type,publisher"
            ),
        }
        if self.contact:
            params["mailto"] = self.contact  # the polite pool
        params.update(query.params_for(self.name))

        response = await self._request(
            "GET", self.base_url, params=params, headers=self._headers()
        )
        items = response.json().get("message", {}).get("items", []) or []

        results = []
        for rank, item in enumerate(items):
            url = item.get("URL") or (
                f"https://doi.org/{item['DOI']}" if item.get("DOI") else ""
            )
            if not url:
                continue
            results.append(
                SearchResult(
                    url=url,
                    title=_first(item.get("title")),
                    snippet=_snippet(item),
                    published_date=_issued(item),
                    provider=self.name,
                    rank=rank,
                    raw=item,
                )
            )

        return SearchResponse(
            query=query.query,
            results=results,
            depth=Depth.SNIPPETS if any(r.snippet for r in results) else Depth.LINKS,
            requested_depth=query.depth,
            providers_used=[self.name],
            cost=Usage(),
        )


def _first(value: Any) -> str:
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return str(value or "")


def _snippet(item: dict[str, Any]) -> str | None:
    """Byline + venue. Crossref abstracts are JATS-XML when present at all, so
    we use the metadata line rather than shipping half-rendered markup."""
    authors = item.get("author") or []
    names = [
        " ".join(x for x in (a.get("given"), a.get("family")) if x)
        for a in authors
        if isinstance(a, dict)
    ]
    byline = ", ".join(n for n in names[:3] if n)
    if len(names) > 3:
        byline += " et al."
    venue = _first(item.get("container-title"))
    cited = item.get("is-referenced-by-count")
    parts = [p for p in (byline, venue) if p]
    if cited:
        parts.append(f"cited by {cited}")
    return " — ".join(parts) or None


def _issued(item: dict[str, Any]):
    """Crossref dates are nested integer parts: {"date-parts": [[2024, 3, 15]]}."""
    from datetime import datetime, timezone

    parts = (item.get("issued") or {}).get("date-parts") or []
    if not parts or not parts[0]:
        return None
    first = [p for p in parts[0] if isinstance(p, int)]
    if not first:
        return None
    year = first[0]
    month = first[1] if len(first) > 1 else 1
    day = first[2] if len(first) > 2 else 1
    try:
        return datetime(year, month, day, tzinfo=timezone.utc)
    except ValueError:
        return None
