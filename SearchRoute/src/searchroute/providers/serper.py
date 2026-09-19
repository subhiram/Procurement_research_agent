"""Serper.dev — fast, cheap Google SERP.

Free tier: 2,500 queries, no card — but **one-time**, not renewing. That makes
it the reserve tank: the ``quota_aware`` strategy deliberately sorts one-time
grants last, because a credit spent here never comes back, while a monthly tier
refills whether you used it or not.
API: https://serper.dev
"""

from __future__ import annotations

from ..quota import Period, QuotaPolicy, Unit
from ..types import (
    Capability,
    Depth,
    SearchQuery,
    SearchResponse,
    SearchResult,
    Usage,
)
from ._util import parse_date
from .base import Provider


class SerperProvider(Provider):
    name = "serper"
    capabilities = frozenset({Capability.SEARCH, Capability.NEWS, Capability.ACADEMIC})
    native_depths = frozenset({Depth.LINKS, Depth.SNIPPETS})
    requires_key = True
    quota = QuotaPolicy(limit=2500, unit=Unit.REQUESTS, period=Period.ONE_TIME)
    quality_hint = 0.78
    default_priority = 70
    base_url = "https://google.serper.dev"

    _ENDPOINTS = {
        Capability.SEARCH: "search",
        Capability.NEWS: "news",
        Capability.ACADEMIC: "scholar",
    }

    async def search(self, query: SearchQuery) -> SearchResponse:
        endpoint = self._ENDPOINTS.get(query.capability, "search")
        payload: dict[str, object] = {
            "q": query.query,
            "num": min(query.max_results, 100),
        }
        if query.lang:
            payload["hl"] = query.lang
        if query.region:
            payload["gl"] = query.region
        payload.update(query.params_for(self.name))

        response = await self._request(
            "POST",
            f"{self.base_url}/{endpoint}",
            headers={"X-API-KEY": self.api_key or "", "Content-Type": "application/json"},
            json=payload,
        )
        body = response.json()
        rows = body.get("organic") or body.get("news") or body.get("organic_results") or []

        results = [
            SearchResult(
                url=item.get("link", ""),
                title=item.get("title") or "",
                snippet=item.get("snippet"),
                published_date=parse_date(item.get("date")),
                provider=self.name,
                rank=rank,
                raw=item,
            )
            for rank, item in enumerate(rows[: query.max_results])
        ]

        return SearchResponse(
            query=query.query,
            results=[r for r in results if r.url],
            depth=Depth.SNIPPETS,
            requested_depth=query.depth,
            providers_used=[self.name],
            cost=Usage({self.name: 1}),
        )
