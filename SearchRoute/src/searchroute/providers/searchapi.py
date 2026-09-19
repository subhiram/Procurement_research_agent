"""SearchApi.io — another Google SERP proxy.

Free tier is a small one-time trial rather than a renewing allowance, so like
Serper it is treated as reserve capacity and sorted late.
API: https://www.searchapi.io/docs
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


class SearchApiProvider(Provider):
    name = "searchapi"
    capabilities = frozenset({Capability.SEARCH, Capability.NEWS, Capability.ACADEMIC})
    native_depths = frozenset({Depth.LINKS, Depth.SNIPPETS})
    requires_key = True
    quota = QuotaPolicy(limit=100, unit=Unit.REQUESTS, period=Period.ONE_TIME)
    quality_hint = 0.72
    default_priority = 80
    base_url = "https://www.searchapi.io/api/v1/search"

    _ENGINES = {
        Capability.SEARCH: "google",
        Capability.NEWS: "google_news",
        Capability.ACADEMIC: "google_scholar",
    }

    async def search(self, query: SearchQuery) -> SearchResponse:
        params: dict[str, object] = {
            "api_key": self.api_key,
            "engine": self._ENGINES.get(query.capability, "google"),
            "q": query.query,
            "num": min(query.max_results, 100),
        }
        if query.lang:
            params["hl"] = query.lang
        if query.region:
            params["gl"] = query.region
        params.update(query.params_for(self.name))

        response = await self._request("GET", self.base_url, params=params)
        body = response.json()
        rows = body.get("organic_results") or body.get("news_results") or []

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
