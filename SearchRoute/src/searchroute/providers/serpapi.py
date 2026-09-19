"""SerpAPI — verbatim Google SERP, including the news and scholar verticals.

Free tier: ~100 searches/month, renews, no card. Small, so it is worth saving
for queries that genuinely need Google's ranking or a vertical the others can't
reach — the ``quota_aware`` strategy handles that automatically.
API: https://serpapi.com/search-api
"""

from __future__ import annotations

import httpx

from ..errors import ProviderError, QuotaExceeded
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


class SerpAPIProvider(Provider):
    name = "serpapi"
    capabilities = frozenset({Capability.SEARCH, Capability.NEWS, Capability.ACADEMIC})
    native_depths = frozenset({Depth.LINKS, Depth.SNIPPETS})
    requires_key = True
    quota = QuotaPolicy(limit=100, unit=Unit.REQUESTS, period=Period.MONTHLY)
    quality_hint = 0.8
    default_priority = 60
    base_url = "https://serpapi.com/search.json"

    #: Which SerpAPI engine serves each capability.
    _ENGINES = {
        Capability.SEARCH: "google",
        Capability.NEWS: "google_news",
        Capability.ACADEMIC: "google_scholar",
    }

    def raise_for_status(self, response: httpx.Response) -> None:
        """SerpAPI reports some failures as 200 with an ``error`` field."""
        if response.is_success:
            try:
                error = response.json().get("error")
            except Exception:
                return
            if error:
                text = str(error)
                if "run out" in text.lower() or "exceeded" in text.lower():
                    raise QuotaExceeded(self.name, text, status=200)
                raise ProviderError(self.name, text, status=200)
            return
        super().raise_for_status(response)

    async def search(self, query: SearchQuery) -> SearchResponse:
        engine = self._ENGINES.get(query.capability, "google")
        params: dict[str, object] = {
            "api_key": self.api_key,
            "engine": engine,
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

        # Each engine names its result list differently.
        rows = (
            body.get("organic_results")
            or body.get("news_results")
            or body.get("scholar_results")
            or []
        )

        results = [
            SearchResult(
                url=item.get("link", ""),
                title=item.get("title") or "",
                snippet=item.get("snippet") or item.get("publication_info", {}).get("summary"),
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
