"""Brave Search — implemented, but **disabled by default**.

Brave retired its free API tier: signup now requires a card, and queries past a
small starting credit are billed. That fails this package's "no card, renewing
free tier" bar, so it is excluded from auto-discovery and never appears in the
default chain.

It is here because it is a genuinely good index and some users already pay for
it. Opting in is explicit:

    SearchRoute(providers=["brave", "duckduckgo"])   # names it directly

Setting ``BRAVE_API_KEY`` alone is deliberately not enough — a provider that can
charge money should not switch itself on because an environment variable exists.
API: https://api-dashboard.search.brave.com
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


class BraveProvider(Provider):
    name = "brave"
    capabilities = frozenset({Capability.SEARCH, Capability.NEWS})
    native_depths = frozenset({Depth.LINKS, Depth.SNIPPETS})
    requires_key = True
    opt_in_only = True
    """Never auto-discovered: this provider can bill the user."""
    quota = QuotaPolicy(limit=2000, unit=Unit.REQUESTS, period=Period.MONTHLY)
    """Nominal free allowance on the metered plan; the account is billed beyond
    its starting credit, so treat this as a guard rail, not a free tier."""
    quality_hint = 0.8
    default_priority = 90
    base_url = "https://api.search.brave.com/res/v1"

    async def search(self, query: SearchQuery) -> SearchResponse:
        news = query.capability is Capability.NEWS
        endpoint = "news/search" if news else "web/search"

        params: dict[str, object] = {
            "q": query.query,
            "count": min(query.max_results, 20),
        }
        if query.lang:
            params["search_lang"] = query.lang
        if query.region:
            params["country"] = query.region
        params.update(query.params_for(self.name))

        response = await self._request(
            "GET",
            f"{self.base_url}/{endpoint}",
            headers={
                "X-Subscription-Token": self.api_key or "",
                "Accept": "application/json",
            },
            params=params,
        )
        body = response.json()
        rows = (body.get("news") if news else body.get("web", {})).get("results", []) or []

        results = [
            SearchResult(
                url=item.get("url", ""),
                title=item.get("title") or "",
                snippet=item.get("description"),
                published_date=parse_date(item.get("age") or item.get("page_age")),
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
