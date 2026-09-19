"""Firecrawl — search plus the strongest extraction of the group.

Free tier: 1,000 credits/month, renews, no card. Search costs 2 credits per 10
results, and scraping each result costs 1 more — which is why ``cost_of`` has to
account for depth rather than returning a flat 1.

Its real strength is ``extract``: it renders JavaScript, so it gets pages that
the keyless local extractor cannot. That earns it the top of the extract chain.
API: https://docs.firecrawl.dev
"""

from __future__ import annotations

from typing import Any

from ..errors import ProviderError
from ..quota import Period, QuotaPolicy, Unit
from ..types import (
    Capability,
    ContentStatus,
    Depth,
    Document,
    SearchQuery,
    SearchResponse,
    SearchResult,
    Usage,
)
from ._util import parse_date
from .base import Provider


class FirecrawlProvider(Provider):
    name = "firecrawl"
    capabilities = frozenset({Capability.SEARCH, Capability.EXTRACT, Capability.CRAWL})
    # No SUMMARY: Firecrawl returns page content, it does not summarize results.
    native_depths = frozenset({Depth.LINKS, Depth.SNIPPETS, Depth.CONTENT})
    requires_key = True
    quota = QuotaPolicy(limit=1000, unit=Unit.CREDITS, period=Period.MONTHLY)
    quality_hint = 0.75
    extract_quality = 0.95
    """The best extractor here: it renders JavaScript, unlike the local fallback."""
    default_priority = 30
    base_url = "https://api.firecrawl.dev/v2"
    timeout = 45.0
    """Scraping renders pages, so it is legitimately slower than a search call."""

    def cost_of(self, query: SearchQuery) -> int:
        # 2 credits per 10 search results...
        cost = 2 * max(1, -(-query.max_results // 10))
        if query.depth >= Depth.CONTENT:
            # ...plus 1 credit for every page actually scraped.
            cost += query.max_results
        return cost

    def extract_cost_of(self, urls: list[str]) -> int:
        return len(urls)

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def search(self, query: SearchQuery) -> SearchResponse:
        wants_content = query.depth >= Depth.CONTENT
        payload: dict[str, Any] = {"query": query.query, "limit": query.max_results}
        if wants_content:
            payload["scrapeOptions"] = {"formats": ["markdown"], "onlyMainContent": True}
        if query.include_domains:
            payload["includeDomains"] = query.include_domains
        if query.exclude_domains:
            payload["excludeDomains"] = query.exclude_domains
        if query.lang:
            payload["lang"] = query.lang
        if query.region:
            payload["country"] = query.region
        payload.update(query.params_for(self.name))

        response = await self._request(
            "POST", f"{self.base_url}/search", headers=self._headers(), json=payload
        )
        body = response.json()
        if not body.get("success", True):
            raise ProviderError(self.name, f"search failed: {body.get('error')}")

        # v2 groups results by source; web is what we want for a plain search.
        data = body.get("data") or {}
        rows = data.get("web", data) if isinstance(data, dict) else data
        if not isinstance(rows, list):
            rows = []

        results = []
        for rank, item in enumerate(rows):
            content = item.get("markdown")
            results.append(
                SearchResult(
                    url=item.get("url", ""),
                    title=item.get("title") or "",
                    snippet=item.get("description"),
                    content=content if wants_content else None,
                    content_status=(
                        (ContentStatus.NATIVE if content else ContentStatus.FAILED)
                        if wants_content
                        else ContentStatus.NOT_REQUESTED
                    ),
                    published_date=parse_date(item.get("date")),
                    provider=self.name,
                    content_provider=self.name if (wants_content and content) else None,
                    rank=rank,
                    raw=item,
                )
            )

        return SearchResponse(
            query=query.query,
            results=results,
            depth=query.depth if wants_content else Depth.SNIPPETS,
            requested_depth=query.depth,
            providers_used=[self.name],
            cost=Usage({self.name: self.cost_of(query)}),
        )

    async def extract(self, urls: list[str], **kwargs: Any) -> list[Document]:
        """Scrape each URL.

        Firecrawl's scrape endpoint takes one URL at a time, so a failure is
        naturally per-URL — exactly the granularity hydration wants.
        """
        import asyncio

        semaphore = asyncio.Semaphore(int(kwargs.get("concurrency", 5)))

        async def one(url: str) -> Document:
            async with semaphore:
                return await self._scrape(url)

        return list(await asyncio.gather(*(one(url) for url in urls)))

    async def _scrape(self, url: str) -> Document:
        try:
            response = await self._request(
                "POST",
                f"{self.base_url}/scrape",
                headers=self._headers(),
                json={"url": url, "formats": ["markdown"], "onlyMainContent": True},
            )
        except ProviderError as exc:
            # One bad URL must not fail the batch; report it and move on.
            return Document(url=url, ok=False, provider=self.name, error=str(exc))

        body = response.json()
        data = body.get("data") or {}
        content = data.get("markdown")
        if not content:
            return Document(
                url=url,
                ok=False,
                provider=self.name,
                error=body.get("error") or "no content returned",
                raw=data,
            )
        metadata = data.get("metadata") or {}
        return Document(
            url=metadata.get("sourceURL") or url,
            title=metadata.get("title") or "",
            content=content,
            ok=True,
            provider=self.name,
            raw=data,
        )
