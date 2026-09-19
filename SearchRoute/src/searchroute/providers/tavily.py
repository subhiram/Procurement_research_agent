"""Tavily — search built for agents.

Free tier: 1,000 credits/month, renews, no card. Returns clean agent-ready
snippets, can return raw page content inline, and has a native answer mode.
API: https://docs.tavily.com
"""

from __future__ import annotations

from typing import Any

from ..quota import Period, QuotaPolicy, Unit
from ..types import (
    Answer,
    Attempt,
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


class TavilyProvider(Provider):
    name = "tavily"
    capabilities = frozenset(
        {Capability.SEARCH, Capability.EXTRACT, Capability.ANSWER, Capability.NEWS}
    )
    native_depths = frozenset({Depth.LINKS, Depth.SNIPPETS, Depth.CONTENT})
    # No SUMMARY: Tavily returns a whole-query answer, not a per-result summary.
    requires_key = True
    quota = QuotaPolicy(limit=1000, unit=Unit.CREDITS, period=Period.MONTHLY)
    quality_hint = 0.85
    extract_quality = 0.8
    default_priority = 20
    base_url = "https://api.tavily.com"

    def cost_of(self, query: SearchQuery) -> int:
        # Basic search is 1 credit; advanced (needed for raw content) is 2.
        return 2 if query.depth >= Depth.CONTENT else 1

    def extract_cost_of(self, urls: list[str]) -> int:
        # Extract bills per 5 URLs.
        return max(1, -(-len(urls) // 5))

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _payload(self, query: SearchQuery, *, include_answer: bool) -> dict[str, Any]:
        wants_content = query.depth >= Depth.CONTENT
        payload: dict[str, Any] = {
            "query": query.query,
            "max_results": query.max_results,
            "search_depth": "advanced" if wants_content else "basic",
            "include_raw_content": "markdown" if wants_content else False,
            "include_answer": include_answer,
        }
        if query.capability is Capability.NEWS:
            payload["topic"] = "news"
        if query.include_domains:
            payload["include_domains"] = query.include_domains
        if query.exclude_domains:
            payload["exclude_domains"] = query.exclude_domains
        if query.start_date:
            payload["start_date"] = query.start_date.date().isoformat()
        if query.end_date:
            payload["end_date"] = query.end_date.date().isoformat()
        payload.update(query.params_for(self.name))
        return payload

    def _to_results(self, data: dict[str, Any], query: SearchQuery) -> list[SearchResult]:
        wants_content = query.depth >= Depth.CONTENT
        results = []
        for rank, item in enumerate(data.get("results", [])):
            raw_content = item.get("raw_content")
            results.append(
                SearchResult(
                    url=item.get("url", ""),
                    title=item.get("title", ""),
                    snippet=item.get("content"),
                    content=raw_content if wants_content else None,
                    content_status=(
                        ContentStatus.NATIVE
                        if wants_content and raw_content
                        else ContentStatus.FAILED
                        if wants_content
                        else ContentStatus.NOT_REQUESTED
                    ),
                    score=item.get("score"),
                    published_date=parse_date(item.get("published_date")),
                    provider=self.name,
                    content_provider=self.name if (wants_content and raw_content) else None,
                    rank=rank,
                    raw=item,
                )
            )
        return results

    async def search(self, query: SearchQuery) -> SearchResponse:
        response = await self._request(
            "POST",
            f"{self.base_url}/search",
            headers=self._headers(),
            json=self._payload(query, include_answer=False),
        )
        data = response.json()
        results = self._to_results(data, query)
        achieved = query.depth if all(r.content for r in results) else Depth.SNIPPETS
        return SearchResponse(
            query=query.query,
            results=results,
            depth=achieved if query.depth >= Depth.CONTENT else query.depth,
            requested_depth=query.depth,
            providers_used=[self.name],
            cost=Usage({self.name: self.cost_of(query)}),
        )

    async def answer(self, query: SearchQuery) -> Answer:
        response = await self._request(
            "POST",
            f"{self.base_url}/search",
            headers=self._headers(),
            json=self._payload(query, include_answer=True),
        )
        data = response.json()
        return Answer(
            query=query.query,
            answer=data.get("answer"),
            results=self._to_results(data, query),
            provider=self.name,
            attempts=[Attempt(self.name, Capability.ANSWER, ok=True)],
            cost=Usage({self.name: self.cost_of(query)}),
        )

    async def extract(self, urls: list[str], **kwargs: Any) -> list[Document]:
        response = await self._request(
            "POST",
            f"{self.base_url}/extract",
            headers=self._headers(),
            json={
                "urls": urls,
                "format": "markdown",
                "extract_depth": kwargs.get("extract_depth", "basic"),
            },
        )
        data = response.json()
        docs: dict[str, Document] = {}
        for item in data.get("results", []):
            url = item.get("url", "")
            docs[url] = Document(
                url=url,
                content=item.get("raw_content"),
                ok=bool(item.get("raw_content")),
                provider=self.name,
                raw=item,
            )
        for item in data.get("failed_results", []):
            url = item.get("url", "")
            docs[url] = Document(
                url=url,
                ok=False,
                provider=self.name,
                error=item.get("error", "extraction failed"),
                raw=item,
            )
        # Preserve caller order, and never silently drop a URL.
        return [
            docs.get(url, Document(url=url, ok=False, provider=self.name, error="no result"))
            for url in urls
        ]
