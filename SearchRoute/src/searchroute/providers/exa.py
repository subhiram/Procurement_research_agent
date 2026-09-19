"""Exa — neural/semantic search with inline page contents.

Free tier: $20 on signup plus $10/month recurring, no card. Balance-capped, so
it cannot overdraw. The most generous recurring tier of the group, and the only
one that returns per-result summaries and highlights natively — which is why it
is the default first hop for SUMMARY and CONTENT depth.
API: https://docs.exa.ai
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


class ExaProvider(Provider):
    name = "exa"
    capabilities = frozenset(
        {Capability.SEARCH, Capability.EXTRACT, Capability.ANSWER, Capability.ACADEMIC}
    )
    native_depths = frozenset({Depth.LINKS, Depth.SNIPPETS, Depth.SUMMARY, Depth.CONTENT})
    requires_key = True
    # Exa's free tier is denominated in dollars, not calls. We model it in
    # tenths of a cent so integer accounting stays exact: $10 == 10_000 units.
    quota = QuotaPolicy(limit=10_000, unit=Unit.CREDITS, period=Period.MONTHLY)
    quality_hint = 0.9
    extract_quality = 0.75
    default_priority = 10
    base_url = "https://api.exa.ai"

    #: Roughly $0.005/search and $0.001/page of contents, in tenths of a cent.
    COST_SEARCH = 50
    COST_CONTENT_PER_RESULT = 10

    def cost_of(self, query: SearchQuery) -> int:
        cost = self.COST_SEARCH
        if query.depth >= Depth.SUMMARY:
            cost += self.COST_CONTENT_PER_RESULT * query.max_results
        return cost

    def extract_cost_of(self, urls: list[str]) -> int:
        return self.COST_CONTENT_PER_RESULT * len(urls)

    def _headers(self) -> dict[str, str]:
        return {"x-api-key": self.api_key or "", "Content-Type": "application/json"}

    def _contents_spec(self, depth: Depth) -> dict[str, Any] | None:
        """Translate our depth into Exa's contents options.

        This is the "native path": one call returns text and summaries instead of
        us paying for a second extract round trip.
        """
        if depth <= Depth.SNIPPETS:
            # Still ask for highlights — they are Exa's snippet equivalent.
            return {"highlights": True}
        if depth is Depth.SUMMARY:
            return {"summary": True, "highlights": True}
        return {"text": True, "summary": True, "highlights": True}

    def _payload(self, query: SearchQuery) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "query": query.query,
            "numResults": query.max_results,
            "contents": self._contents_spec(query.depth),
        }
        if query.capability is Capability.ACADEMIC:
            payload["category"] = "research paper"
        elif query.capability is Capability.NEWS:
            payload["category"] = "news"
        if query.include_domains:
            payload["includeDomains"] = query.include_domains
        if query.exclude_domains:
            payload["excludeDomains"] = query.exclude_domains
        if query.start_date:
            payload["startPublishedDate"] = query.start_date.isoformat()
        if query.end_date:
            payload["endPublishedDate"] = query.end_date.isoformat()
        payload.update(query.params_for(self.name))
        return payload

    def _to_results(self, data: dict[str, Any], depth: Depth) -> list[SearchResult]:
        results = []
        for rank, item in enumerate(data.get("results", [])):
            text = item.get("text")
            summary = item.get("summary")
            highlights = item.get("highlights") or []
            snippet = summary or (highlights[0] if highlights else None)

            if depth >= Depth.CONTENT:
                status = ContentStatus.NATIVE if text else ContentStatus.FAILED
            else:
                status = ContentStatus.NOT_REQUESTED

            results.append(
                SearchResult(
                    url=item.get("url", ""),
                    title=item.get("title") or "",
                    snippet=snippet,
                    content=text if depth >= Depth.CONTENT else None,
                    summary=summary,
                    content_status=status,
                    score=item.get("score"),
                    published_date=parse_date(item.get("publishedDate")),
                    provider=self.name,
                    content_provider=self.name if (depth >= Depth.CONTENT and text) else None,
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
            json=self._payload(query),
        )
        data = response.json()
        results = self._to_results(data, query.depth)
        return SearchResponse(
            query=query.query,
            results=results,
            depth=query.depth,
            requested_depth=query.depth,
            providers_used=[self.name],
            cost=Usage({self.name: self.cost_of(query)}),
        )

    async def answer(self, query: SearchQuery) -> Answer:
        response = await self._request(
            "POST",
            f"{self.base_url}/answer",
            headers=self._headers(),
            json={"query": query.query, "text": True},
        )
        data = response.json()
        citations = [
            SearchResult(
                url=item.get("url", ""),
                title=item.get("title") or "",
                snippet=item.get("snippet"),
                content=item.get("text"),
                content_status=(
                    ContentStatus.NATIVE if item.get("text") else ContentStatus.NOT_REQUESTED
                ),
                provider=self.name,
                rank=rank,
                raw=item,
            )
            for rank, item in enumerate(data.get("citations", []))
        ]
        return Answer(
            query=query.query,
            answer=data.get("answer"),
            results=citations,
            provider=self.name,
            attempts=[Attempt(self.name, Capability.ANSWER, ok=True)],
            cost=Usage({self.name: self.COST_SEARCH}),
        )

    async def extract(self, urls: list[str], **kwargs: Any) -> list[Document]:
        response = await self._request(
            "POST",
            f"{self.base_url}/contents",
            headers=self._headers(),
            json={"urls": urls, "text": True},
        )
        data = response.json()
        docs = {
            item.get("url", ""): Document(
                url=item.get("url", ""),
                title=item.get("title") or "",
                content=item.get("text"),
                ok=bool(item.get("text")),
                provider=self.name,
                raw=item,
            )
            for item in data.get("results", [])
        }
        return [
            docs.get(url, Document(url=url, ok=False, provider=self.name, error="no result"))
            for url in urls
        ]
