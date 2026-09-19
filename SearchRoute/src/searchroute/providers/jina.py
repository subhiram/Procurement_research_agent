"""Jina Reader — search via s.jina.ai, page reading via r.jina.ai.

The two endpoints have different auth requirements, verified against the live
API: ``r.jina.ai`` (reading a page) answers **without** a key, while
``s.jina.ai`` (search) returns **401** unless one is supplied. So this provider
declares ``keyed_capabilities = {SEARCH}`` — keyless it offers extraction only,
and with a key it offers both.

That makes it the best keyless extractor here, better than the local fallback
because Jina renders pages server-side, so it sits between the paid extractors
and ``http_extract`` in the chain.
API: https://jina.ai/reader
"""

from __future__ import annotations

from typing import Any

from ..errors import ProviderError
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


class JinaProvider(Provider):
    name = "jina"
    capabilities = frozenset({Capability.SEARCH, Capability.EXTRACT})
    native_depths = frozenset({Depth.LINKS, Depth.SNIPPETS, Depth.CONTENT})
    requires_key = False
    """The reader endpoint needs no credentials."""
    keyed_capabilities = frozenset({Capability.SEARCH})
    """s.jina.ai returns 401 without a key; r.jina.ai does not."""
    quota = None
    """Token-metered upstream rather than call-metered, so we don't model a
    local allowance — we rely on the provider's own 402/429 to tell us."""
    quality_hint = 0.6
    extract_quality = 0.8
    default_priority = 400
    search_url = "https://s.jina.ai"
    reader_url = "https://r.jina.ai"
    timeout = 45.0

    def cost_of(self, query: SearchQuery) -> int:
        return 0

    def extract_cost_of(self, urls: list[str]) -> int:
        return 0

    def _headers(self, **extra: str) -> dict[str, str]:
        headers = {"Accept": "application/json", **extra}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def search(self, query: SearchQuery) -> SearchResponse:
        """Search. Jina always reads the pages it finds, so results arrive with
        content whether or not we asked — we just don't surface it below CONTENT
        depth, to keep responses the same shape as every other provider."""
        wants_content = query.depth >= Depth.CONTENT
        headers = self._headers()
        if not wants_content:
            # Ask for less when we don't need the body; it's faster and cheaper.
            headers["X-Respond-With"] = "no-content"

        response = await self._request(
            "GET",
            f"{self.search_url}/",
            headers=headers,
            params={"q": query.query, **query.params_for(self.name)},
        )
        body = response.json()
        rows = body.get("data") or []
        if not isinstance(rows, list):
            raise ProviderError(self.name, f"unexpected response shape: {type(rows).__name__}")

        results = []
        for rank, item in enumerate(rows[: query.max_results]):
            content = item.get("content")
            results.append(
                SearchResult(
                    url=item.get("url", ""),
                    title=item.get("title") or "",
                    snippet=item.get("description") or (content[:300] if content else None),
                    content=content if wants_content else None,
                    content_status=(
                        (ContentStatus.NATIVE if content else ContentStatus.FAILED)
                        if wants_content
                        else ContentStatus.NOT_REQUESTED
                    ),
                    published_date=parse_date(item.get("publishedTime") or item.get("date")),
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
            cost=Usage(),
        )

    async def extract(self, urls: list[str], **kwargs: Any) -> list[Document]:
        import asyncio

        semaphore = asyncio.Semaphore(int(kwargs.get("concurrency", 3)))

        async def one(url: str) -> Document:
            async with semaphore:
                return await self._read(url)

        return list(await asyncio.gather(*(one(url) for url in urls)))

    async def _read(self, url: str) -> Document:
        try:
            response = await self._request(
                "GET",
                f"{self.reader_url}/{url}",
                headers=self._headers(**{"X-Return-Format": "markdown"}),
            )
        except ProviderError as exc:
            return Document(url=url, ok=False, provider=self.name, error=str(exc))

        body = response.json()
        data = body.get("data") or {}
        content = data.get("content")
        if not content:
            return Document(
                url=url, ok=False, provider=self.name, error="no content returned", raw=data
            )
        return Document(
            url=data.get("url") or url,
            title=data.get("title") or "",
            content=content,
            ok=True,
            provider=self.name,
            raw=data,
        )
