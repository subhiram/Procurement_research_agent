"""Wikipedia — encyclopedic reference, keyless and unmetered.

Declares ``REFERENCE``, never ``SEARCH``. That is the whole reason this provider
can exist safely: somebody asking "best pizza in NYC" must not get encyclopedia
articles, and the router's capability filter makes that automatic — a provider
is invisible to any capability it doesn't declare.

It serves ``CONTENT`` natively (the API returns clean article extracts), so it
needs no hydration hop. It deliberately does **not** declare ``EXTRACT``: that
would put it in the general extract chain, where it would be offered arbitrary
URLs and fail on everything that isn't a Wikipedia article.

API: https://www.mediawiki.org/wiki/API:Main_page
"""

from __future__ import annotations

from typing import Any

from ..types import (
    Capability,
    ContentStatus,
    Depth,
    SearchQuery,
    SearchResponse,
    SearchResult,
    Usage,
)
from ._util import parse_date
from .base import Provider


class WikipediaProvider(Provider):
    name = "wikipedia"
    capabilities = frozenset({Capability.REFERENCE})
    native_depths = frozenset({Depth.LINKS, Depth.SNIPPETS, Depth.CONTENT})
    requires_key = False
    quota = None
    quality_hint = 0.85
    """Excellent for what it covers, useless outside it — which is why it is
    scoped to REFERENCE rather than competing in general search."""
    default_priority = 15
    min_interval = 0.0

    #: Wikimedia's policy rejects generic browser User-Agents from scripts. We
    #: proved this in http_extract: a spoofed Chrome UA gets 403, a descriptive
    #: one gets 200.
    base_url = "https://en.wikipedia.org/w/api.php"

    #: Article extracts are large; cap what we pull per article.
    CHARS_PER_ARTICLE = 20_000

    def __init__(self, *args, lang: str = "en", contact: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.lang = lang
        self.contact = contact
        self.base_url = f"https://{lang}.wikipedia.org/w/api.php"

    def cost_of(self, query: SearchQuery) -> int:
        return 0

    def _headers(self) -> dict[str, str]:
        agent = "SearchRoute/0.1.0 (+https://github.com/subhiram/searchroute)"
        if self.contact:
            agent = f"SearchRoute/0.1.0 (+https://github.com/subhiram/searchroute; {self.contact})"
        return {"User-Agent": agent, "Accept": "application/json"}

    async def search(self, query: SearchQuery) -> SearchResponse:
        lang = query.lang or self.lang
        base = f"https://{lang}.wikipedia.org/w/api.php"

        hits = await self._find(base, query)
        if not hits:
            return SearchResponse(
                query=query.query,
                results=[],
                depth=Depth.SNIPPETS,
                requested_depth=query.depth,
                providers_used=[self.name],
                cost=Usage(),
            )

        extracts: dict[int, str] = {}
        if query.depth >= Depth.CONTENT:
            extracts = await self._extracts(base, [h["pageid"] for h in hits])

        results = []
        for rank, hit in enumerate(hits):
            page_id = hit["pageid"]
            text = extracts.get(page_id)
            title = hit.get("title", "")
            results.append(
                SearchResult(
                    url=f"https://{lang}.wikipedia.org/?curid={page_id}",
                    title=title,
                    # The API returns the snippet with HTML search-match markup.
                    snippet=_strip_markup(hit.get("snippet", "")),
                    content=text,
                    content_status=(
                        (ContentStatus.NATIVE if text else ContentStatus.FAILED)
                        if query.depth >= Depth.CONTENT
                        else ContentStatus.NOT_REQUESTED
                    ),
                    published_date=parse_date(hit.get("timestamp")),
                    provider=self.name,
                    content_provider=self.name if text else None,
                    rank=rank,
                    raw=hit,
                )
            )

        return SearchResponse(
            query=query.query,
            results=results,
            depth=Depth.CONTENT if extracts else Depth.SNIPPETS,
            requested_depth=query.depth,
            providers_used=[self.name],
            cost=Usage(),
        )

    async def _find(self, base: str, query: SearchQuery) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "action": "query",
            "list": "search",
            "srsearch": query.query,
            "srlimit": min(query.max_results, 50),
            "format": "json",
        }
        params.update(query.params_for(self.name))
        response = await self._request("GET", base, params=params, headers=self._headers())
        body = response.json()
        return body.get("query", {}).get("search", []) or []

    async def _extracts(self, base: str, page_ids: list[int]) -> dict[int, str]:
        """Fetch plain-text article bodies for the pages we just found."""
        response = await self._request(
            "GET",
            base,
            params={
                "action": "query",
                "prop": "extracts",
                "explaintext": 1,
                "exlimit": "max",
                "pageids": "|".join(str(p) for p in page_ids),
                "format": "json",
            },
            headers=self._headers(),
        )
        pages = response.json().get("query", {}).get("pages", {}) or {}
        out: dict[int, str] = {}
        for page in pages.values():
            text = page.get("extract")
            if text:
                out[page.get("pageid")] = text[: self.CHARS_PER_ARTICLE]
        return out


def _strip_markup(text: str) -> str:
    """Wikipedia search snippets come back with <span class="searchmatch"> tags."""
    import re

    return re.sub(r"<[^>]+>", "", text or "").strip()
