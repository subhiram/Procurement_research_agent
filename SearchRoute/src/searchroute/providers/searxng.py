"""SearXNG — a self-hosted metasearch instance.

The only genuinely unlimited option here: point it at your own instance and it
costs nothing per query forever. That makes it the ideal head of the chain when
available, which is why it sorts ahead of the metered providers.

Set ``SEARXNG_URL`` to your instance. Public instances exist but mostly
rate-limit or block automated use — run your own if you intend to rely on it.
The instance must have the JSON format enabled in ``settings.yml``:

    search:
      formats:
        - html
        - json
"""

from __future__ import annotations

from ..errors import ConfigError, ProviderError
from ..types import (
    Capability,
    Depth,
    SearchQuery,
    SearchResponse,
    SearchResult,
    Usage,
)
from ._util import clamp_score, parse_date
from .base import Provider


class SearXNGProvider(Provider):
    name = "searxng"
    capabilities = frozenset({Capability.SEARCH, Capability.NEWS, Capability.ACADEMIC})
    native_depths = frozenset({Depth.LINKS, Depth.SNIPPETS})
    requires_key = False
    quota = None
    """Self-hosted: unmetered by definition."""
    quality_hint = 0.6
    default_priority = 5
    """Ahead of the metered providers — spending nothing is strictly better when
    the results are good enough."""

    _CATEGORIES = {
        Capability.SEARCH: "general",
        Capability.NEWS: "news",
        Capability.ACADEMIC: "science",
    }

    def __init__(self, *args, base_url: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_url = (base_url or "").rstrip("/")

    @property
    def configured(self) -> bool:
        # Keyless, but useless without an instance to talk to.
        return bool(self.base_url)

    def cost_of(self, query: SearchQuery) -> int:
        return 0

    async def search(self, query: SearchQuery) -> SearchResponse:
        if not self.base_url:
            raise ConfigError("searxng requires SEARXNG_URL (your instance's base URL)")

        params: dict[str, object] = {
            "q": query.query,
            "format": "json",
            "categories": self._CATEGORIES.get(query.capability, "general"),
        }
        if query.lang:
            params["language"] = query.lang
        params.update(query.params_for(self.name))

        response = await self._request("GET", f"{self.base_url}/search", params=params)
        try:
            body = response.json()
        except ValueError as exc:
            # The single most common misconfiguration, and the error message
            # from a raw HTML body would be useless without this hint.
            raise ProviderError(
                self.name,
                "instance did not return JSON — enable the 'json' format in "
                "settings.yml under search.formats",
            ) from exc

        rows = body.get("results") or []
        results = [
            SearchResult(
                url=item.get("url", ""),
                title=item.get("title") or "",
                snippet=item.get("content"),
                score=clamp_score(item.get("score")),
                published_date=parse_date(item.get("publishedDate")),
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
            cost=Usage(),
        )
