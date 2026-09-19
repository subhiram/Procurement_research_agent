"""DuckDuckGo via the ``ddgs`` library — the keyless last resort.

Deliberately last in every default chain. This is not an official API: ``ddgs``
scrapes, it trips bot detection well under 30 requests/minute from one IP, and
the library positions itself as educational-use-only. It exists here so that
search still returns *something* when every keyed provider is drained or down —
it is a safety net, not a tier you should plan capacity around.

Because it is unofficial we treat rate limiting as expected rather than
exceptional: a 202/ratelimit response cools the provider off instead of
retrying into a longer ban.
"""

from __future__ import annotations

import asyncio
from typing import Any

from ..errors import ProviderError, RateLimited
from ..types import (
    Capability,
    Depth,
    SearchQuery,
    SearchResponse,
    SearchResult,
    Usage,
)
from .base import Provider


class DuckDuckGoProvider(Provider):
    name = "duckduckgo"
    capabilities = frozenset({Capability.SEARCH, Capability.NEWS})
    native_depths = frozenset({Depth.LINKS, Depth.SNIPPETS})
    requires_key = False
    quota = None  # unmetered, but IP rate-limited — see module docstring
    quality_hint = 0.45
    extract_quality = 0.0
    default_priority = 900
    last_resort = True

    #: Paced, because "unmetered" above is not the same as "unlimited".
    #:
    #: There is no quota to track, so nothing else here slows it down - but it
    #: is rate-limited by IP, and `ddgs` responds to being pushed by rotating
    #: onto a backup backend, which is where the failures actually show up
    #: (an agent fanning six queries out three at a time produced a DNS error
    #: from mojeek rather than a rate-limit from DuckDuckGo).
    #:
    #: It matters more than the number suggests: this is the only keyless web
    #: search provider, so on a deployment with no API keys it is the entire
    #: web-search capability, and a burst failure means zero results rather
    #: than a slower answer. The RateGate holds its lock across the wait, so
    #: concurrent callers queue instead of bursting.
    min_interval = 1.0

    def cost_of(self, query: SearchQuery) -> int:
        return 0

    def _search_sync(self, query: SearchQuery) -> list[dict[str, Any]]:
        try:
            from ddgs import DDGS
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise ProviderError(
                self.name,
                "the 'ddgs' package is required: pip install 'searchroute[ddg]'",
            ) from exc

        kwargs: dict[str, Any] = {"max_results": query.max_results}
        if query.region:
            kwargs["region"] = query.region
        kwargs.update(query.params_for(self.name))

        with DDGS() as ddgs:
            if query.capability is Capability.NEWS:
                return list(ddgs.news(query.query, **kwargs))
            return list(ddgs.text(query.query, **kwargs))

    async def search(self, query: SearchQuery) -> SearchResponse:
        try:
            # ddgs is synchronous; keep it off the event loop.
            rows = await asyncio.to_thread(self._search_sync, query)
        except ProviderError:
            raise
        except Exception as exc:
            message = str(exc).lower()
            if "ratelimit" in message or "202" in message or "429" in message:
                raise RateLimited(
                    self.name,
                    f"duckduckgo rate limited (unofficial endpoint): {exc}",
                ) from exc
            raise ProviderError(self.name, f"duckduckgo search failed: {exc}") from exc

        results = [
            SearchResult(
                url=row.get("href") or row.get("url") or "",
                title=row.get("title") or "",
                snippet=row.get("body") or row.get("excerpt"),
                provider=self.name,
                rank=rank,
                raw=row,
            )
            for rank, row in enumerate(rows)
        ]
        return SearchResponse(
            query=query.query,
            results=[r for r in results if r.url],
            depth=Depth.SNIPPETS,
            requested_depth=query.depth,
            providers_used=[self.name],
            cost=Usage(),
        )
