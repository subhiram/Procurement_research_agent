"""Hacker News via the Algolia API — developer discussion.

Keyless and unmetered, ``DISCUSSION`` only. Valuable for a genuinely different
reason from the other sources: it surfaces *practitioner opinion* — what people
actually hit in production — which general web search buries under marketing
pages. That is also why it must never join the general SEARCH chain.

API: https://hn.algolia.com/api
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from ..types import (
    Capability,
    Depth,
    SearchQuery,
    SearchResponse,
    SearchResult,
    Usage,
)
from .base import Provider


class HackerNewsProvider(Provider):
    name = "hackernews"
    capabilities = frozenset({Capability.DISCUSSION})
    native_depths = frozenset({Depth.LINKS, Depth.SNIPPETS})
    requires_key = False
    quota = None
    quality_hint = 0.6
    default_priority = 18
    min_interval = 0.0
    base_url = "https://hn.algolia.com/api/v1"

    def cost_of(self, query: SearchQuery) -> int:
        return 0

    async def search(self, query: SearchQuery) -> SearchResponse:
        params: dict[str, Any] = {
            "query": query.query,
            "hitsPerPage": min(query.max_results, 100),
            "tags": "story",
        }
        if query.start_date:
            params["numericFilters"] = f"created_at_i>{int(query.start_date.timestamp())}"
        params.update(query.params_for(self.name))

        response = await self._request("GET", f"{self.base_url}/search", params=params)
        hits = response.json().get("hits", []) or []

        results = []
        for rank, hit in enumerate(hits):
            object_id = hit.get("objectID", "")
            discussion = f"https://news.ycombinator.com/item?id={object_id}"
            # A story usually links somewhere; an Ask HN post does not. Prefer
            # the linked article, but always keep the thread reachable.
            url = hit.get("url") or discussion

            results.append(
                SearchResult(
                    url=url,
                    title=hit.get("title") or hit.get("story_title") or "",
                    snippet=_snippet(hit),
                    published_date=_created(hit),
                    provider=self.name,
                    rank=rank,
                    raw={**hit, "discussion_url": discussion},
                )
            )

        return SearchResponse(
            query=query.query,
            results=[r for r in results if r.url],
            depth=Depth.SNIPPETS,
            requested_depth=query.depth,
            providers_used=[self.name],
            cost=Usage(),
        )


def _snippet(hit: dict[str, Any]) -> str | None:
    """Points and comment count are the signal here — they say how much
    discussion a thread actually carries."""
    points = hit.get("points")
    comments = hit.get("num_comments")
    author = hit.get("author")
    parts = []
    if points is not None:
        parts.append(f"{points} points")
    if comments is not None:
        parts.append(f"{comments} comments")
    if author:
        parts.append(f"by {author}")
    text = (hit.get("story_text") or hit.get("comment_text") or "").strip()
    head = " · ".join(parts)
    if text:
        import re

        clean = re.sub(r"<[^>]+>", "", text)[:300]
        return f"{head} — {clean}" if head else clean
    return head or None


def _created(hit: dict[str, Any]) -> datetime | None:
    stamp = hit.get("created_at_i")
    if isinstance(stamp, (int, float)):
        return datetime.fromtimestamp(stamp, tz=timezone.utc)
    return None
