"""Google Programmable Search (Custom Search JSON API).

Free tier: 100 queries **per day**, no card. The daily reset makes it the best
filler in the chain — a monthly tier that ran dry on the 3rd is gone until the
1st, but this one comes back tomorrow.

Needs two settings: an API key and a search-engine id (``cx``), from
https://programmablesearchengine.google.com. Set ``GOOGLE_API_KEY`` and
``GOOGLE_CSE_ID``.
"""

from __future__ import annotations

import httpx

from ..errors import AuthError, QuotaExceeded, RateLimited
from ..quota import Anchor, Period, QuotaPolicy, Unit
from ..types import (
    Capability,
    Depth,
    ErrorKind,
    SearchQuery,
    SearchResponse,
    SearchResult,
    Usage,
)
from ._util import parse_date
from .base import Provider

#: Google signals an exhausted quota with 403 plus one of these reasons, not
#: with 402 or 429. Treating that as an auth failure would be wrong twice over:
#: it would trip the circuit breaker permanently and hide a tier that will come
#: back tomorrow.
_QUOTA_REASONS = frozenset(
    {"dailyLimitExceeded", "quotaExceeded", "rateLimitExceeded", "userRateLimitExceeded"}
)


class GoogleCSEProvider(Provider):
    name = "google_cse"
    capabilities = frozenset({Capability.SEARCH})
    native_depths = frozenset({Depth.LINKS, Depth.SNIPPETS})
    requires_key = True
    quota = QuotaPolicy(
        limit=100, unit=Unit.REQUESTS, period=Period.DAILY, anchor=Anchor.CALENDAR
    )
    quality_hint = 0.8
    """Real Google index; excellent recall, snippets only."""
    default_priority = 40
    base_url = "https://www.googleapis.com/customsearch/v1"

    #: The API caps a single request at 10 results.
    MAX_PER_REQUEST = 10

    def __init__(self, *args, cx: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.cx = cx

    @property
    def configured(self) -> bool:
        # A key without a search-engine id cannot make a single call, so treat
        # it as unconfigured rather than failing at request time.
        return bool(self.api_key) and bool(self.cx)

    def cost_of(self, query: SearchQuery) -> int:
        # Billed per request, and each request returns at most 10 results.
        return max(1, -(-min(query.max_results, 100) // self.MAX_PER_REQUEST))

    def _quota_reason(self, response: httpx.Response) -> str | None:
        try:
            errors = response.json().get("error", {}).get("errors", [])
        except Exception:
            return None
        for item in errors:
            if item.get("reason") in _QUOTA_REASONS:
                return item.get("reason")
        return None

    def raise_for_status(self, response: httpx.Response) -> None:
        if response.is_success:
            return
        if response.status_code in (403, 429):
            reason = self._quota_reason(response)
            if reason:
                raise QuotaExceeded(
                    self.name,
                    f"daily quota exhausted ({reason}); resets at midnight UTC",
                    status=response.status_code,
                )
            if response.status_code == 429:
                raise RateLimited(self.name, "rate limited", status=429)
            raise AuthError(
                self.name, f"forbidden: {response.text[:200]}", status=response.status_code
            )
        super().raise_for_status(response)

    def classify_error(self, exc: BaseException) -> ErrorKind:
        if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 403:
            if self._quota_reason(exc.response):
                return ErrorKind.QUOTA
        return super().classify_error(exc)

    async def search(self, query: SearchQuery) -> SearchResponse:
        wanted = min(query.max_results, 100)
        results: list[SearchResult] = []
        cost = 0

        # Page through in tens until we have what was asked for.
        for start in range(1, wanted + 1, self.MAX_PER_REQUEST):
            params: dict[str, object] = {
                "key": self.api_key,
                "cx": self.cx,
                "q": query.query,
                "num": min(self.MAX_PER_REQUEST, wanted - len(results)),
                "start": start,
            }
            if query.lang:
                params["lr"] = f"lang_{query.lang}"
            if query.region:
                params["gl"] = query.region
            if query.include_domains and len(query.include_domains) == 1:
                # The API takes a single siteSearch domain; multiples are left
                # to the client-side filter so semantics stay consistent.
                params["siteSearch"] = query.include_domains[0]
                params["siteSearchFilter"] = "i"
            params.update(query.params_for(self.name))

            response = await self._request("GET", self.base_url, params=params)
            cost += 1
            body = response.json()
            items = body.get("items") or []
            if not items:
                break

            for item in items:
                results.append(
                    SearchResult(
                        url=item.get("link", ""),
                        title=item.get("title") or "",
                        snippet=item.get("snippet"),
                        published_date=parse_date(
                            (item.get("pagemap", {}).get("metatags") or [{}])[0].get(
                                "article:published_time"
                            )
                        ),
                        provider=self.name,
                        rank=len(results),
                        raw=item,
                    )
                )
            if len(results) >= wanted:
                break

        return SearchResponse(
            query=query.query,
            results=results[:wanted],
            depth=Depth.SNIPPETS,
            requested_depth=query.depth,
            providers_used=[self.name],
            cost=Usage({self.name: cost}),
        )
