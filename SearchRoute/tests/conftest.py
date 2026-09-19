"""Fake providers for exercising the router without touching the network."""

from __future__ import annotations

from typing import Any

import pytest

from searchroute.errors import AuthError, ProviderError, QuotaExceeded, RateLimited
from searchroute.providers.base import Provider
from searchroute.quota import Period, QuotaPolicy, Unit
from searchroute.types import (
    Capability,
    ContentStatus,
    Depth,
    Document,
    SearchQuery,
    SearchResponse,
    SearchResult,
    Usage,
)


class FakeProvider(Provider):
    """A configurable stand-in.

    Records every call so tests can assert not just the final result but which
    providers were actually spent — the thing that matters for a cost-routing
    library.
    """

    name = "fake"
    capabilities = frozenset({Capability.SEARCH, Capability.EXTRACT})
    native_depths = frozenset({Depth.LINKS, Depth.SNIPPETS})
    requires_key = False
    quality_hint = 0.5
    extract_quality = 0.5

    def __init__(
        self,
        name: str = "fake",
        *,
        results: int = 3,
        fail_with: BaseException | None = None,
        depths: frozenset[Depth] | None = None,
        capabilities: frozenset[Capability] | None = None,
        quota: QuotaPolicy | None = None,
        cost: int = 1,
        priority: int = 100,
        quality: float = 0.5,
        extract_quality: float = 0.5,
        last_resort: bool = False,
        extract_fails: set[str] | None = None,
        content_prefix: str = "CONTENT",
        **kwargs: Any,
    ):
        super().__init__(api_key="test", **kwargs)
        self.name = name
        self.native_depths = depths or frozenset({Depth.LINKS, Depth.SNIPPETS})
        if capabilities is not None:
            self.capabilities = capabilities
        self.quota = quota
        self.quality_hint = quality
        self.extract_quality = extract_quality
        self.default_priority = priority
        self.last_resort = last_resort
        self._results = results
        self._fail_with = fail_with
        self._cost = cost
        self._extract_fails = extract_fails or set()
        self._content_prefix = content_prefix
        self.search_calls: list[SearchQuery] = []
        self.extract_calls: list[list[str]] = []

    def cost_of(self, query: SearchQuery) -> int:
        return self._cost

    def extract_cost_of(self, urls: list[str]) -> int:
        return len(urls)

    async def search(self, query: SearchQuery) -> SearchResponse:
        self.search_calls.append(query)
        if self._fail_with is not None:
            raise self._fail_with

        serves_content = query.depth >= Depth.CONTENT and Depth.CONTENT in self.native_depths
        results = [
            SearchResult(
                url=f"https://{self.name}.test/{i}",
                title=f"{self.name} result {i}",
                snippet=f"snippet {i} from {self.name}",
                content=f"{self._content_prefix} {i}" if serves_content else None,
                content_status=(
                    ContentStatus.NATIVE if serves_content else ContentStatus.NOT_REQUESTED
                ),
                provider=self.name,
                rank=i,
                raw={"i": i},
            )
            for i in range(self._results)
        ]
        return SearchResponse(
            query=query.query,
            results=results,
            depth=query.depth if serves_content else Depth.SNIPPETS,
            requested_depth=query.depth,
            providers_used=[self.name],
            cost=Usage({self.name: self._cost}),
        )

    async def extract(self, urls: list[str], **kwargs: Any) -> list[Document]:
        self.extract_calls.append(list(urls))
        if self._fail_with is not None:
            raise self._fail_with
        return [
            Document(
                url=url,
                content=None if url in self._extract_fails else f"{self._content_prefix} {url}",
                ok=url not in self._extract_fails,
                provider=self.name,
                error="blocked" if url in self._extract_fails else None,
            )
            for url in urls
        ]


class SharedURLProvider(FakeProvider):
    """Returns the same URLs as its peers, with cosmetic differences.

    Used to prove dedup and rank fusion actually collapse duplicates instead of
    handing a research agent the same article three times.
    """

    async def search(self, query: SearchQuery) -> SearchResponse:
        self.search_calls.append(query)
        results = [
            SearchResult(
                url=f"https://shared.test/article-{i}?utm_source={self.name}",
                title=f"Article {i}",
                snippet=f"from {self.name}",
                provider=self.name,
                rank=i,
                raw={},
            )
            for i in range(self._results)
        ]
        return SearchResponse(
            query=query.query,
            results=results,
            depth=Depth.SNIPPETS,
            providers_used=[self.name],
            cost=Usage({self.name: self._cost}),
        )


@pytest.fixture
def monthly_quota():
    return QuotaPolicy(limit=100, unit=Unit.CREDITS, period=Period.MONTHLY)


@pytest.fixture
def daily_quota():
    return QuotaPolicy(limit=10, unit=Unit.REQUESTS, period=Period.DAILY)


@pytest.fixture
def auth_error():
    return AuthError("fake", "bad key", status=401)


@pytest.fixture
def quota_error():
    return QuotaExceeded("fake", "out of credits", status=402)


@pytest.fixture
def rate_limit_error():
    return RateLimited("fake", "slow down", status=429)


@pytest.fixture
def fatal_error():
    return ProviderError("fake", "boom", status=400)
