"""Normalized types every provider collapses into.

The whole point of the package is that a caller reasons about ``SearchResult`` and
never about a vendor's JSON shape. ``raw`` always carries the untouched provider
payload so nothing is lost when a caller needs something we didn't model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, IntEnum
from typing import Any


class Capability(str, Enum):
    """What a provider can do. The router never selects a provider for a
    capability it does not declare."""

    SEARCH = "search"
    EXTRACT = "extract"
    ANSWER = "answer"
    NEWS = "news"
    ACADEMIC = "academic"
    REFERENCE = "reference"
    """Encyclopedic lookup (Wikipedia). Separate from SEARCH on purpose: nobody
    asking a general question wants only encyclopedia articles back."""
    DISCUSSION = "discussion"
    """Forum and community threads (Hacker News)."""
    CRAWL = "crawl"


class Depth(IntEnum):
    """How rich each result is. Ordered on purpose: the hydration decision is a
    ``achieved < requested`` comparison."""

    LINKS = 0
    """title + url only."""
    SNIPPETS = 1
    """+ the provider's snippet. The default."""
    SUMMARY = 2
    """+ a provider-generated summary/highlights. Never synthesized locally."""
    CONTENT = 3
    """+ full cleaned page markdown."""


class ContentStatus(str, Enum):
    """Why a result's ``content`` looks the way it does.

    This is the distinction a research consumer actually needs: ``FAILED`` means
    "we could not fetch this page", which is a very different thing from
    ``NOT_REQUESTED`` meaning "you did not ask for content".
    """

    NOT_REQUESTED = "not_requested"
    NATIVE = "native"
    """The search provider returned content inline, no extra call."""
    HYDRATED = "hydrated"
    """Filled in by a follow-up extract call."""
    FAILED = "failed"
    """Hydration was attempted and did not succeed (paywall, 403, timeout)."""
    SKIPPED = "skipped"
    """Outside the ``max_hydrate`` budget."""


class ErrorKind(str, Enum):
    """Drives the router's fallback decision. See ``Provider.classify_error``."""

    AUTH = "auth"
    QUOTA = "quota"
    RATE_LIMIT = "rate_limit"
    TRANSIENT = "transient"
    FATAL = "fatal"


@dataclass(slots=True)
class SearchQuery:
    """A request, after profile/client defaults have been merged in."""

    query: str
    max_results: int = 10
    depth: Depth = Depth.SNIPPETS
    capability: Capability = Capability.SEARCH
    include_domains: list[str] = field(default_factory=list)
    exclude_domains: list[str] = field(default_factory=list)
    start_date: datetime | None = None
    end_date: datetime | None = None
    lang: str | None = None
    region: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    """Provider-specific passthrough, keyed by provider name."""

    def params_for(self, provider: str) -> dict[str, Any]:
        """Escape hatch: per-provider raw params, e.g. ``extra={"exa": {"type": "neural"}}``."""
        value = self.extra.get(provider) or {}
        return dict(value) if isinstance(value, dict) else {}


@dataclass(slots=True)
class SearchResult:
    url: str
    title: str = ""
    snippet: str | None = None
    content: str | None = None
    """Full cleaned markdown. Populated only at CONTENT depth."""
    summary: str | None = None
    """Provider-generated only. The library never writes this itself."""
    content_status: ContentStatus = ContentStatus.NOT_REQUESTED
    score: float | None = None
    published_date: datetime | None = None
    provider: str = ""
    """Which provider found this result."""
    content_provider: str | None = None
    """Which provider extracted the content. Often differs from ``provider``."""
    rank: int = 0
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        """Best available text, richest first. Convenience for callers that just
        want something to feed a model."""
        return self.content or self.summary or self.snippet or ""


@dataclass(slots=True)
class Document:
    """The result of an ``extract`` call against a single URL."""

    url: str
    content: str | None = None
    title: str = ""
    ok: bool = True
    provider: str = ""
    error: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Attempt:
    """One provider call. The full list is attached to every response so a
    failure is diagnosable rather than opaque."""

    provider: str
    capability: Capability
    ok: bool
    latency_ms: float = 0.0
    cost: int = 0
    error: str | None = None
    error_kind: ErrorKind | None = None
    n_results: int = 0


@dataclass(slots=True)
class Usage:
    """What a call actually spent, per provider, in that provider's own units."""

    by_provider: dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(self.by_provider.values())

    def add(self, provider: str, cost: int) -> None:
        if cost:
            self.by_provider[provider] = self.by_provider.get(provider, 0) + cost


@dataclass(slots=True)
class SearchResponse:
    query: str
    results: list[SearchResult] = field(default_factory=list)
    answer: str | None = None
    depth: Depth = Depth.SNIPPETS
    """The depth actually achieved."""
    requested_depth: Depth = Depth.SNIPPETS
    degraded: bool = False
    """True when any result fell short of ``requested_depth``."""
    providers_used: list[str] = field(default_factory=list)
    attempts: list[Attempt] = field(default_factory=list)
    cost: Usage = field(default_factory=Usage)
    notes: list[str] = field(default_factory=list)
    """Why the response looks the way it does, in plain language.

    For situations the caller can act on but would otherwise have to dig out of
    ``result.raw`` — most importantly "you asked for content but no extractor is
    configured", which otherwise looks identical to every page being blocked.
    """

    def __len__(self) -> int:
        return len(self.results)

    def __iter__(self):
        return iter(self.results)

    @property
    def urls(self) -> list[str]:
        return [r.url for r in self.results]


@dataclass(slots=True)
class Answer:
    """A direct answer plus the results it was grounded in."""

    query: str
    answer: str | None = None
    results: list[SearchResult] = field(default_factory=list)
    provider: str = ""
    attempts: list[Attempt] = field(default_factory=list)
    cost: Usage = field(default_factory=Usage)
