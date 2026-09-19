"""Provider registry.

Built-ins register themselves at import. Third parties can add their own with
the ``@register`` decorator or by handing the client a ready-made instance, and
it will be routed, quota-tracked and failed-over exactly like a built-in — an
internal company search endpoint is a first-class provider here.
"""

from __future__ import annotations

from typing import TypeVar

from ..errors import ConfigError
from .base import Provider

_REGISTRY: dict[str, type[Provider]] = {}

P = TypeVar("P", bound=Provider)


def register(cls: type[P]) -> type[P]:
    """Class decorator. The provider's ``name`` is its registry key."""
    if not cls.name:
        raise ConfigError(f"{cls.__name__} must define a non-empty 'name'")
    _REGISTRY[cls.name] = cls
    return cls


def get(name: str) -> type[Provider]:
    try:
        return _REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY)) or "none"
        raise ConfigError(f"unknown provider {name!r}; known providers: {known}") from None


def available() -> list[str]:
    return sorted(_REGISTRY)


def all_classes() -> dict[str, type[Provider]]:
    return dict(_REGISTRY)


def load_entry_points() -> None:
    """Discover third-party providers published under the ``searchroute.providers``
    entry-point group, so installing a plugin package is enough to enable it."""
    try:
        from importlib.metadata import entry_points
    except ImportError:  # pragma: no cover
        return
    try:
        points = entry_points(group="searchroute.providers")
    except TypeError:  # pragma: no cover - Python 3.9 style API
        points = entry_points().get("searchroute.providers", [])  # type: ignore[attr-defined]
    for point in points:
        try:
            candidate = point.load()
        except Exception:  # pragma: no cover - a broken plugin must not break import
            continue
        if isinstance(candidate, type) and issubclass(candidate, Provider):
            register(candidate)


def _register_builtins() -> None:
    from .arxiv import ArxivProvider
    from .brave import BraveProvider
    from .crossref import CrossrefProvider
    from .duckduckgo import DuckDuckGoProvider
    from .exa import ExaProvider
    from .firecrawl import FirecrawlProvider
    from .google_cse import GoogleCSEProvider
    from .hackernews import HackerNewsProvider
    from .http_extract import HTTPExtractProvider
    from .jina import JinaProvider
    from .pubmed import PubMedProvider
    from .searchapi import SearchApiProvider
    from .searxng import SearXNGProvider
    from .serpapi import SerpAPIProvider
    from .serper import SerperProvider
    from .tavily import TavilyProvider
    from .wikipedia import WikipediaProvider

    for cls in (
        SearXNGProvider,      # free and unmetered when self-hosted
        ExaProvider,          # most generous recurring tier
        ArxivProvider,        # capability-scoped: ACADEMIC only
        PubMedProvider,       # ACADEMIC only
        WikipediaProvider,    # REFERENCE only
        CrossrefProvider,     # ACADEMIC only
        HackerNewsProvider,   # DISCUSSION only
        TavilyProvider,
        FirecrawlProvider,
        GoogleCSEProvider,    # resets daily
        SerpAPIProvider,
        SerperProvider,       # one-time grant: reserve tank
        SearchApiProvider,
        BraveProvider,        # opt-in only; costs money
        JinaProvider,
        HTTPExtractProvider,  # keyless extract tail
        DuckDuckGoProvider,   # keyless search tail
    ):
        register(cls)

    # Third-party providers published under the entry-point group. Guarded so a
    # plain directory copy — no install metadata — is a silent no-op rather than
    # an import-time crash.
    try:
        load_entry_points()
    except Exception:  # pragma: no cover - defensive
        pass


_register_builtins()
