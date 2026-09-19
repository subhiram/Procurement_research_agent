"""SearchRoute — one search interface, many providers.

    from searchroute import SearchRoute

    sr = SearchRoute()                       # auto-discovers keys from the env
    for hit in sr.search("what is RAG?"):
        print(hit.title, hit.url)

Ask for more per call when you need it:

    sr.search("q", depth="content")          # full cleaned markdown
    sr.search("q", strategy="fanout", n=3)   # several providers, fused
"""

from .client import AsyncSearchRoute, SearchRoute
from .config import Settings, discover_providers
from .errors import (
    AuthError,
    CapabilityNotSupported,
    ConfigError,
    NoProviderAvailable,
    ProviderError,
    QuotaExceeded,
    RateLimited,
    SearchRouteError,
    TransientError,
)
from .observability import (
    EVENT_EXTRACT_FAILURE,
    EVENT_FAILURE,
    EVENT_SUCCESS,
    StatsCollector,
    log_events,
)
from .profiles import PROFILES
from .providers import Provider, register
from .providers import available as available_providers
from .quota import Anchor, Period, QuotaPolicy, Unit
from .types import (
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

__version__ = "0.1.0"

__all__ = [
    "SearchRoute",
    "AsyncSearchRoute",
    "Provider",
    "register",
    "available_providers",
    "Settings",
    "discover_providers",
    "PROFILES",
    # observability
    "StatsCollector",
    "log_events",
    "EVENT_SUCCESS",
    "EVENT_FAILURE",
    "EVENT_EXTRACT_FAILURE",
    # types
    "Answer",
    "Attempt",
    "Capability",
    "ContentStatus",
    "Depth",
    "Document",
    "SearchQuery",
    "SearchResponse",
    "SearchResult",
    "Usage",
    # quota
    "QuotaPolicy",
    "Period",
    "Unit",
    "Anchor",
    # errors
    "SearchRouteError",
    "ConfigError",
    "CapabilityNotSupported",
    "ProviderError",
    "AuthError",
    "QuotaExceeded",
    "RateLimited",
    "TransientError",
    "NoProviderAvailable",
]
