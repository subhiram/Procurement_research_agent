"""llm_router - one callable that routes an LLM request across free tiers.

    from llm_router import route
    response = route(messages, strategy="free_first")

    from llm_router import LLMRouter
    agent = create_react_agent(model=LLMRouter(strategy="sticky"), tools=tools)

Nothing outside this package should import a provider SDK. Adding or dropping a
provider is a change to config/models.yaml and config/limits.yaml plus, for a
new provider, one wrapper class in providers.py.
"""

try:
    # Provider keys live in os.environ (see Endpoint.api_key in registry.py), so
    # a .env file has to be loaded into the process before anything reads it.
    # find_dotenv() walks up from the current working directory, so this works
    # whether the caller's script lives at the repo root or a few folders deep.
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True))
except ImportError:
    pass  # python-dotenv is optional; real env vars still work without it

from .ladder import Candidate, explain, resolve_candidates, soonest_retry
from .ledger import Availability, UsageLedger
from .policies import (
    POLICIES,
    SessionState,
    free_first,
    get_policy,
    register_policy,
    sticky,
)
from .providers import (
    AllCandidatesExhausted,
    AuthError,
    BaseProvider,
    NoCandidates,
    ProviderError,
    ProviderNotInstalled,
    RateLimited,
    RouterError,
    get_provider,
    register_provider,
    supported_providers,
)
from .registry import (
    TIERS,
    ConfigError,
    Endpoint,
    Registry,
    default_registry,
    load_registry,
    reset_default_registry,
)
from .router import (
    Attempt,
    LLMRouter,
    RouteResult,
    aroute,
    configure,
    default_ledger,
    default_sessions,
    reset_state,
    route,
    route_stream,
)

__version__ = "0.1.0"

__all__ = [
    # the two entry points
    "route",
    "LLMRouter",
    # async and streaming variants
    "aroute",
    "route_stream",
    # results
    "RouteResult",
    "Attempt",
    # errors
    "RouterError",
    "RateLimited",
    "ProviderError",
    "AuthError",
    "ProviderNotInstalled",
    "AllCandidatesExhausted",
    "NoCandidates",
    "ConfigError",
    # machinery, for tuning and inspection
    "UsageLedger",
    "Availability",
    "SessionState",
    "Registry",
    "Endpoint",
    "Candidate",
    "TIERS",
    "resolve_candidates",
    "soonest_retry",
    "explain",
    "load_registry",
    "default_registry",
    "reset_default_registry",
    "default_ledger",
    "default_sessions",
    "configure",
    "reset_state",
    # extension points
    "POLICIES",
    "free_first",
    "sticky",
    "get_policy",
    "register_policy",
    "BaseProvider",
    "get_provider",
    "register_provider",
    "supported_providers",
    "__version__",
]
