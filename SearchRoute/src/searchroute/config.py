"""Configuration and environment discovery.

The zero-config path matters most: construct ``SearchRoute()`` with nothing, and
whichever providers have keys in the environment light up. Adding a key to
``.env`` is the whole integration story for a new provider.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from .errors import ConfigError
from .types import Depth

#: Environment variables that supply each provider's credential. The first
#: present name wins, so both vendor-conventional and namespaced forms work.
ENV_KEYS: dict[str, tuple[str, ...]] = {
    "exa": ("EXA_API_KEY", "SEARCHROUTE_EXA_API_KEY"),
    "tavily": ("TAVILY_API_KEY", "SEARCHROUTE_TAVILY_API_KEY"),
    "firecrawl": ("FIRECRAWL_API_KEY", "SEARCHROUTE_FIRECRAWL_API_KEY"),
    "serpapi": ("SERPAPI_API_KEY", "SERPAPI_KEY"),
    "serper": ("SERPER_API_KEY", "SERPER_KEY"),
    "searchapi": ("SEARCHAPI_API_KEY", "SEARCHAPI_KEY"),
    "google_cse": ("GOOGLE_API_KEY", "GOOGLE_SEARCH_API_KEY"),
    "jina": ("JINA_API_KEY",),
    "brave": ("BRAVE_API_KEY", "BRAVE_SEARCH_API_KEY"),
}

#: Extra non-credential settings a provider needs before it can run.
ENV_OPTIONS: dict[str, dict[str, tuple[str, ...]]] = {
    "google_cse": {"cx": ("GOOGLE_CSE_ID", "GOOGLE_CX")},
    "searxng": {"base_url": ("SEARXNG_URL", "SEARXNG_BASE_URL")},
}

def discover_key(provider: str, env: dict[str, str] | None = None) -> str | None:
    env = env if env is not None else dict(os.environ)
    for name in ENV_KEYS.get(provider, ()):
        value = env.get(name)
        if value:
            return value.strip()
    return None


def discover_options(provider: str, env: dict[str, str] | None = None) -> dict[str, Any]:
    env = env if env is not None else dict(os.environ)
    options: dict[str, Any] = {}
    for option, names in ENV_OPTIONS.get(provider, {}).items():
        for name in names:
            value = env.get(name)
            if value:
                options[option] = value.strip()
                break
    return options


#: Settings a provider cannot run without, beyond a key. Google CSE needs its
#: search-engine id; SearXNG needs an instance to talk to.
REQUIRED_OPTIONS: dict[str, str] = {
    "google_cse": "cx",
    "searxng": "base_url",
}

#: Providers whose constructors accept a ``contact`` address for their polite
#: pools. Passing it to a provider that doesn't take it would be a TypeError.
CONTACT_AWARE = ("crossref", "pubmed", "wikipedia", "arxiv")


def discover_contact(env: dict[str, str] | None = None) -> str | None:
    env = env if env is not None else dict(os.environ)
    value = env.get("SEARCHROUTE_CONTACT")
    return value.strip() if value else None


def discover_providers(env: dict[str, str] | None = None) -> list[str]:
    """Provider names usable right now, ordered by default priority.

    A provider qualifies when it has whatever it needs to make a call: a key if
    it requires one, plus any mandatory option. Providers marked ``opt_in_only``
    are skipped even when fully configured — a key in the environment is not
    consent to spend money.
    """
    env = env if env is not None else dict(os.environ)
    from .providers import registry

    qualified: list[tuple[int, str]] = []
    for name, cls in registry.all_classes().items():
        if cls.opt_in_only:
            continue
        if cls.requires_key and not discover_key(name, env):
            continue
        required = REQUIRED_OPTIONS.get(name)
        if required and not discover_options(name, env).get(required):
            continue
        qualified.append((cls.default_priority, name))

    qualified.sort()
    return [name for _, name in qualified]


@dataclass(slots=True)
class Settings:
    """Resolved client configuration."""

    providers: list[str] = field(default_factory=list)
    """Explicit order. Empty means auto-discover."""
    exclude: list[str] = field(default_factory=list)
    strategy: str = "priority"
    depth: Depth = Depth.SNIPPETS
    max_results: int = 10
    max_hydrate: int = 10
    hydrate_concurrency: int = 5
    hydrate_providers: list[str] = field(default_factory=list)
    timeout: float = 20.0
    max_retries: int = 1
    reserve_pct: float = 0.05
    quota_store: Any = None
    api_keys: dict[str, str] = field(default_factory=dict)
    provider_options: dict[str, dict[str, Any]] = field(default_factory=dict)
    contact: str | None = None
    """Contact address sent to the free scholarly APIs.

    Crossref, PubMed and Wikipedia all ask for one, and Crossref routes
    identified traffic to a better-served "polite pool". Costs nothing and is
    what their terms request. Discovered from ``SEARCHROUTE_CONTACT``.
    """

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Settings:
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        unknown = set(data) - known
        if unknown:
            raise ConfigError(f"unknown settings: {', '.join(sorted(unknown))}")
        payload = dict(data)
        if "depth" in payload:
            payload["depth"] = coerce_depth(payload["depth"])
        return cls(**payload)


def coerce_depth(value: Depth | str | int) -> Depth:
    """Accept ``Depth.CONTENT``, ``"content"`` or ``3`` — callers shouldn't have
    to import an enum for the common case."""
    if isinstance(value, Depth):
        return value
    if isinstance(value, int):
        return Depth(value)
    try:
        return Depth[str(value).strip().upper()]
    except KeyError:
        valid = ", ".join(d.name.lower() for d in Depth)
        raise ConfigError(f"unknown depth {value!r}; expected one of: {valid}") from None


def load_yaml(path: str) -> Settings:
    """Load settings from a YAML file. Requires ``searchroute[yaml]``."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - depends on extras
        raise ConfigError("PyYAML is required: pip install 'searchroute[yaml]'") from exc
    with open(path) as fh:
        return Settings.from_dict(yaml.safe_load(fh) or {})
