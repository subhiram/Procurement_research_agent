"""The public surface.

``AsyncSearchRoute`` is the real implementation; ``SearchRoute`` is a thin sync
facade over it, because agent code is split roughly evenly between the two and
neither should feel second-class.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable, Sequence
from typing import Any

import httpx

from . import profiles as _profiles
from .config import (
    CONTACT_AWARE,
    Settings,
    coerce_depth,
    discover_contact,
    discover_key,
    discover_options,
    discover_providers,
)
from .errors import ConfigError
from .ledger import make_store
from .observability import StatsCollector
from .providers import registry
from .providers.base import Provider
from .quota import QuotaTracker
from .router.breaker import CircuitBreaker
from .router.engine import Engine
from .router.hydrate import hydrate
from .router.strategy import resolve as resolve_strategy
from .types import (
    Answer,
    Capability,
    Depth,
    Document,
    SearchQuery,
    SearchResponse,
)


def _build_providers(
    settings: Settings,
    extra: Sequence[Provider] | None,
    client: httpx.AsyncClient | None,
) -> dict[str, Provider]:
    """Instantiate the configured providers, in the caller's order.

    Auto-discovery only kicks in when the caller named nothing at all. Handing us
    explicit providers means those are the ones you want — quietly adding
    whatever else happened to have a key in the environment would make behaviour
    depend on the shell.
    """
    if settings.providers:
        names = list(settings.providers)
    elif extra:
        names = []
    else:
        names = discover_providers()
    names = [n for n in names if n not in set(settings.exclude)]

    contact = settings.contact or discover_contact()

    built: dict[str, Provider] = {}
    for name in names:
        cls = registry.get(name)
        key = settings.api_keys.get(name) or discover_key(name)
        options = {**discover_options(name), **settings.provider_options.get(name, {})}
        # Only the scholarly providers take a contact address; passing it to the
        # rest would be a TypeError.
        if contact and name in CONTACT_AWARE:
            options.setdefault("contact", contact)
        provider = cls(api_key=key, timeout=settings.timeout, client=client, **options)
        if not provider.configured:
            # Explicitly requested but unusable: that's a config error worth
            # naming, not a silent omission that shows up later as "no results".
            if settings.providers:
                raise ConfigError(
                    f"provider {name!r} was requested but has no API key configured"
                )
            continue
        built[name] = provider

    for provider in extra or ():
        built[provider.name] = provider
    return built


class AsyncSearchRoute:
    """Route search, extract and answer calls across many providers."""

    def __init__(
        self,
        providers: Sequence[str] | None = None,
        *,
        profile: str | None = None,
        strategy: Any = None,
        depth: Depth | str | int | None = None,
        max_results: int | None = None,
        max_hydrate: int | None = None,
        exclude: Sequence[str] | None = None,
        api_keys: dict[str, str] | None = None,
        provider_options: dict[str, dict[str, Any]] | None = None,
        quota_store: Any = None,
        reserve_pct: float | None = None,
        contact: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        hooks: Sequence[Callable[[str, dict], None]] | None = None,
        custom_providers: Sequence[Provider] | None = None,
        http_client: httpx.AsyncClient | None = None,
        settings: Settings | None = None,
    ):
        base: dict[str, Any] = _profiles.get(profile) if profile else {}
        # Explicit kwargs beat the profile; the profile beats the defaults.
        overrides = {
            "providers": list(providers) if providers else None,
            "exclude": list(exclude) if exclude else None,
            "strategy": strategy if isinstance(strategy, str) else None,
            "depth": depth,
            "max_results": max_results,
            "max_hydrate": max_hydrate,
            "api_keys": api_keys,
            "provider_options": provider_options,
            "quota_store": quota_store,
            "reserve_pct": reserve_pct,
            "contact": contact,
            "timeout": timeout,
            "max_retries": max_retries,
        }
        base.update({k: v for k, v in overrides.items() if v is not None})
        if "depth" in base:
            base["depth"] = coerce_depth(base["depth"])

        self.settings = settings or Settings.from_dict(base)

        self._providers = _build_providers(self.settings, custom_providers, http_client)

        store = make_store(self.settings.quota_store)
        self.quota = QuotaTracker(store=store, reserve_pct=self.settings.reserve_pct)
        for name, provider in self._providers.items():
            self.quota.register(name, provider.quota)

        strategy_obj = (
            resolve_strategy(strategy)
            if strategy is not None and not isinstance(strategy, str)
            else resolve_strategy(self.settings.strategy)
        )
        # Always collect stats. It is a few counters per provider, and the
        # alternative is that "why is this slow / failing" needs a code change
        # and a redeploy to answer.
        self._stats = StatsCollector()
        self.engine = Engine(
            providers=self._providers,
            strategy=strategy_obj,
            quota=self.quota,
            breaker=CircuitBreaker(),
            max_retries=self.settings.max_retries,
            hooks=[*(hooks or ()), self._stats],
        )

    # ---- introspection ---------------------------------------------------

    @property
    def providers(self) -> list[str]:
        return list(self._providers)

    def status(self) -> dict[str, Any]:
        """Live view of quota and provider health — what the smoke script prints
        and what an app should log when a search degrades."""
        return {
            "providers": self.providers,
            "strategy": getattr(self.engine.strategy, "name", "custom"),
            "quota": self.quota.snapshot(),
            "breakers": self.engine.breaker.snapshot(),
        }

    def stats(self) -> dict[str, dict[str, Any]]:
        """Per-provider call counts, success rate, latency and error breakdown
        since this client was constructed.

        Distinct from ``status()``: that is the *current* state (what's left,
        what's open), this is the *history* (what happened).
        """
        return self._stats.snapshot()

    # ---- request building ------------------------------------------------

    def _query(self, query: str, capability: Capability, kwargs: dict[str, Any]) -> SearchQuery:
        depth = kwargs.pop("depth", None)
        return SearchQuery(
            query=query,
            max_results=kwargs.pop("max_results", None) or self.settings.max_results,
            depth=coerce_depth(depth) if depth is not None else self.settings.depth,
            capability=capability,
            include_domains=list(kwargs.pop("include_domains", None) or []),
            exclude_domains=list(kwargs.pop("exclude_domains", None) or []),
            start_date=kwargs.pop("start_date", None),
            end_date=kwargs.pop("end_date", None),
            lang=kwargs.pop("lang", None),
            region=kwargs.pop("region", None),
            extra=kwargs.pop("extra", None) or {},
        )

    def _with_strategy(self, spec: Any, **kwargs) -> Engine:
        """A per-call strategy override, without disturbing the client's default."""
        if spec is None:
            return self.engine
        engine = Engine(
            providers=self.engine.providers,
            strategy=resolve_strategy(spec, **kwargs),
            quota=self.quota,
            breaker=self.engine.breaker,  # share health state across calls
            rate=self.engine.rate,  # and pacing — a per-call strategy override
            # must not reset how often we knock on a rate-limited provider
            max_retries=self.engine.max_retries,
            hooks=self.engine.hooks,
        )
        engine._counter = self.engine._counter
        return engine

    # ---- operations ------------------------------------------------------

    async def search(
        self,
        query: str,
        *,
        depth: Depth | str | int | None = None,
        max_results: int | None = None,
        strategy: Any = None,
        n: int | None = None,
        providers: Sequence[str] | None = None,
        exclude: Sequence[str] | None = None,
        capability: Capability = Capability.SEARCH,
        max_hydrate: int | None = None,
        hydrate_providers: Sequence[str] | None = None,
        include_domains: Sequence[str] | None = None,
        exclude_domains: Sequence[str] | None = None,
        start_date: Any = None,
        end_date: Any = None,
        lang: str | None = None,
        region: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> SearchResponse:
        """Search, falling back across providers, then fill the depth gap.

        Every parameter is spelled out rather than swallowed by ``**kwargs``.
        That makes them discoverable to tooling, and — more importantly — turns
        a typo into a ``TypeError`` instead of a silently ignored argument:
        ``max_reslts=5`` should fail loudly, not quietly return the default.
        """
        request = self._query(
            query,
            capability,
            {
                "depth": depth,
                "max_results": max_results,
                "include_domains": include_domains,
                "exclude_domains": exclude_domains,
                "start_date": start_date,
                "end_date": end_date,
                "lang": lang,
                "region": region,
                "extra": extra,
            },
        )
        engine = self._with_strategy(strategy, **({"n": n} if n else {}))

        response = await engine.search(request, only=providers, exclude=exclude)

        budget = max_hydrate if max_hydrate is not None else self.settings.max_hydrate
        return await hydrate(
            response,
            request.depth,
            extractor=engine.extract,
            max_hydrate=budget,
            concurrency=self.settings.hydrate_concurrency,
            only=list(hydrate_providers or self.settings.hydrate_providers) or None,
        )

    async def extract(
        self,
        urls: Sequence[str],
        *,
        providers: Sequence[str] | None = None,
        exclude: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Fetch and clean pages, falling back per URL across the extract chain."""
        return await self.engine.extract(
            list(urls), only=providers, exclude=exclude, **kwargs
        )

    async def answer(
        self,
        query: str,
        *,
        providers: Sequence[str] | None = None,
        max_results: int | None = None,
        depth: Depth | str | int | None = None,
        lang: str | None = None,
        region: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Answer:
        """A provider-generated direct answer.

        Falls back to a plain search when no ANSWER-capable provider is
        available — the results come back with ``answer=None`` rather than the
        library writing one, which it has no model to do.
        """
        options = {
            "max_results": max_results,
            "depth": depth,
            "lang": lang,
            "region": region,
            "extra": extra,
        }
        request = self._query(query, Capability.ANSWER, dict(options))
        candidates = self.engine.candidates(Capability.ANSWER, request, only=providers)
        for provider in self.engine._ordered(candidates, Capability.ANSWER, request):
            try:
                result = await provider.answer(request)
            except Exception as exc:  # noqa: BLE001
                self.engine._record_failure(provider.name, provider.classify_error(exc))
                continue
            self.quota.debit(provider.name, result.cost.by_provider.get(provider.name, 0))
            return result

        search_request = self._query(query, Capability.SEARCH, dict(options))
        fallback = await self.engine.search(search_request, only=providers)
        return Answer(
            query=query,
            answer=None,
            results=fallback.results,
            provider=",".join(fallback.providers_used),
            attempts=fallback.attempts,
            cost=fallback.cost,
        )

    async def aclose(self) -> None:
        await self.engine.aclose()

    async def __aenter__(self) -> AsyncSearchRoute:
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()


class SearchRoute:
    """Synchronous facade.

    Runs the async client on a dedicated background event loop, so it works
    unchanged inside a notebook, a Flask handler, or a plain script — places
    where ``asyncio.run`` would fail because a loop is already running.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="searchroute"
        )
        self._thread.start()
        self._async = AsyncSearchRoute(*args, **kwargs)

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _call(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    @property
    def providers(self) -> list[str]:
        return self._async.providers

    @property
    def settings(self) -> Settings:
        return self._async.settings

    def status(self) -> dict[str, Any]:
        return self._async.status()

    def stats(self) -> dict[str, dict[str, Any]]:
        return self._async.stats()

    def search(self, query: str, **kwargs: Any) -> SearchResponse:
        return self._call(self._async.search(query, **kwargs))

    def extract(self, urls: Sequence[str], **kwargs: Any) -> list[Document]:
        return self._call(self._async.extract(urls, **kwargs))

    def answer(self, query: str, **kwargs: Any) -> Answer:
        return self._call(self._async.answer(query, **kwargs))

    def close(self) -> None:
        try:
            self._call(self._async.aclose())
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=5)

    def __enter__(self) -> SearchRoute:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
