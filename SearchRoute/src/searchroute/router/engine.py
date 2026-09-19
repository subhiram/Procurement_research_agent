"""The execution loop: filter, order, execute, fall back, hydrate.

Every call goes through the same pipeline regardless of capability:

1. Filter to providers that could plausibly serve this call.
2. Let the strategy order them.
3. Execute — sequentially, racing, or fanned out.
4. Classify any failure and decide: retry, skip, or disable.
5. Normalize, dedupe, fuse.
6. Hydrate if the achieved depth fell short of what was asked for.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from ..errors import NoProviderAvailable
from ..normalize import apply_domain_filters, canonicalize, dedupe, reciprocal_rank_fusion
from ..providers.base import Provider
from ..quota import QuotaTracker
from ..types import (
    Attempt,
    Capability,
    Depth,
    Document,
    ErrorKind,
    SearchQuery,
    SearchResponse,
    Usage,
)
from .breaker import CircuitBreaker
from .ratelimit import RateGate
from .strategy import RouteContext, Strategy, concurrency_of, is_fanout

Hook = Callable[[str, dict], None]


def _match_request(returned: str, requested: list[str], index: int) -> str:
    """Map a provider's returned URL back to the one we requested.

    Exact match first, then canonical match (which absorbs trailing slashes,
    scheme upgrades and ``www.``), then position — every extract provider is
    contracted to preserve order and never drop a URL, so the index is a sound
    last resort and is what keeps redirects from losing their content.
    """
    if returned in requested:
        return returned
    target = canonicalize(returned)
    for url in requested:
        if canonicalize(url) == target:
            return url
    return requested[index] if index < len(requested) else returned


@dataclass(slots=True)
class _Outcome:
    """One provider call's result, success or failure."""

    provider: Provider
    attempt: Attempt
    response: SearchResponse | None = None
    error: BaseException | None = None


@dataclass
class Engine:
    providers: dict[str, Provider]
    strategy: Strategy
    quota: QuotaTracker
    breaker: CircuitBreaker = field(default_factory=CircuitBreaker)
    rate: RateGate = field(default_factory=RateGate)
    max_retries: int = 1
    hooks: list[Hook] = field(default_factory=list)
    _counter: int = 0

    # ---- hooks -----------------------------------------------------------

    def _emit(self, event: str, payload: dict) -> None:
        for hook in self.hooks:
            try:
                hook(event, payload)
            except Exception:
                # An observability hook must never break the search it observes.
                pass

    # ---- candidate selection --------------------------------------------

    def candidates(
        self,
        capability: Capability,
        query: SearchQuery,
        *,
        only: Sequence[str] | None = None,
        exclude: Sequence[str] | None = None,
    ) -> list[Provider]:
        """Providers that could serve this call right now.

        Note what is *not* filtered here: depth. A provider that cannot serve the
        requested depth natively stays eligible, because the hydration stage can
        make up the difference. Filtering on depth would throw away the whole
        point of composing search + extract.
        """
        exclude = set(exclude or ())
        pool = list(self.providers.values())

        if only:
            missing = [n for n in only if n not in self.providers]
            if missing:
                raise NoProviderAvailable(
                    f"requested provider(s) not configured: {', '.join(missing)}"
                )
            pool = [self.providers[n] for n in only]

        out = []
        for provider in pool:
            if provider.name in exclude:
                continue
            # can_serve, not supports: a provider may implement a capability it
            # lacks the credentials for (keyless Jina can extract but not search).
            if not provider.can_serve(capability):
                continue
            if not self.breaker.allows(provider.name):
                continue
            if not self.quota.can_afford(provider.name, provider.cost_of(query)):
                continue
            out.append(provider)
        return out

    def _ordered(
        self, candidates: list[Provider], capability: Capability, query: SearchQuery
    ) -> list[Provider]:
        self._counter += 1
        ctx = RouteContext(
            query=query,
            capability=capability,
            quota=self.quota,
            latency=self.breaker.latencies(),
            counter=self._counter,
        )
        ordered = list(self.strategy.order(candidates, ctx))

        # Stable partition: last-resort providers go to the back no matter what
        # the strategy decided. A quota-aware ordering would otherwise rank the
        # unofficial free backends first precisely because they cost nothing.
        #
        # This only ever *reorders*. Deciding which providers are eligible is
        # already done by construction and by the only=/exclude= filters, so
        # dropping one here would silently discard a fallback the caller asked
        # for — `providers=["tavily", "duckduckgo"]` means exactly "try Tavily,
        # then DuckDuckGo".
        preferred = [p for p in ordered if not p.last_resort]
        fallback = [p for p in ordered if p.last_resort]
        return preferred + fallback

    # ---- single attempt --------------------------------------------------

    async def _call_one(self, provider: Provider, query: SearchQuery) -> _Outcome:
        """One provider, with retries for the failures worth retrying."""
        attempt = Attempt(provider=provider.name, capability=query.capability, ok=False)
        started = time.perf_counter()
        last_error: BaseException | None = None

        for retry in range(self.max_retries + 1):
            try:
                # Pace before dispatch, not after: the interval is a promise to
                # the provider about how often we knock, so it has to gate the
                # request itself — including retries, which are extra knocks.
                await self.rate.acquire(provider.name, provider.min_interval)
                response = await provider.search(query)
            except Exception as exc:  # noqa: BLE001 - classified immediately below
                last_error = exc
                kind = provider.classify_error(exc)
                attempt.error_kind = kind

                if kind is ErrorKind.RATE_LIMIT and retry < self.max_retries:
                    await self._backoff(exc, retry)
                    continue
                if kind is ErrorKind.TRANSIENT and retry < self.max_retries:
                    await self._backoff(exc, retry)
                    continue
                break
            else:
                latency = (time.perf_counter() - started) * 1000
                attempt.ok = True
                attempt.latency_ms = latency
                attempt.n_results = len(response.results)
                attempt.cost = response.cost.by_provider.get(provider.name, 0)

                self.breaker.record_success(provider.name, latency)
                self.quota.debit(provider.name, attempt.cost)
                self._emit("success", {"provider": provider.name, "latency_ms": latency})
                return _Outcome(provider=provider, attempt=attempt, response=response)

        # Every retry exhausted.
        attempt.latency_ms = (time.perf_counter() - started) * 1000
        attempt.error = str(last_error)
        self._record_failure(provider.name, attempt.error_kind)
        self._emit(
            "failure",
            {"provider": provider.name, "kind": attempt.error_kind, "error": attempt.error},
        )
        return _Outcome(provider=provider, attempt=attempt, error=last_error)

    def _record_failure(self, provider: str, kind: ErrorKind | None) -> None:
        """Translate a failure into the right recovery model.

        These are genuinely different situations and conflating them is how
        fallback chains go wrong: a drained tier should be skipped until its
        window resets, a bad key should be skipped for the whole process, and a
        flaky endpoint should be retried later.
        """
        if kind is ErrorKind.QUOTA:
            self.quota.mark_exhausted(provider)
        elif kind is ErrorKind.AUTH:
            # A wrong key will not fix itself; stop paying latency for it.
            self.breaker.trip(provider)
        elif kind is ErrorKind.RATE_LIMIT:
            self.quota.disable_until(
                provider, datetime.now(timezone.utc) + timedelta(seconds=60)
            )
        else:
            self.breaker.record_failure(provider)

    @staticmethod
    async def _backoff(exc: BaseException, retry: int) -> None:
        retry_after = getattr(exc, "retry_after", None)
        # Jitter matters: without it, concurrent workers retry in lockstep and
        # re-trigger the same rate limit.
        delay = retry_after if retry_after else (0.5 * (2**retry)) + random.uniform(0, 0.3)
        await asyncio.sleep(min(delay, 10.0))

    # ---- search ----------------------------------------------------------

    async def search(
        self,
        query: SearchQuery,
        *,
        only: Sequence[str] | None = None,
        exclude: Sequence[str] | None = None,
    ) -> SearchResponse:
        capability = query.capability
        candidates = self.candidates(capability, query, only=only, exclude=exclude)
        if not candidates:
            raise NoProviderAvailable(
                self._why_empty(capability, query, only=only, exclude=exclude)
            )

        ordered = self._ordered(candidates, capability, query)
        width = concurrency_of(self.strategy)

        if width > 1 and is_fanout(self.strategy):
            response = await self._run_fanout(ordered[:width], query)
        elif width > 1:
            response = await self._run_race(ordered[:width], query, ordered[width:])
        else:
            response = await self._run_sequential(ordered, query)

        response.results = apply_domain_filters(
            response.results, query.include_domains, query.exclude_domains
        )
        response.results = response.results[: query.max_results]
        for position, result in enumerate(response.results):
            result.rank = position
        response.requested_depth = query.depth
        return response

    async def _run_sequential(
        self, ordered: list[Provider], query: SearchQuery
    ) -> SearchResponse:
        attempts: list[Attempt] = []
        for provider in ordered:
            outcome = await self._call_one(provider, query)
            attempts.append(outcome.attempt)
            if outcome.response is not None and outcome.response.results:
                response = outcome.response
                response.attempts = attempts
                response.results = dedupe(response.results)
                return response
            # A success with zero results is not a failure — but it is also not
            # an answer, so we keep walking the chain.
        raise NoProviderAvailable("all providers failed or returned nothing", attempts)

    async def _run_race(
        self, racers: list[Provider], query: SearchQuery, rest: list[Provider]
    ) -> SearchResponse:
        """First success wins; the losers are cancelled to stop their spend."""
        attempts: list[Attempt] = []
        tasks = {
            asyncio.create_task(self._call_one(p, query)): p for p in racers
        }
        winner: SearchResponse | None = None
        try:
            for future in asyncio.as_completed(list(tasks)):
                outcome = await future
                attempts.append(outcome.attempt)
                if outcome.response is not None and outcome.response.results:
                    winner = outcome.response
                    break
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

        if winner is not None:
            winner.attempts = attempts
            winner.results = dedupe(winner.results)
            return winner
        if rest:
            fallback = await self._run_sequential(rest, query)
            fallback.attempts = attempts + fallback.attempts
            return fallback
        raise NoProviderAvailable("all raced providers failed", attempts)

    async def _run_fanout(
        self, group: list[Provider], query: SearchQuery
    ) -> SearchResponse:
        """Query several providers and fuse everything they return."""
        outcomes = await asyncio.gather(
            *(self._call_one(p, query) for p in group), return_exceptions=False
        )
        attempts = [o.attempt for o in outcomes]
        successful = [o for o in outcomes if o.response is not None and o.response.results]
        if not successful:
            raise NoProviderAvailable("all fanned-out providers failed", attempts)

        fused = reciprocal_rank_fusion([o.response.results for o in successful])  # type: ignore[union-attr]
        cost = Usage()
        for outcome in successful:
            for name, amount in outcome.response.cost.by_provider.items():  # type: ignore[union-attr]
                cost.add(name, amount)

        achieved = min(
            (o.response.depth for o in successful),  # type: ignore[union-attr]
            default=Depth.SNIPPETS,
        )
        return SearchResponse(
            query=query.query,
            results=fused,
            depth=achieved,
            requested_depth=query.depth,
            providers_used=[o.provider.name for o in successful],
            attempts=attempts,
            cost=cost,
        )

    # ---- extract ---------------------------------------------------------

    async def extract(
        self,
        urls: list[str],
        *,
        only: Sequence[str] | None = None,
        exclude: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Extract with fallback, per URL.

        URLs that one provider fails on are retried against the next provider
        rather than the whole batch being retried — a single paywalled page
        should not cost the other nine their content.
        """
        probe = SearchQuery(query="", depth=Depth.CONTENT, capability=Capability.EXTRACT)
        candidates = self.candidates(Capability.EXTRACT, probe, only=only, exclude=exclude)
        if not candidates:
            raise NoProviderAvailable("no provider configured for extract")

        ctx = RouteContext(query=probe, capability=Capability.EXTRACT, quota=self.quota)
        ordered = self.strategy.order(candidates, ctx)

        results: dict[str, Document] = {
            url: Document(url=url, ok=False, error="not attempted") for url in urls
        }
        pending = list(urls)

        for provider in ordered:
            if not pending:
                break
            cost = provider.extract_cost_of(pending)
            if not self.quota.can_afford(provider.name, cost):
                continue
            try:
                await self.rate.acquire(provider.name, provider.min_interval)
                docs = await provider.extract(pending, **kwargs)
            except Exception as exc:  # noqa: BLE001
                self._record_failure(provider.name, provider.classify_error(exc))
                self._emit("extract_failure", {"provider": provider.name, "error": str(exc)})
                continue

            self.breaker.record_success(provider.name)
            still_pending: list[str] = []
            succeeded: list[str] = []

            # Match each returned document back to the URL we *asked* for.
            #
            # Providers echo the URL they resolved to, which is routinely not
            # the string we sent: a trailing slash added, http upgraded to
            # https, a redirect followed. Keying results by `doc.url` would
            # file the content under an address the caller never asked about,
            # and hand them back the "not attempted" placeholder instead — a
            # silent content loss on any redirect.
            for index, doc in enumerate(docs):
                requested = _match_request(doc.url, pending, index)
                if doc.ok and doc.content:
                    results[requested] = doc
                    succeeded.append(requested)
                else:
                    still_pending.append(requested)
                    # Keep the most informative failure to report back.
                    if results[requested].error == "not attempted":
                        results[requested] = doc

            if succeeded:
                self.quota.debit(provider.name, provider.extract_cost_of(succeeded))
            pending = still_pending

        return [results[url] for url in urls]

    # ---- diagnostics -----------------------------------------------------

    def _why_empty(
        self,
        capability: Capability,
        query: SearchQuery,
        *,
        only: Sequence[str] | None,
        exclude: Sequence[str] | None,
    ) -> str:
        """Explain which filter removed every provider.

        "No provider available" on its own is a miserable thing to debug, so we
        say whether it was keys, quota or breakers.
        """
        if not self.providers:
            return (
                "no providers configured — set an API key (e.g. TAVILY_API_KEY, "
                "EXA_API_KEY) or install 'searchroute[ddg]' for keyless fallback"
            )
        reasons = []
        for provider in self.providers.values():
            if only and provider.name not in only:
                continue
            if exclude and provider.name in exclude:
                continue
            if not provider.supports(capability):
                reasons.append(f"{provider.name}: does not support {capability.value}")
            elif capability in provider.keyed_capabilities and not provider.api_key:
                reasons.append(
                    f"{provider.name}: {capability.value} needs an API key "
                    f"(other capabilities work without one)"
                )
            elif not provider.configured:
                reasons.append(f"{provider.name}: no API key configured")
            elif not self.breaker.allows(provider.name):
                reasons.append(f"{provider.name}: circuit open after repeated failures")
            elif not self.quota.can_afford(provider.name, provider.cost_of(query)):
                remaining = self.quota.remaining(provider.name)
                reasons.append(f"{provider.name}: quota exhausted (remaining={remaining})")
        return "no provider available for this call — " + "; ".join(reasons)

    async def aclose(self) -> None:
        await asyncio.gather(
            *(p.aclose() for p in self.providers.values()), return_exceptions=True
        )
