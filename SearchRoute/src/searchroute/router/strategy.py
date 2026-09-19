"""Provider-ordering strategies.

A strategy answers one question: given the providers that *could* serve this
call, in what order should we try them? It never executes anything — the engine
owns execution — which keeps strategies trivial to write and to test.

Users can pass a built-in name, an instance, or a plain callable
``(candidates, ctx) -> ordered candidates``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from ..errors import ConfigError
from ..providers.base import Provider
from ..quota import QuotaTracker
from ..types import Capability, SearchQuery


@dataclass(slots=True)
class RouteContext:
    """What a strategy gets to reason about."""

    query: SearchQuery
    capability: Capability
    quota: QuotaTracker
    latency: dict[str, float] = field(default_factory=dict)
    """Provider name -> EWMA latency in ms, from previous calls."""
    counter: int = 0
    """Monotonic call count, used for round-robin rotation."""


@runtime_checkable
class Strategy(Protocol):
    name: str

    def order(self, candidates: list[Provider], ctx: RouteContext) -> list[Provider]: ...


class PriorityStrategy:
    """Try providers in the order they were configured.

    The default, because it is predictable: the same query hits the same
    provider every time, which makes cost and behaviour easy to reason about.

    Deliberately a stable no-op. The candidate list already arrives in the right
    order — either the caller's explicit ``providers=[...]`` list (or a per-call
    ``providers=`` pin), or, when auto-discovered, sorted by ``default_priority``
    in ``discover_providers``. Re-sorting by ``default_priority`` here would
    override the caller: ``providers=["tavily", "exa"]`` would silently run Exa
    first, because Exa happens to carry a lower default. Caller order wins.
    """

    name = "priority"

    def order(self, candidates: list[Provider], ctx: RouteContext) -> list[Provider]:
        return list(candidates)


class QuotaAwareStrategy:
    """Prefer whoever has the most headroom left in their window.

    This is what makes several small free tiers behave like one larger one: free
    and unmetered providers go first, then metered ones ranked by the *fraction*
    of their allowance remaining (fraction, not absolute, so a 100/day tier and a
    1,000/month tier are compared fairly). One-time grants sort last — once spent
    they never come back, so they are the reserve tank.
    """

    name = "quota_aware"

    def order(self, candidates: list[Provider], ctx: RouteContext) -> list[Provider]:
        def key(provider: Provider):
            policy = provider.quota
            unmetered = policy is None or not policy.metered
            one_time = bool(policy and not policy.renews and policy.metered)
            remaining = ctx.quota.fraction_remaining(provider.name)
            return (
                one_time,           # False (renewing) sorts first
                not unmetered,      # unmetered first
                -remaining,         # then most headroom
                provider.default_priority,
            )

        return sorted(candidates, key=key)


class QualityStrategy:
    """Best result quality first, spend be damned.

    Uses ``extract_quality`` when the call is an extract, because the strongest
    searcher is usually not the strongest extractor.
    """

    name = "quality"

    def order(self, candidates: list[Provider], ctx: RouteContext) -> list[Provider]:
        extracting = ctx.capability is Capability.EXTRACT

        def key(provider: Provider):
            score = provider.extract_quality if extracting else provider.quality_hint
            # Serving the requested depth natively is itself a quality signal:
            # it avoids a lossy second-hop extraction.
            native = provider.serves_natively(ctx.query.depth)
            return (-score, not native, provider.default_priority)

        return sorted(candidates, key=key)


class LatencyStrategy:
    """Fastest observed provider first. Unmeasured providers are tried
    optimistically so they get a chance to establish a baseline."""

    name = "latency"

    def order(self, candidates: list[Provider], ctx: RouteContext) -> list[Provider]:
        return sorted(
            candidates,
            key=lambda p: (ctx.latency.get(p.name, 0.0), p.default_priority),
        )


class RoundRobinStrategy:
    """Rotate the head of the list on each call.

    Spreads load so several tiers drain evenly rather than burning one to zero
    before touching the next.
    """

    name = "round_robin"

    def order(self, candidates: list[Provider], ctx: RouteContext) -> list[Provider]:
        ordered = sorted(candidates, key=lambda p: p.default_priority)
        if not ordered:
            return ordered
        offset = ctx.counter % len(ordered)
        return ordered[offset:] + ordered[:offset]


@dataclass
class RaceStrategy:
    """Fire the top ``n`` concurrently and take the first success.

    Trades credits for latency. The engine reads ``concurrency`` to decide how
    many to launch; ordering still applies within the race.
    """

    n: int = 2
    name: str = "race"
    inner: Strategy = field(default_factory=LatencyStrategy)

    @property
    def concurrency(self) -> int:
        return max(1, self.n)

    def order(self, candidates: list[Provider], ctx: RouteContext) -> list[Provider]:
        return self.inner.order(candidates, ctx)


@dataclass
class FanoutStrategy:
    """Query the top ``n`` in parallel and fuse all their results.

    The research setting: maximum recall, and cross-provider agreement becomes a
    useful relevance signal via rank fusion. Costs n calls per query.
    """

    n: int = 3
    name: str = "fanout"
    inner: Strategy = field(default_factory=QualityStrategy)

    @property
    def concurrency(self) -> int:
        return max(1, self.n)

    def order(self, candidates: list[Provider], ctx: RouteContext) -> list[Provider]:
        return self.inner.order(candidates, ctx)


_BUILTINS: dict[str, Callable[..., Strategy]] = {
    "priority": PriorityStrategy,
    "quota_aware": QuotaAwareStrategy,
    "quality": QualityStrategy,
    "latency": LatencyStrategy,
    "round_robin": RoundRobinStrategy,
    "race": RaceStrategy,
    "fanout": FanoutStrategy,
}


class _CallableStrategy:
    """Wraps a bare function so users can pass a lambda."""

    name = "custom"

    def __init__(self, fn: Callable[[list[Provider], RouteContext], list[Provider]]):
        self._fn = fn

    def order(self, candidates: list[Provider], ctx: RouteContext) -> list[Provider]:
        return self._fn(candidates, ctx)


def resolve(spec: str | Strategy | Callable | None, **kwargs) -> Strategy:
    """Turn whatever the user passed into a Strategy."""
    if spec is None:
        return PriorityStrategy()
    if isinstance(spec, str):
        factory = _BUILTINS.get(spec)
        if factory is None:
            raise ConfigError(
                f"unknown strategy {spec!r}; available: {', '.join(sorted(_BUILTINS))}"
            )
        return factory(**kwargs) if kwargs else factory()
    if isinstance(spec, Strategy):
        return spec
    if callable(spec):
        return _CallableStrategy(spec)
    raise ConfigError(f"cannot interpret strategy {spec!r}")


def concurrency_of(strategy: Strategy) -> int:
    """How many providers this strategy wants running at once. 1 means the
    engine walks the chain sequentially."""
    return int(getattr(strategy, "concurrency", 1) or 1)


def is_fanout(strategy: Strategy) -> bool:
    """Fan-out fuses every response; race takes only the first winner."""
    return isinstance(strategy, FanoutStrategy)
