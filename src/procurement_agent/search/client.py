"""Search, extraction and the per-run credit budget.

Provider selection, fallback, quota tracking and circuit breaking all belong to
SearchRoute. What stays here is the part SearchRoute deliberately does not do:
cap what a *single run* may spend.

Those are different budgets and both are needed. SearchRoute's ledger is a
monthly, per-provider balance shared by everything on the machine. `SessionBudget`
is a per-run ceiling, so one wide fan-out cannot quietly consume the month in a
single request. Without it a single bad query set is indistinguishable from
normal use until the month is gone.

The budget is enforced here rather than in the nodes because nodes fan out
concurrently: a check-then-spend in each branch would race. One lock makes the
ceiling real.
"""

from __future__ import annotations

import asyncio
import logging
from contextvars import ContextVar
from typing import Any

from procurement_agent.config import Settings, get_settings
from procurement_agent.trace import RunTrace, search_hook
from searchroute import (
    AsyncSearchRoute,
    Capability,
    Depth,
    SearchRouteError,
    discover_providers,
)

log = logging.getLogger(__name__)

#: The run whose trace should receive search events. A ContextVar rather than a
#: parameter because SearchRoute's hooks are fixed at client construction while
#: the client outlives any single run - and because contextvars follow asyncio
#: tasks, so the concurrent fan-out branches each stay attributed correctly.
_active_trace: ContextVar[RunTrace | None] = ContextVar("active_trace", default=None)


def bind_trace(trace: RunTrace | None) -> None:
    """Attribute subsequent search events in this task to `trace`."""
    _active_trace.set(trace)


def _record(kind: str, **fields: Any) -> None:
    trace = _active_trace.get()
    if trace is not None:
        trace.add(kind, **fields)


class BudgetExhausted(RuntimeError):
    """The run's search credit budget is spent.

    Not an error so much as a stop signal: callers should finish with what they
    have and mark the run truncated.
    """


_client: AsyncSearchRoute | None = None
_client_lock = asyncio.Lock()


async def get_client(settings: Settings | None = None) -> AsyncSearchRoute:
    """The process-wide search client.

    One instance for the process lifetime, deliberately: the quota ledger,
    circuit breakers, rate gate and HTTP connection pool all live on it, and a
    per-call client would reset the breakers and lose the pool on every node.

    The trace hook is installed here rather than per call because SearchRoute
    takes its hooks at construction. It dispatches to whichever run is currently
    executing, so one long-lived client still produces per-run records.
    """
    global _client
    if _client is not None:
        return _client
    async with _client_lock:
        if _client is None:
            settings = settings or get_settings()
            available = discover_providers()
            log.info("search providers discovered: %s", ", ".join(available) or "none")
            _client = AsyncSearchRoute(
                # quota_aware rather than the default priority order: on free
                # tiers the right provider is whichever still has budget, and a
                # fixed order is only correct until the first one runs dry.
                strategy="quota_aware",
                max_results=settings.results_per_query,
                hooks=[_dispatch_hook],
            )
    return _client


def _dispatch_hook(event: str, payload: dict) -> None:
    """Forward a SearchRoute provider event to the active run's trace."""
    trace = _active_trace.get()
    if trace is not None:
        search_hook(trace)(event, payload)


async def close_client() -> None:
    """Release the HTTP pool. Safe to call when no client was ever built."""
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


class SessionBudget:
    """A ceiling on what one run may spend, honoured across concurrent branches.

    One of these exists per run, created in `graph.build.run_config()` and
    carried in the run config rather than in state, because it holds a lock and
    a lock cannot be serialised into a checkpoint.

    Nodes take a `sub()` allocation instead of constructing their own. That
    distinction is load-bearing: separate budgets per node are separate pools,
    and the run-wide ceiling silently stops existing - which is exactly what
    happened here, with five private pools letting a run spend far more than the
    configured maximum while reporting the maximum.

    An allocation is a **spending limit against the run's pool, not a transfer
    out of it**. Taking the credits up front looks equivalent and is not: an
    allocation that goes unused - the academic pass on a material with no
    literature - would keep them, and the vendor search that the run exists to
    perform would find the pool drained by nodes that never spent anything.
    Measured: four allocations totalling 12 removed 12 of 25 credits from a run
    that made five charged searches.
    """

    def __init__(
        self,
        credits: int | None = None,
        settings: Settings | None = None,
        *,
        parent: SessionBudget | None = None,
        name: str = "run",
        limit: int | None = None,
    ) -> None:
        settings = settings or get_settings()
        self._pool = (
            credits if credits is not None else settings.search_credits_per_session
        )
        self._lock = asyncio.Lock()
        self._parent = parent
        self.name = name
        #: How much of the parent this allocation may spend. None on the run
        #: budget itself, which is bounded by its own pool.
        self._limit = limit
        self._spent = 0
        self.truncated = False
        #: Allocations already made, by name. See `sub()`.
        self._children: dict[str, SessionBudget] = {}
        #: Pages retrieved by the local crawler - i.e. credits not spent.
        self.crawled_free = 0

    @property
    def credits_remaining(self) -> int:
        """What this budget may still spend.

        For an allocation that is the smaller of its own limit and whatever the
        run has left, so a generous allocation cannot promise credits the run no
        longer has.
        """
        if self._parent is None:
            return self._pool
        return max(0, min(self._limit - self._spent, self._parent.credits_remaining))

    async def sub(self, credits: int, name: str) -> SessionBudget:
        """A named spending limit against this budget.

        **Memoised by name**, which is what makes a fan-out share one limit:
        `contact_extraction` runs once per candidate, and without this each of
        the twelve branches would get its own - reproducing, inside one node,
        exactly the private-pool bug this class was rewritten to fix.

        Nothing is deducted here. The limit only caps what the node may spend;
        the credits stay in the run's pool until they are actually used.
        """
        async with self._lock:
            existing = self._children.get(name)
            if existing is not None:
                return existing
            child = SessionBudget(parent=self, name=name, limit=credits)
            self._children[name] = child
            return child

    async def _reserve(self, cost: int = 1) -> None:
        """Take `cost` credits before the call, atomically.

        Deducted up front rather than after the fact, and that ordering is the
        whole point: a check now and a deduction later is a check-then-act race,
        and concurrent branches would all pass the check before any of them
        deducted. Reserving under the lock makes the ceiling real no matter how
        wide the fan-out.

        An allocation charges its parent too, so the run-wide ceiling holds
        whatever the individual limits add up to.
        """
        if self.credits_remaining < cost:
            self.truncated = True
            if self._parent is not None:
                self._parent.truncated = True
            raise BudgetExhausted(
                f"search budget exhausted for {self.name} "
                f"({self.credits_remaining} left, {cost} needed)"
            )
        if self._parent is None:
            async with self._lock:
                self._pool -= cost
            return
        # Charge the run first: if it refuses, nothing was spent anywhere.
        await self._parent._reserve(cost)
        async with self._lock:
            self._spent += cost

    async def _settle(self, reserved: int, provider_cost: int) -> None:
        """Correct the reservation once the real cost is known."""
        difference = self.charge_for(provider_cost) - reserved
        if not difference:
            return
        if self._parent is not None:
            await self._parent._settle(reserved, provider_cost)
            async with self._lock:
                self._spent = max(0, self._spent + difference)
            return
        async with self._lock:
            self._pool = max(0, self._pool - difference)

    @staticmethod
    def charge_for(provider_cost: int) -> int:
        """What one call costs this budget, given the provider's own billing.

        **This budget counts metered searches, not provider credits**, and the
        difference is not cosmetic. Provider-native units are not comparable
        with each other: the same content search bills 2 on Tavily, 100 on Exa
        and 0 on arXiv. A ceiling denominated in those units means "12 searches"
        against one provider and "nothing at all" against another - measured on
        a real run, three Exa queries reported 300 and exhausted a 25-credit
        budget before the vendor search had issued half its queries.

        So a call that costs the account something counts as one, whatever the
        provider charges internally, and a keyless provider - arXiv, PubMed,
        Crossref, Wikipedia, DuckDuckGo - counts as nothing because it is
        genuinely free. The provider-native figure is still recorded in the
        trace, and the actual money is SearchRoute's own monthly ledger's job.
        """
        return 1 if provider_cost > 0 else 0


def budget_of(config: Any, settings: Settings | None = None) -> SessionBudget:
    """The run's shared budget, from the run config.

    Falls back to a fresh full budget when there is none - a node invoked
    directly (tests, the standalone email endpoint) should still work, and the
    alternative is every node needing a None check before it can search.
    """
    budget = ((config or {}).get("configurable") or {}).get("search_budget")
    if budget is None:
        return SessionBudget(settings=settings)
    return budget


async def search(
    query: str,
    budget: SessionBudget,
    *,
    capability: Capability = Capability.SEARCH,
    depth: Depth = Depth.SNIPPETS,
    max_results: int = 5,
    max_hydrate: int = 0,
    settings: Settings | None = None,
) -> list[Any]:
    """Run one query, charging the run budget for what it cost.

    Raises `BudgetExhausted` when the run is out of credits; callers should
    treat that as "stop and report what you have", not as a failure. Provider
    errors come back as an empty list rather than an exception - SearchRoute has
    already tried the alternatives by that point, so there is nothing left to
    fall back to and a failed enrichment query must not fail the run.

    `capability` is a filter, not a hint: arxiv, pubmed and crossref are
    ACADEMIC-only and wikipedia is REFERENCE-only, so a plain SEARCH call never
    reaches any of them.
    """
    await budget._reserve()
    client = await get_client(settings)
    try:
        response = await client.search(
            query,
            capability=capability,
            depth=depth,
            max_results=max_results,
            max_hydrate=max_hydrate,
        )
    except SearchRouteError as exc:
        # The reservation is deliberately not returned: the providers were
        # called, so the credits are spent whether or not anything useful came
        # back, and refunding them would let a failing provider be retried
        # without limit.
        log.warning("search failed for %r: %s", query, exc)
        return []

    # One charge per metered search; keyless providers cost nothing. The
    # provider's own figure goes into the trace below, not into the ceiling.
    await budget._settle(1, response.cost.total)
    _record(
        "search",
        query=query[:200],
        capability=capability.value,
        depth=response.depth.name,
        degraded=response.degraded,
        providers=list(response.providers_used),
        # `charged` is against the run ceiling; `provider_cost` is the
        # provider's own units, which differ by two orders of magnitude between
        # providers and are only comparable within one.
        charged=budget.charge_for(response.cost.total),
        provider_cost=response.cost.total,
        results=len(response.results),
        allocation=budget.name,
    )
    if response.degraded:
        log.info(
            "search %r degraded: got %s, wanted %s (%s)",
            query, response.depth.name, response.requested_depth.name,
            "; ".join(response.notes) or "no reason given",
        )
    log.info(
        "search provider=%s query=%r hits=%d cost=%d credits_left=%d",
        ",".join(response.providers_used) or "none",
        query,
        len(response.results),
        response.cost.total,
        budget.credits_remaining,
    )
    return list(response.results)


async def extract(
    urls: list[str], budget: SessionBudget, settings: Settings | None = None
) -> dict[str, str]:
    """Fetch page text for URLs, returning url -> content for what succeeded.

    Best-effort throughout: anything that cannot be fetched is simply absent
    from the result. A candidate with no page text still ships as a lead
    carrying its source URL, which beats dropping a real company over one failed
    fetch.
    """
    if not urls:
        return {}
    try:
        await budget._reserve(1)
    except BudgetExhausted:
        log.info("skipping extraction: budget exhausted, using snippets as-is")
        return {}

    client = await get_client(settings)
    try:
        documents = await client.extract(urls)
    except SearchRouteError as exc:
        log.warning("extraction failed, using snippets as-is: %s", exc)
        return {}

    # One charge for the extraction pass, not one per URL: the same reasoning
    # as search, and Tavily bills per five URLs while Exa bills per one.
    await budget._settle(1, 1)
    _record(
        "extract",
        urls=len(urls),
        retrieved=sum(1 for d in documents if d.ok and d.content),
        charged=1,
        allocation=budget.name,
    )
    return {
        doc.url: doc.content
        for doc in documents
        if doc.ok and doc.content
    }


def result_text(result: Any) -> str:
    """Best available text for a hit: full content, else summary, else snippet."""
    return result.text or ""
