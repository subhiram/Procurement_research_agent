"""Router behaviour: fallback, error classification, quota, breakers, strategies."""

from __future__ import annotations

import pytest

from searchroute.errors import NoProviderAvailable
from searchroute.ledger import MemoryStore
from searchroute.quota import Period, QuotaPolicy, QuotaTracker
from searchroute.router.breaker import BreakerState, CircuitBreaker
from searchroute.router.engine import Engine
from searchroute.router.strategy import resolve
from searchroute.types import Capability, SearchQuery

from .conftest import FakeProvider, SharedURLProvider


def build_engine(providers, strategy="priority", **kwargs) -> Engine:
    mapping = {p.name: p for p in providers}
    tracker = QuotaTracker(store=MemoryStore(), reserve_pct=kwargs.pop("reserve_pct", 0.0))
    for name, provider in mapping.items():
        tracker.register(name, provider.quota)
    return Engine(
        providers=mapping,
        strategy=resolve(strategy) if isinstance(strategy, str) else strategy,
        quota=tracker,
        breaker=CircuitBreaker(**kwargs.pop("breaker", {})),
        **kwargs,
    )


def q(text="test query", **kwargs) -> SearchQuery:
    return SearchQuery(query=text, **kwargs)


class TestFallback:
    async def test_first_provider_wins(self):
        a = FakeProvider("a", priority=1)
        b = FakeProvider("b", priority=2)
        engine = build_engine([a, b])

        response = await engine.search(q())

        assert response.providers_used == ["a"]
        assert len(b.search_calls) == 0, "second provider should not be touched"

    async def test_falls_through_to_next_on_failure(self, fatal_error):
        a = FakeProvider("a", priority=1, fail_with=fatal_error)
        b = FakeProvider("b", priority=2)
        engine = build_engine([a, b])

        response = await engine.search(q())

        assert response.providers_used == ["b"]
        assert [at.provider for at in response.attempts] == ["a", "b"]
        assert response.attempts[0].ok is False

    async def test_empty_results_are_not_an_answer(self):
        """A provider that succeeds with zero hits should not end the chain."""
        a = FakeProvider("a", priority=1, results=0)
        b = FakeProvider("b", priority=2, results=3)
        engine = build_engine([a, b])

        response = await engine.search(q())

        assert response.providers_used == ["b"]

    async def test_all_failed_raises_with_full_trail(self, fatal_error):
        a = FakeProvider("a", fail_with=fatal_error)
        b = FakeProvider("b", fail_with=fatal_error)
        engine = build_engine([a, b])

        with pytest.raises(NoProviderAvailable) as exc:
            await engine.search(q())

        assert len(exc.value.attempts) == 2
        assert "a" in str(exc.value) and "b" in str(exc.value)

    async def test_caller_order_beats_default_priority(self):
        """`providers=["tavily", "exa"]` must try Tavily first, even though Exa
        carries a lower default_priority. The caller's list is the intent."""
        # `second` has the lower default_priority, i.e. it would win a re-sort.
        first = FakeProvider("first", priority=99)
        second = FakeProvider("second", priority=1)
        engine = build_engine([first, second])

        response = await engine.search(q())

        assert response.providers_used == ["first"]
        assert second.search_calls == []

    async def test_per_call_pin_order_is_respected(self):
        a = FakeProvider("a", priority=1)
        b = FakeProvider("b", priority=2)
        engine = build_engine([a, b])

        q_ = q()
        pinned = engine.candidates(Capability.SEARCH, q_, only=["b", "a"])
        ordered = engine._ordered(pinned, Capability.SEARCH, q_)

        assert [p.name for p in ordered] == ["b", "a"]

    async def test_explicitly_named_last_resort_is_not_dropped(self):
        """`providers=["a", "ddg"]` means "try a, then ddg" — a fallback the
        caller asked for by name must never be silently discarded."""
        primary = FakeProvider("primary", priority=1)
        backup = FakeProvider("backup", priority=2, last_resort=True)
        engine = build_engine([primary, backup])

        q_ = q()
        ordered = engine._ordered(
            engine.candidates(Capability.SEARCH, q_), Capability.SEARCH, q_
        )

        assert [p.name for p in ordered] == ["primary", "backup"]

    async def test_last_resort_still_serves_when_primary_fails(self, fatal_error):
        primary = FakeProvider("primary", priority=1, fail_with=fatal_error)
        backup = FakeProvider("backup", priority=2, last_resort=True)
        engine = build_engine([primary, backup])

        response = await engine.search(q())

        assert response.providers_used == ["backup"]

    async def test_last_resort_is_always_last(self):
        """Even a strategy that loves free providers must not rank the unofficial
        backend first."""
        cheap = FakeProvider("ddg-ish", quota=None, last_resort=True, priority=1)
        keyed = FakeProvider(
            "keyed",
            quota=QuotaPolicy(limit=100, period=Period.MONTHLY),
            priority=50,
        )
        engine = build_engine([cheap, keyed], strategy="quota_aware")

        response = await engine.search(q())

        assert response.providers_used == ["keyed"]
        assert cheap.search_calls == []


class TestErrorClassification:
    async def test_quota_error_marks_tier_drained(self, quota_error):
        drained = FakeProvider(
            "drained",
            priority=1,
            fail_with=quota_error,
            quota=QuotaPolicy(limit=100, period=Period.MONTHLY),
        )
        backup = FakeProvider("backup", priority=2)
        engine = build_engine([drained, backup])

        await engine.search(q())

        # The provider told us it's out, so the local ledger must believe it
        # over its own count and stop offering the provider.
        assert engine.quota.remaining("drained") == 0
        assert engine.candidates(Capability.SEARCH, q()) == [backup]

    async def test_auth_error_trips_breaker_immediately(self, auth_error):
        bad = FakeProvider("bad", priority=1, fail_with=auth_error)
        good = FakeProvider("good", priority=2)
        engine = build_engine([bad, good])

        await engine.search(q())

        # A wrong key will not fix itself: one failure is enough.
        assert engine.breaker.state("bad") is BreakerState.OPEN

    async def test_rate_limit_retries_then_moves_on(self, rate_limit_error):
        limited = FakeProvider("limited", priority=1, fail_with=rate_limit_error)
        backup = FakeProvider("backup", priority=2)
        engine = build_engine([limited, backup], max_retries=1)

        response = await engine.search(q())

        assert len(limited.search_calls) == 2, "should retry once before giving up"
        assert response.providers_used == ["backup"]
        # Rate limiting is temporary, so it cools off rather than tripping open.
        assert engine.quota.is_disabled("limited") is True

    async def test_transient_error_is_retried(self, monkeypatch):
        from searchroute.errors import TransientError

        flaky = FakeProvider("flaky", fail_with=TransientError("flaky", "timeout"))
        engine = build_engine([flaky], max_retries=2)

        with pytest.raises(NoProviderAvailable):
            await engine.search(q())

        assert len(flaky.search_calls) == 3, "initial attempt plus two retries"


class TestQuota:
    async def test_exhausted_provider_is_skipped(self):
        small = FakeProvider(
            "small", priority=1, cost=10, quota=QuotaPolicy(limit=10, period=Period.MONTHLY)
        )
        backup = FakeProvider("backup", priority=2)
        engine = build_engine([small, backup])

        first = await engine.search(q())
        assert first.providers_used == ["small"]

        # The single call consumed the entire allowance.
        second = await engine.search(q())
        assert second.providers_used == ["backup"]

    async def test_reserve_holds_capacity_back(self):
        provider = FakeProvider(
            "p", cost=1, quota=QuotaPolicy(limit=100, period=Period.MONTHLY)
        )
        engine = build_engine([provider], reserve_pct=0.1)

        # 10% held back, so only 90 are spendable.
        assert engine.quota.remaining("p") == 90

    async def test_cost_is_debited_per_call(self):
        provider = FakeProvider(
            "p", cost=5, quota=QuotaPolicy(limit=100, period=Period.MONTHLY)
        )
        engine = build_engine([provider])

        await engine.search(q())
        await engine.search(q())

        assert engine.quota.remaining("p") == 90

    async def test_unmetered_provider_never_runs_out(self):
        provider = FakeProvider("free", quota=None)
        engine = build_engine([provider])

        assert engine.quota.remaining("free") is None
        assert engine.quota.can_afford("free", 10_000) is True


class TestBreaker:
    async def test_opens_after_threshold(self, fatal_error):
        flaky = FakeProvider("flaky", fail_with=fatal_error)
        backup = FakeProvider("backup", priority=200)
        engine = build_engine([flaky, backup], breaker={"threshold": 2})

        await engine.search(q())
        assert engine.breaker.state("flaky") is BreakerState.CLOSED

        await engine.search(q())
        assert engine.breaker.state("flaky") is BreakerState.OPEN

        calls_before = len(flaky.search_calls)
        await engine.search(q())
        assert len(flaky.search_calls) == calls_before, "open circuit should skip the call"

    async def test_success_resets_failures(self, fatal_error):
        breaker = CircuitBreaker(threshold=3)
        breaker.record_failure("p")
        breaker.record_failure("p")
        breaker.record_success("p", latency_ms=50)

        assert breaker.state("p") is BreakerState.CLOSED
        breaker.record_failure("p")
        assert breaker.state("p") is BreakerState.CLOSED, "counter should have reset"


class TestStrategies:
    async def test_quota_aware_prefers_more_headroom(self):
        nearly_out = FakeProvider(
            "nearly_out", cost=1, quota=QuotaPolicy(limit=100, period=Period.MONTHLY)
        )
        fresh = FakeProvider(
            "fresh", cost=1, quota=QuotaPolicy(limit=100, period=Period.MONTHLY)
        )
        engine = build_engine([nearly_out, fresh], strategy="quota_aware")
        engine.quota.debit("nearly_out", 95)

        response = await engine.search(q())

        assert response.providers_used == ["fresh"]

    async def test_quota_aware_saves_one_time_grants_for_last(self):
        """A grant that never renews is a reserve tank, not a first choice."""
        one_time = FakeProvider(
            "one_time",
            quota=QuotaPolicy(limit=2500, period=Period.ONE_TIME),
            priority=1,
        )
        renewing = FakeProvider(
            "renewing", quota=QuotaPolicy(limit=100, period=Period.MONTHLY), priority=2
        )
        engine = build_engine([one_time, renewing], strategy="quota_aware")

        response = await engine.search(q())

        assert response.providers_used == ["renewing"]

    async def test_quality_strategy_picks_best_scorer(self):
        weak = FakeProvider("weak", quality=0.2, priority=1)
        strong = FakeProvider("strong", quality=0.9, priority=99)
        engine = build_engine([weak, strong], strategy="quality")

        response = await engine.search(q())

        assert response.providers_used == ["strong"]

    async def test_round_robin_rotates(self):
        a = FakeProvider("a", priority=1)
        b = FakeProvider("b", priority=2)
        engine = build_engine([a, b], strategy="round_robin")

        used = [(await engine.search(q())).providers_used[0] for _ in range(4)]

        assert len(set(used)) == 2, f"expected rotation across both providers, got {used}"

    async def test_race_returns_first_success(self):
        from searchroute.router.strategy import RaceStrategy

        a = FakeProvider("a", priority=1)
        b = FakeProvider("b", priority=2)
        engine = build_engine([a, b], strategy=RaceStrategy(n=2))

        response = await engine.search(q())

        assert response.providers_used[0] in {"a", "b"}
        assert len(response.results) > 0

    async def test_fanout_fuses_and_dedupes(self):
        from searchroute.router.strategy import FanoutStrategy

        a = SharedURLProvider("a", results=3, priority=1)
        b = SharedURLProvider("b", results=3, priority=2)
        engine = build_engine([a, b], strategy=FanoutStrategy(n=2))

        response = await engine.search(q(max_results=10))

        assert len(a.search_calls) == 1 and len(b.search_calls) == 1
        # Both returned the same three articles with different utm_source values.
        assert len(response.results) == 3, "duplicates should collapse after canonicalization"
        assert set(response.providers_used) == {"a", "b"}
        # Cross-provider agreement is recorded as a signal.
        assert response.results[0].raw["_searchroute"]["found_by"] == ["a", "b"]


class TestDiagnostics:
    async def test_empty_candidates_explains_why(self, monkeypatch):
        broke = FakeProvider("broke", quota=QuotaPolicy(limit=1, period=Period.MONTHLY), cost=5)
        engine = build_engine([broke])

        with pytest.raises(NoProviderAvailable) as exc:
            await engine.search(q())

        assert "quota exhausted" in str(exc.value)

    async def test_unknown_pinned_provider_is_named(self):
        engine = build_engine([FakeProvider("a")])

        with pytest.raises(NoProviderAvailable, match="nonexistent"):
            await engine.search(q(), only=["nonexistent"])
