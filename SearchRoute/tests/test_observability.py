"""Hooks and the stats collector."""

from __future__ import annotations

from searchroute.client import AsyncSearchRoute
from searchroute.observability import (
    EVENT_EXTRACT_FAILURE,
    EVENT_FAILURE,
    EVENT_SUCCESS,
    StatsCollector,
)
from searchroute.types import ErrorKind

from .conftest import FakeProvider


def client(providers, **kwargs) -> AsyncSearchRoute:
    kwargs.setdefault("quota_store", "memory")
    return AsyncSearchRoute(custom_providers=providers, **kwargs)


class TestStatsCollector:
    def test_counts_successes_and_latency(self):
        stats = StatsCollector()
        stats(EVENT_SUCCESS, {"provider": "exa", "latency_ms": 100.0})
        stats(EVENT_SUCCESS, {"provider": "exa", "latency_ms": 200.0})

        snap = stats.snapshot()["exa"]
        assert snap["calls"] == 2
        assert snap["successes"] == 2
        assert snap["success_rate"] == 1.0
        assert snap["avg_latency_ms"] == 150.0

    def test_records_error_kinds(self):
        stats = StatsCollector()
        stats(EVENT_FAILURE, {"provider": "exa", "kind": ErrorKind.QUOTA, "error": "out"})
        stats(EVENT_FAILURE, {"provider": "exa", "kind": ErrorKind.QUOTA, "error": "out"})
        stats(EVENT_FAILURE, {"provider": "exa", "kind": ErrorKind.AUTH, "error": "bad key"})

        snap = stats.snapshot()["exa"]
        assert snap["failures"] == 3
        assert snap["errors_by_kind"] == {"quota": 2, "auth": 1}
        assert snap["success_rate"] == 0.0

    def test_mixed_outcomes_give_a_real_success_rate(self):
        stats = StatsCollector()
        for _ in range(3):
            stats(EVENT_SUCCESS, {"provider": "p", "latency_ms": 10})
        stats(EVENT_FAILURE, {"provider": "p", "kind": ErrorKind.TRANSIENT})

        assert stats.snapshot()["p"]["success_rate"] == 0.75
        assert stats.total_calls == 4

    def test_extract_failures_are_tracked_separately(self):
        stats = StatsCollector()
        stats(EVENT_EXTRACT_FAILURE, {"provider": "firecrawl", "error": "down"})

        snap = stats.snapshot()["firecrawl"]
        assert snap["errors_by_kind"] == {"extract": 1}

    def test_events_without_a_provider_are_ignored(self):
        stats = StatsCollector()
        stats(EVENT_SUCCESS, {"latency_ms": 5})
        assert stats.snapshot() == {}

    def test_reset(self):
        stats = StatsCollector()
        stats(EVENT_SUCCESS, {"provider": "p", "latency_ms": 1})
        stats.reset()
        assert stats.snapshot() == {}


class TestClientIntegration:
    async def test_stats_are_collected_without_opting_in(self):
        provider = FakeProvider("p")
        sr = client([provider])

        await sr.search("q")
        await sr.search("q")

        snap = sr.stats()["p"]
        assert snap["calls"] == 2 and snap["successes"] == 2

    async def test_failures_show_up_with_their_kind(self, quota_error):
        broken = FakeProvider("broken", priority=1, fail_with=quota_error)
        working = FakeProvider("working", priority=2)
        sr = client([broken, working])

        await sr.search("q")

        assert sr.stats()["broken"]["errors_by_kind"] == {"quota": 1}
        assert sr.stats()["working"]["successes"] == 1

    async def test_user_hooks_still_run_alongside_stats(self):
        seen = []
        provider = FakeProvider("p")
        sr = client([provider], hooks=[lambda e, p: seen.append(e)])

        await sr.search("q")

        assert seen == ["success"]
        assert sr.stats()["p"]["calls"] == 1

    async def test_a_raising_hook_cannot_break_the_search(self):
        """Observability must never take down the thing it observes."""

        def bad_hook(event, payload):
            raise RuntimeError("hook exploded")

        provider = FakeProvider("p")
        sr = client([provider], hooks=[bad_hook])

        response = await sr.search("q")

        assert len(response.results) > 0
        assert sr.stats()["p"]["calls"] == 1

    async def test_status_and_stats_answer_different_questions(self):
        provider = FakeProvider("p")
        sr = client([provider])
        await sr.search("q")

        # status = current state; stats = history.
        assert set(sr.status()) == {"providers", "strategy", "quota", "breakers"}
        assert sr.stats()["p"]["calls"] == 1
