"""Per-provider request pacing.

This is the constraint the quota ledger cannot express. arXiv is free and
unmetered yet allows one request every 3 seconds, so "plenty of quota left" says
nothing about whether a burst is acceptable.
"""

from __future__ import annotations

import asyncio
import time

from searchroute.router.ratelimit import RateGate

from .conftest import FakeProvider
from .test_router import build_engine, q


class TestRateGate:
    async def test_zero_interval_never_waits(self):
        gate = RateGate()
        started = time.monotonic()
        for _ in range(50):
            await gate.acquire("fast", 0.0)
        assert time.monotonic() - started < 0.05

    async def test_second_call_waits_the_interval(self):
        gate = RateGate()
        await gate.acquire("slow", 0.15)
        started = time.monotonic()
        await gate.acquire("slow", 0.15)
        assert time.monotonic() - started >= 0.14

    async def test_concurrent_callers_queue_instead_of_bursting(self):
        """The whole point: five parallel calls must be spaced, not fired at once.

        A gate that read the timestamp outside the lock would let all five see
        the same value and burst — which is exactly what gets an API key banned.
        """
        gate = RateGate()
        interval = 0.05
        stamps: list[float] = []

        async def call():
            await gate.acquire("arxiv", interval)
            stamps.append(time.monotonic())

        await asyncio.gather(*(call() for _ in range(5)))

        stamps.sort()
        gaps = [b - a for a, b in zip(stamps, stamps[1:], strict=False)]
        assert all(gap >= interval * 0.8 for gap in gaps), f"burst detected: {gaps}"

    async def test_providers_are_paced_independently(self):
        """Pacing arXiv must not delay an unrelated provider."""
        gate = RateGate()
        await gate.acquire("slow", 1.0)

        started = time.monotonic()
        await gate.acquire("other", 0.0)
        assert time.monotonic() - started < 0.05

    async def test_reset_clears_state(self):
        gate = RateGate()
        await gate.acquire("p", 5.0)
        gate.reset("p")
        started = time.monotonic()
        await gate.acquire("p", 5.0)
        assert time.monotonic() - started < 0.05


class TestEngineRespectsIt:
    async def test_sequential_calls_are_paced(self):
        slow = FakeProvider("slow")
        slow.min_interval = 0.1
        engine = build_engine([slow])

        started = time.monotonic()
        await engine.search(q())
        await engine.search(q())

        assert time.monotonic() - started >= 0.09

    async def test_fanout_does_not_burst_a_paced_provider(self):
        from searchroute.router.strategy import FanoutStrategy

        a = FakeProvider("a", priority=1)
        b = FakeProvider("b", priority=2)
        for provider in (a, b):
            provider.min_interval = 0.0
        paced = FakeProvider("paced", priority=3)
        paced.min_interval = 0.08

        engine = build_engine([a, b, paced], strategy=FanoutStrategy(n=3))

        started = time.monotonic()
        await engine.search(q())
        # Only one call to `paced` in a single fan-out, so no wait yet; the
        # second fan-out must wait for its interval.
        await engine.search(q())
        assert time.monotonic() - started >= 0.07

    async def test_unpaced_providers_are_unaffected(self):
        fast = FakeProvider("fast")
        engine = build_engine([fast])

        started = time.monotonic()
        for _ in range(5):
            await engine.search(q())
        assert time.monotonic() - started < 0.2
