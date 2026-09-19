"""Per-provider request pacing.

A second constraint the quota ledger cannot express. Quota counts *credits*;
this counts *rate*. They are independent: arXiv is free and unmetered, yet asks
for one request every 3 seconds, and having 100% of your quota left does nothing
to make a burst acceptable.

Free public APIs stay free because people respect these limits, so a fan-out
that fires five arXiv queries at once is not just rude, it gets the whole
library blocked for everyone using it.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field


@dataclass
class RateGate:
    """Serializes calls per provider, spacing them by ``min_interval``.

    One lock per provider, so pacing arXiv never delays a Tavily call happening
    concurrently. Providers with ``min_interval == 0`` take a fast path and
    allocate nothing.
    """

    _locks: dict[str, asyncio.Lock] = field(default_factory=dict)
    _next_allowed: dict[str, float] = field(default_factory=dict)

    def _lock(self, provider: str) -> asyncio.Lock:
        lock = self._locks.get(provider)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[provider] = lock
        return lock

    async def acquire(self, provider: str, min_interval: float) -> float:
        """Wait until this provider may be called again. Returns seconds waited.

        The lock is held across the sleep on purpose: two concurrent callers
        must queue behind each other rather than both reading the same
        ``next_allowed`` and then firing together, which is precisely the burst
        the interval exists to prevent.
        """
        if min_interval <= 0:
            return 0.0

        async with self._lock(provider):
            now = time.monotonic()
            earliest = self._next_allowed.get(provider, 0.0)
            waited = 0.0
            if earliest > now:
                waited = earliest - now
                await asyncio.sleep(waited)
                now = time.monotonic()
            self._next_allowed[provider] = now + min_interval
            return waited

    def reset(self, provider: str | None = None) -> None:
        """Forget pacing state. Mostly for tests."""
        if provider is None:
            self._next_allowed.clear()
        else:
            self._next_allowed.pop(provider, None)
