"""Per-provider circuit breakers.

Without these, a provider having a bad ten minutes costs every single call a
timeout before the fallback kicks in. The breaker remembers the failure and
skips straight past it, then lets a single probe through once the cooldown
expires to find out whether it recovered.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum


class BreakerState(str, Enum):
    CLOSED = "closed"
    """Healthy — calls flow."""
    OPEN = "open"
    """Failing — calls are skipped without being attempted."""
    HALF_OPEN = "half_open"
    """Cooldown elapsed; one probe is allowed through."""


@dataclass(slots=True)
class _Entry:
    failures: int = 0
    opened_at: float = 0.0
    state: BreakerState = BreakerState.CLOSED
    ewma_latency: float = 0.0


@dataclass(slots=True)
class CircuitBreaker:
    """Tracks consecutive failures per provider.

    Only *infrastructure* failures should reach here. Quota exhaustion is not a
    breaker concern — it's tracked by the ledger and resets on a schedule, so
    tripping a breaker on it would be the wrong recovery model.
    """

    threshold: int = 3
    cooldown: float = 60.0
    _entries: dict[str, _Entry] = field(default_factory=dict)

    def _entry(self, provider: str) -> _Entry:
        return self._entries.setdefault(provider, _Entry())

    def state(self, provider: str) -> BreakerState:
        entry = self._entry(provider)
        if entry.state is BreakerState.OPEN:
            if time.monotonic() - entry.opened_at >= self.cooldown:
                entry.state = BreakerState.HALF_OPEN
        return entry.state

    def allows(self, provider: str) -> bool:
        return self.state(provider) is not BreakerState.OPEN

    def record_success(self, provider: str, latency_ms: float = 0.0) -> None:
        entry = self._entry(provider)
        entry.failures = 0
        entry.state = BreakerState.CLOSED
        if latency_ms:
            # EWMA so the latency strategy reacts to trends, not single spikes.
            entry.ewma_latency = (
                latency_ms
                if not entry.ewma_latency
                else 0.7 * entry.ewma_latency + 0.3 * latency_ms
            )

    def record_failure(self, provider: str) -> None:
        entry = self._entry(provider)
        entry.failures += 1
        if entry.failures >= self.threshold:
            entry.state = BreakerState.OPEN
            entry.opened_at = time.monotonic()

    def trip(self, provider: str) -> None:
        """Open immediately, skipping the failure count. Used for auth errors:
        a bad key will not fix itself on retry."""
        entry = self._entry(provider)
        entry.failures = self.threshold
        entry.state = BreakerState.OPEN
        entry.opened_at = time.monotonic()

    def latencies(self) -> dict[str, float]:
        return {name: e.ewma_latency for name, e in self._entries.items() if e.ewma_latency}

    def snapshot(self) -> dict[str, dict]:
        return {
            name: {
                "state": self.state(name).value,
                "failures": e.failures,
                "latency_ms": round(e.ewma_latency, 1),
            }
            for name, e in self._entries.items()
        }
