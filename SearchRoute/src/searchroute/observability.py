"""Seeing what the router actually did.

The engine already emits events through its hook list; this formalizes the event
names and payloads, and ships a collector so an app gets useful numbers without
writing its own aggregation.

Deliberately dependency-free — no OpenTelemetry coupling. A hook is a plain
callable, so wiring this into OTel, StatsD or structured logs is a few lines in
the consuming app rather than a dependency in the library.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("searchroute")

#: Emitted when a provider call succeeds.
#: payload: {"provider": str, "latency_ms": float}
EVENT_SUCCESS = "success"

#: Emitted when a provider call fails after its retries.
#: payload: {"provider": str, "kind": ErrorKind | None, "error": str}
EVENT_FAILURE = "failure"

#: Emitted when an extract provider fails; hydration continues down the chain.
#: payload: {"provider": str, "error": str}
EVENT_EXTRACT_FAILURE = "extract_failure"

EVENTS = (EVENT_SUCCESS, EVENT_FAILURE, EVENT_EXTRACT_FAILURE)


@dataclass(slots=True)
class ProviderStats:
    calls: int = 0
    successes: int = 0
    failures: int = 0
    total_latency_ms: float = 0.0
    errors_by_kind: dict[str, int] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        return self.successes / self.calls if self.calls else 0.0

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.successes if self.successes else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "successes": self.successes,
            "failures": self.failures,
            "success_rate": round(self.success_rate, 3),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "errors_by_kind": dict(self.errors_by_kind),
        }


class StatsCollector:
    """A hook that aggregates per-provider outcomes.

        stats = StatsCollector()
        sr = SearchRoute(hooks=[stats])
        ...
        stats.snapshot()

    Counts what the router *did*, which is different from what the quota ledger
    says it spent: a call that failed still cost latency and still tells you a
    provider is unhealthy, even though it consumed no credits.
    """

    def __init__(self) -> None:
        self._providers: dict[str, ProviderStats] = defaultdict(ProviderStats)

    def __call__(self, event: str, payload: dict[str, Any]) -> None:
        name = payload.get("provider")
        if not name:
            return
        stats = self._providers[name]

        if event == EVENT_SUCCESS:
            stats.calls += 1
            stats.successes += 1
            stats.total_latency_ms += float(payload.get("latency_ms") or 0.0)
        elif event == EVENT_FAILURE:
            stats.calls += 1
            stats.failures += 1
            kind = payload.get("kind")
            # ErrorKind is a str-Enum; normalize either form to its value.
            key = getattr(kind, "value", None) or str(kind or "unknown")
            stats.errors_by_kind[key] = stats.errors_by_kind.get(key, 0) + 1
        elif event == EVENT_EXTRACT_FAILURE:
            stats.failures += 1
            stats.errors_by_kind["extract"] = stats.errors_by_kind.get("extract", 0) + 1

    def snapshot(self) -> dict[str, dict[str, Any]]:
        return {name: s.to_dict() for name, s in sorted(self._providers.items())}

    def reset(self) -> None:
        self._providers.clear()

    @property
    def total_calls(self) -> int:
        return sum(s.calls for s in self._providers.values())


def log_events(level: int = logging.INFO):
    """A hook that logs every event through the ``searchroute`` logger.

        sr = SearchRoute(hooks=[log_events()])
    """

    def hook(event: str, payload: dict[str, Any]) -> None:
        if event == EVENT_SUCCESS:
            logger.log(
                level,
                "%s ok in %.0fms",
                payload.get("provider"),
                payload.get("latency_ms") or 0.0,
            )
        else:
            # Failures are worth a louder level than successes.
            logger.warning(
                "%s %s: %s",
                payload.get("provider"),
                event,
                payload.get("error"),
            )

    return hook
