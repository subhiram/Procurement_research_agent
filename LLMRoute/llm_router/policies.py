"""Selection policies: given the ladder, decide what order to actually try.

The ladder says what is *acceptable*. A policy says what is *preferred*.

    free_first  take the ladder as it comes. "Just get it done."
    sticky      keep a session on the endpoint it has been using, so a long run
                does not silently change model half way through.

Both return an ordered list rather than a single pick, because a candidate can
still 429 on the actual call and the router needs somewhere to go next.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Sequence

from .ladder import Candidate

logger = logging.getLogger("llm_router.policies")

#: Sessions remembered before the oldest is dropped. Only bounded so a
#: long-lived service cannot grow this map forever.
MAX_SESSIONS = 10_000


@dataclass(frozen=True)
class SessionPin:
    """What a session settled on last time."""

    endpoint_key: str
    tier: str
    logical_model: str


class SessionState:
    """Which endpoint each session_id is pinned to. Thread-safe, in-memory."""

    def __init__(self, max_sessions: int = MAX_SESSIONS) -> None:
        self._pins: OrderedDict[str, SessionPin] = OrderedDict()
        self._max = max_sessions
        self._lock = threading.RLock()

    def get(self, session_id: str | None) -> SessionPin | None:
        if not session_id:
            return None
        with self._lock:
            pin = self._pins.get(session_id)
            if pin is not None:
                self._pins.move_to_end(session_id)
            return pin

    def remember(self, session_id: str | None, candidate: Candidate) -> None:
        """Pin a session to an endpoint after a successful call.

        A tier downgrade is deliberately *not* pinned. Downgrades are meant to
        be temporary relief while the requested tier is out of quota; pinning
        one would quietly hold the rest of the session at the lower quality even
        after the good endpoint came back - the exact silent quality drop
        sticky exists to prevent.
        """
        if not session_id or candidate.tier_downgraded:
            return
        with self._lock:
            self._pins[session_id] = SessionPin(
                endpoint_key=candidate.endpoint.key,
                tier=candidate.endpoint.tier,
                logical_model=candidate.endpoint.logical_model,
            )
            self._pins.move_to_end(session_id)
            while len(self._pins) > self._max:
                self._pins.popitem(last=False)

    def forget(self, session_id: str) -> None:
        with self._lock:
            self._pins.pop(session_id, None)

    def clear(self) -> None:
        with self._lock:
            self._pins.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._pins)


def free_first(
    candidates: Sequence[Candidate],
    session_id: str | None = None,
    session_state: SessionState | None = None,
) -> list[Candidate]:
    """Take the ladder in order: the first thing with quota wins.

    No preference beyond the ladder itself, and no signalling about tier
    downgrades - the caller asked for whatever works.
    """
    return list(candidates)


def sticky(
    candidates: Sequence[Candidate],
    session_id: str | None = None,
    session_state: SessionState | None = None,
) -> list[Candidate]:
    """Prefer the endpoint this session already used, if it is still healthy.

    Falls back to plain ladder order when the session has no pin, or the pinned
    endpoint is out of quota. A tier downgrade taken this way is surfaced by the
    router (metadata plus a warning) rather than passed off silently, because a
    consistency-sensitive run needs to know its model just changed.
    """
    ordered = list(candidates)
    pin = session_state.get(session_id) if session_state else None
    if pin is None:
        return ordered

    for index, candidate in enumerate(ordered):
        if candidate.endpoint.key == pin.endpoint_key:
            if index:
                ordered.insert(0, ordered.pop(index))
            return ordered

    logger.info(
        "session %s: pinned endpoint %s is unavailable, falling back down the ladder",
        session_id, pin.endpoint_key,
    )
    return ordered


Policy = Callable[..., "list[Candidate]"]

POLICIES: dict[str, Policy] = {
    "free_first": free_first,
    "sticky": sticky,
}

#: Policies that must not let a quality drop pass unremarked.
_SIGNALS_DOWNGRADE = frozenset({"sticky"})


def get_policy(strategy: str | Policy) -> tuple[str, Policy]:
    """Resolve a strategy name (or a custom callable) to (name, policy)."""
    if callable(strategy):
        return getattr(strategy, "__name__", "custom"), strategy
    try:
        return strategy, POLICIES[strategy]
    except KeyError:
        raise ValueError(
            f"unknown strategy {strategy!r}; available: {', '.join(sorted(POLICIES))}"
        ) from None


def signals_downgrade(strategy_name: str) -> bool:
    return strategy_name in _SIGNALS_DOWNGRADE


def register_policy(name: str, policy: Policy, *, signal_downgrade: bool = False) -> None:
    """Add a custom strategy, usable as route(..., strategy=name)."""
    global _SIGNALS_DOWNGRADE
    POLICIES[name] = policy
    if signal_downgrade:
        _SIGNALS_DOWNGRADE = _SIGNALS_DOWNGRADE | {name}
