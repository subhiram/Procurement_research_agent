"""Free-tier accounting.

This is the piece that makes running on several free tiers actually viable: we
keep a local count of what each provider has spent inside its current reset
window, so the router can prefer whichever tier has room instead of discovering
exhaustion by getting a 402 back.

The ledger is deliberately *advisory*. It can drift — another process, another
machine, a call we never saw. So a real ``QuotaExceeded`` from the provider
always wins and immediately syncs the local counter to drained.
"""

from __future__ import annotations

import calendar
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Protocol, runtime_checkable


class Period(str, Enum):
    NONE = "none"
    """Unmetered (self-hosted SearXNG, keyless public APIs)."""
    DAILY = "daily"
    MONTHLY = "monthly"
    ONE_TIME = "one_time"
    """A signup grant that never renews. Spend it last."""


class Unit(str, Enum):
    REQUESTS = "requests"
    CREDITS = "credits"


class Anchor(str, Enum):
    CALENDAR = "calendar"
    """Resets on the 1st (or at UTC midnight for daily)."""
    SIGNUP_DAY = "signup_day"
    """Resets on the day of the month you signed up."""


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _month_anchor(year: int, month: int, day: int) -> datetime:
    """The anchor instant for a given month, clamped to that month's length.

    Always re-derived from the *original* anchor day. Clamping once and reusing
    the clamped value would drift: someone who signed up on the 31st would see
    their February-clamped 28th carried back into January.
    """
    day = min(day, calendar.monthrange(year, month)[1])
    return datetime(year, month, day, tzinfo=timezone.utc)


@dataclass(frozen=True, slots=True)
class QuotaPolicy:
    """What one provider's free tier looks like."""

    limit: int
    unit: Unit = Unit.REQUESTS
    period: Period = Period.MONTHLY
    anchor: Anchor = Anchor.CALENDAR
    anchor_day: int = 1
    """Day of month the window rolls over when ``anchor`` is SIGNUP_DAY."""

    @property
    def metered(self) -> bool:
        return self.period is not Period.NONE and self.limit > 0

    @property
    def renews(self) -> bool:
        """One-time grants are worth preserving: once spent they never come back."""
        return self.period in (Period.DAILY, Period.MONTHLY)

    def window_start(self, now: datetime | None = None) -> datetime:
        """Start of the window ``now`` falls inside."""
        now = now or _utcnow()
        if self.period is Period.DAILY:
            return now.replace(hour=0, minute=0, second=0, microsecond=0)
        if self.period is Period.MONTHLY:
            target_day = 1 if self.anchor is Anchor.CALENDAR else self.anchor_day
            candidate = _month_anchor(now.year, now.month, target_day)
            if candidate > now:
                # The anchor day hasn't arrived yet this month, so the current
                # window opened last month.
                year, month = (
                    (now.year - 1, 12) if now.month == 1 else (now.year, now.month - 1)
                )
                candidate = _month_anchor(year, month, target_day)
            return candidate
        # NONE and ONE_TIME have no rolling window.
        return datetime.min.replace(tzinfo=timezone.utc)

    def window_end(self, now: datetime | None = None) -> datetime | None:
        now = now or _utcnow()
        start = self.window_start(now)
        if self.period is Period.DAILY:
            return start + timedelta(days=1)
        if self.period is Period.MONTHLY:
            days = calendar.monthrange(start.year, start.month)[1]
            return start + timedelta(days=days)
        return None


@dataclass(slots=True)
class QuotaState:
    """Per-provider counters, as persisted."""

    used: int = 0
    window_start: str = ""
    disabled_until: str | None = None
    """Set when a provider hard-fails (auth) or is cooling off."""

    def to_dict(self) -> dict:
        return {
            "used": self.used,
            "window_start": self.window_start,
            "disabled_until": self.disabled_until,
        }

    @classmethod
    def from_dict(cls, data: dict) -> QuotaState:
        return cls(
            used=int(data.get("used", 0)),
            window_start=data.get("window_start", ""),
            disabled_until=data.get("disabled_until"),
        )


@runtime_checkable
class StateStore(Protocol):
    """Where quota counters live.

    Implementations are in ``ledger.py``: in-memory for tests, a lock-protected
    JSON file by default, Redis for multi-machine deployments.
    """

    def get(self, provider: str) -> QuotaState: ...

    def set(self, provider: str, state: QuotaState) -> None: ...

    def all(self) -> dict[str, QuotaState]: ...

    def update(
        self, provider: str, mutate: Callable[[QuotaState], QuotaState]
    ) -> QuotaState:
        """Apply ``mutate`` to the stored state atomically.

        Every counter change must go through this rather than get-then-set.
        A read outside the lock followed by a write inside it is a lost update:
        with several worker processes debiting the same key, most of the spend
        silently disappears and the tracker thinks the tier is still full.
        """
        ...


@dataclass(slots=True)
class QuotaTracker:
    """Reads and writes provider spend through a pluggable store.

    ``reserve_pct`` holds a sliver of every tier back, so a background job can't
    drain the last credit that an interactive call was going to need.
    """

    store: StateStore
    reserve_pct: float = 0.05
    _policies: dict[str, QuotaPolicy] = field(default_factory=dict)

    def register(self, provider: str, policy: QuotaPolicy | None) -> None:
        if policy is not None:
            self._policies[provider] = policy

    def _state(self, provider: str, policy: QuotaPolicy, now: datetime) -> QuotaState:
        state = self.store.get(provider)
        expected = policy.window_start(now).isoformat()
        if policy.renews and state.window_start != expected:
            # Window rolled over: the tier refilled.
            state = QuotaState(used=0, window_start=expected)
            self.store.set(provider, state)
        elif not state.window_start:
            state.window_start = expected
        return state

    def effective_limit(self, policy: QuotaPolicy) -> int:
        return max(0, int(policy.limit * (1.0 - self.reserve_pct)))

    def remaining(self, provider: str, now: datetime | None = None) -> int | None:
        """Credits left in this window. ``None`` means unmetered."""
        policy = self._policies.get(provider)
        if policy is None or not policy.metered:
            return None
        state = self._state(provider, policy, now or _utcnow())
        return max(0, self.effective_limit(policy) - state.used)

    def fraction_remaining(self, provider: str, now: datetime | None = None) -> float:
        """0-1, used for ordering. Unmetered providers score 1.0 — they're free."""
        policy = self._policies.get(provider)
        if policy is None or not policy.metered:
            return 1.0
        limit = self.effective_limit(policy)
        if limit <= 0:
            return 0.0
        return self.remaining(provider, now) / limit

    def can_afford(self, provider: str, cost: int, now: datetime | None = None) -> bool:
        now = now or _utcnow()
        if self.is_disabled(provider, now):
            return False
        remaining = self.remaining(provider, now)
        return remaining is None or remaining >= cost

    def _rollover(self, state: QuotaState, policy: QuotaPolicy, expected: str) -> QuotaState:
        """Reset the counter if the window turned over since we last looked."""
        if policy.renews and state.window_start != expected:
            return QuotaState(used=0, window_start=expected, disabled_until=state.disabled_until)
        if not state.window_start:
            state.window_start = expected
        return state

    def debit(self, provider: str, cost: int, now: datetime | None = None) -> None:
        policy = self._policies.get(provider)
        if policy is None or not policy.metered or cost <= 0:
            return
        expected = policy.window_start(now or _utcnow()).isoformat()

        def mutate(state: QuotaState) -> QuotaState:
            state = self._rollover(state, policy, expected)
            state.used += cost
            return state

        # Read-modify-write happens inside the store's lock, not around it.
        self.store.update(provider, mutate)

    def mark_exhausted(self, provider: str, now: datetime | None = None) -> None:
        """The provider told us it's out. Believe it over our own count."""
        policy = self._policies.get(provider)
        now = now or _utcnow()
        if policy is None or not policy.metered:
            # Unmetered provider that still refused us: cool off for an hour.
            self.disable_until(provider, now + timedelta(hours=1))
            return
        expected = policy.window_start(now).isoformat()

        def mutate(state: QuotaState) -> QuotaState:
            state = self._rollover(state, policy, expected)
            state.used = max(state.used, policy.limit)
            return state

        self.store.update(provider, mutate)

    def disable_until(self, provider: str, until: datetime) -> None:
        def mutate(state: QuotaState) -> QuotaState:
            state.disabled_until = until.isoformat()
            return state

        self.store.update(provider, mutate)

    def is_disabled(self, provider: str, now: datetime | None = None) -> bool:
        state = self.store.get(provider)
        if not state.disabled_until:
            return False
        now = now or _utcnow()
        try:
            until = datetime.fromisoformat(state.disabled_until)
        except ValueError:
            return False
        if until.tzinfo is None:
            until = until.replace(tzinfo=timezone.utc)
        return now < until

    def snapshot(self, now: datetime | None = None) -> dict[str, dict]:
        """Human-readable state, for logging and the smoke script."""
        now = now or _utcnow()
        out: dict[str, dict] = {}
        for provider, policy in self._policies.items():
            end = policy.window_end(now)
            out[provider] = {
                "limit": policy.limit,
                "unit": policy.unit.value,
                "period": policy.period.value,
                "remaining": self.remaining(provider, now),
                "resets_at": end.isoformat() if end else None,
                "disabled": self.is_disabled(provider, now),
            }
        return out
