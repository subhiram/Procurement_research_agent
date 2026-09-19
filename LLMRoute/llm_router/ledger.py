"""UsageLedger - rolling-window quota tracking per endpoint.

Only Groq (response headers) and OpenRouter (a key endpoint) can be asked how
much quota is left. For everyone else this ledger is the only source of truth:
it is seeded from limits.yaml, counts what we spend locally, and is corrected
whenever a real 429 arrives.

Counting is deliberately conservative. Every ambiguity resolves towards
"we have used more than we think", because over-counting costs a fallback hop
while under-counting costs a 429 and a wasted round trip.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Deque, Iterable, Mapping

from .registry import REQUESTS, TOKENS, Endpoint, window_name

logger = logging.getLogger("llm_router.ledger")

# When a 429 arrives with no Retry-After and the endpoint's provider config has
# no cooldown either.
DEFAULT_COOLDOWN_SECONDS = 60.0

# A 429 that arrives with an implausibly long Retry-After is clamped, so one bad
# header cannot park an endpoint for the rest of the process's life.
MAX_COOLDOWN_SECONDS = 3600.0


@dataclass(frozen=True)
class Availability:
    """Why an endpoint can or cannot be called right now."""

    ok: bool
    reason: str | None = None
    #  seconds until this endpoint could plausibly work again; None = unknown
    retry_after: float | None = None

    def __bool__(self) -> bool:
        return self.ok


AVAILABLE = Availability(ok=True)


@dataclass
class _Counter:
    """Rolling-window usage for one (kind, window) pair on one ledger key.

    A counter holds usage only, never a limit. The two are separate because a
    shared bucket is checked against whichever endpoint is asking: Google's
    quota is per project, but gemini-flash-lite is allowed far more of it than
    gemini-pro. Storing one limit on the bucket would force the tightest of them
    onto every model and throw away most of the free tier.
    """

    window: int
    #: Highest limit seen for this counter. Display only, never enforced.
    display_limit: int = 0
    events: Deque[tuple[float, int]] = field(default_factory=deque)
    total: int = 0
    # Usage asserted by a provider's own headers, which outranks local counting
    # while it is still fresh (see UsageLedger.sync_from_headers).
    asserted_used: int | None = None
    asserted_until: float = 0.0

    def prune(self, now: float) -> None:
        cutoff = now - self.window
        events = self.events
        while events and events[0][0] <= cutoff:
            self.total -= events.popleft()[1]
        if self.total < 0:  # defensive; should not happen
            self.total = 0
        if self.asserted_used is not None and now >= self.asserted_until:
            self.asserted_used = None

    def used(self, now: float) -> int:
        self.prune(now)
        if self.asserted_used is not None:
            return max(self.total, self.asserted_used)
        return self.total

    def add(self, now: float, amount: int) -> None:
        if amount <= 0:
            return
        self.prune(now)
        self.events.append((now, amount))
        self.total += amount
        if self.asserted_used is not None:
            self.asserted_used += amount

    def remaining(self, now: float, limit: int) -> int:
        return max(0, limit - self.used(now))

    def retry_after(self, now: float, need: int, limit: int) -> float:
        """Seconds until `need` units free up under `limit`. 0 if already free."""
        self.prune(now)
        deficit = self.used(now) + need - limit
        if deficit <= 0:
            return 0.0
        # Events expire oldest-first; walk forward until enough has aged out.
        freed = 0
        for timestamp, amount in self.events:
            freed += amount
            if freed >= deficit:
                return max(0.0, timestamp + self.window - now)
        # Nothing local to expire (usage is asserted, or need > limit).
        return float(self.window)


@dataclass
class _Bucket:
    """All state for one ledger key."""

    counters: dict[tuple[str, int], _Counter] = field(default_factory=dict)
    cooldown_until: float = 0.0
    cooldown_reason: str | None = None
    last_call_at: float = 0.0
    calls: int = 0
    rate_limit_events: int = 0


def _match_window(endpoint: Endpoint, kind: str, reported_limit: int, fallback: int) -> int:
    """Which of this endpoint's configured windows a reported limit belongs to.

    Providers report an absolute ceiling, not a window label, so the ceiling
    itself is the identifying signal: limits.yaml already states, per window,
    what that ceiling should be (rpm vs rpd are normally different numbers), so
    whichever configured window's value the header actually matches is almost
    certainly the one it is describing.
    """
    candidates = [limit for limit in endpoint.limits if limit.kind == kind]
    for limit in candidates:
        if limit.value == reported_limit:
            return limit.window
    if candidates:
        # No exact match: the provider's real number differs from our config's
        # guess, and a single header pair alone cannot say which window it
        # belongs to. Defaulting to the *largest* configured window is the safe
        # side of that guess - it is exactly this mismatch (a big absolute
        # number landing on a short window) that made the ledger think a
        # request was fine when the real limit was much tighter, in the case
        # that motivated this function. Log it either way, since it means
        # limits.yaml no longer matches the account.
        closest = max(candidates, key=lambda limit: limit.window)
        logger.info(
            "%s reported %s limit %d for a window not in limits.yaml "
            "(configured: %s); assuming the %s window",
            endpoint.key, kind, reported_limit,
            [limit.value for limit in candidates], window_name(closest.window),
        )
        return closest.window
    return fallback


class UsageLedger:
    """Tracks request and token spend per endpoint bucket.

    The bucket is `Endpoint.ledger_key`, so per-model providers get one bucket
    per model while per-account, per-project and per-session providers pool all
    their models into a single bucket - which is how those providers actually
    meter them. Google AI Studio is the case that matters: quota is per project,
    so spending it on gemini-flash genuinely does deny gemini-pro.

    Thread-safe. `try_acquire` is the atomic check-and-reserve the router uses,
    so concurrent callers cannot both pass a check for the last free slot.
    """

    def __init__(self, *, time_fn: Callable[[], float] | None = None) -> None:
        self._now: Callable[[], float] = time_fn or time.time
        self._buckets: dict[str, _Bucket] = {}
        self._lock = threading.RLock()

    # -- internals ---------------------------------------------------------- #

    def _bucket(self, endpoint: Endpoint) -> _Bucket:
        bucket = self._buckets.get(endpoint.ledger_key)
        if bucket is None:
            bucket = _Bucket()
            self._buckets[endpoint.ledger_key] = bucket
        # Counters are registered on first sight, since a shared bucket may see
        # models with different window sets.
        for limit in endpoint.limits:
            key = (limit.kind, limit.window)
            counter = bucket.counters.get(key)
            if counter is None:
                counter = _Counter(window=limit.window)
                bucket.counters[key] = counter
            counter.display_limit = max(counter.display_limit, limit.value)
        return bucket

    def _availability(
        self, endpoint: Endpoint, *, estimated_tokens: int, now: float
    ) -> Availability:
        """Every reason this endpoint is blocked, reported as the worst one.

        All blockers are evaluated rather than short-circuiting on the first,
        because the one that matters is whichever clears last: being told to
        wait 12s for call spacing is misleading when the daily quota is also
        gone. The router uses that retry_after to pick the soonest live
        endpoint, so it has to be the true wait.
        """
        bucket = self._bucket(endpoint)
        blockers: list[Availability] = []

        if now < bucket.cooldown_until:
            blockers.append(Availability(
                ok=False,
                reason=bucket.cooldown_reason or "cooling down after a rate limit",
                retry_after=bucket.cooldown_until - now,
            ))

        if endpoint.min_interval > 0 and bucket.last_call_at:
            elapsed = now - bucket.last_call_at
            if elapsed < endpoint.min_interval:
                blockers.append(Availability(
                    ok=False,
                    reason=(
                        f"minimum spacing {endpoint.min_interval:g}s between calls "
                        f"not yet elapsed"
                    ),
                    retry_after=endpoint.min_interval - elapsed,
                ))

        # Checked against *this endpoint's* limits, over the bucket's usage.
        for limit in endpoint.limits:
            need = 1 if limit.kind == REQUESTS else max(0, estimated_tokens)
            if need == 0:
                continue
            counter = bucket.counters.get((limit.kind, limit.window))
            if counter is None:
                continue
            used = counter.used(now)
            if used + need > limit.value:
                blockers.append(Availability(
                    ok=False,
                    reason=(
                        f"{limit.kind} limit reached "
                        f"({used}/{limit.value} per {window_name(limit.window)})"
                    ),
                    retry_after=counter.retry_after(now, need, limit.value),
                ))

        if not blockers:
            return AVAILABLE
        return max(blockers, key=lambda b: b.retry_after or 0.0)

    def _record_call(self, endpoint: Endpoint, tokens_used: int | None, now: float) -> None:
        bucket = self._bucket(endpoint)
        bucket.calls += 1
        bucket.last_call_at = now
        for (kind, _window), counter in bucket.counters.items():
            if kind == REQUESTS:
                counter.add(now, 1)
            elif tokens_used:
                counter.add(now, int(tokens_used))

    # -- public API --------------------------------------------------------- #

    def can_call(self, endpoint: Endpoint, *, estimated_tokens: int = 0) -> bool:
        """True if this endpoint has quota and is not in a cooldown."""
        with self._lock:
            return self._availability(
                endpoint, estimated_tokens=estimated_tokens, now=self._now()
            ).ok

    def availability(
        self, endpoint: Endpoint, *, estimated_tokens: int = 0
    ) -> Availability:
        """can_call() with the reason and an estimated retry_after attached."""
        with self._lock:
            return self._availability(
                endpoint, estimated_tokens=estimated_tokens, now=self._now()
            )

    def try_acquire(
        self, endpoint: Endpoint, *, estimated_tokens: int = 0
    ) -> Availability:
        """Atomically check quota and reserve one request against it.

        The router calls this instead of can_call() + record_call() so two
        threads cannot both win the last slot. Token spend is unknown until the
        response comes back, so it is added afterwards with record_tokens().
        """
        with self._lock:
            now = self._now()
            verdict = self._availability(
                endpoint, estimated_tokens=estimated_tokens, now=now
            )
            if verdict.ok:
                self._record_call(endpoint, None, now)
            return verdict

    def record_call(self, endpoint: Endpoint, tokens_used: int | None = None) -> None:
        """Record one completed call, and its token spend if known."""
        with self._lock:
            self._record_call(endpoint, tokens_used, self._now())

    def record_tokens(self, endpoint: Endpoint, tokens_used: int | None) -> None:
        """Add token spend for a request already reserved by try_acquire()."""
        if not tokens_used:
            return
        with self._lock:
            now = self._now()
            bucket = self._bucket(endpoint)
            for (kind, _window), counter in bucket.counters.items():
                if kind == TOKENS:
                    counter.add(now, int(tokens_used))

    def record_rate_limited(
        self,
        endpoint: Endpoint,
        retry_after: float | None = None,
        *,
        reason: str | None = None,
    ) -> float:
        """Mark an endpoint unavailable after a real 429.

        This is the correction mechanism: whatever limits.yaml claimed, the
        provider has just told us the truth. Returns the cooldown applied.
        """
        with self._lock:
            now = self._now()
            bucket = self._bucket(endpoint)
            bucket.rate_limit_events += 1
            if retry_after is None or retry_after <= 0:
                cooldown = endpoint.cooldown_seconds or DEFAULT_COOLDOWN_SECONDS
            else:
                cooldown = min(float(retry_after), MAX_COOLDOWN_SECONDS)
            # A 429 is itself evidence the request was spent.
            bucket.cooldown_until = max(bucket.cooldown_until, now + cooldown)
            bucket.cooldown_reason = reason or (
                f"rate limited by {endpoint.provider}; retrying in {cooldown:.0f}s"
            )
            logger.info(
                "%s rate limited, cooling down %.0fs (%s)",
                endpoint.key, cooldown, bucket.cooldown_reason,
            )
            return cooldown

    def record_unavailable(
        self, endpoint: Endpoint, cooldown: float, *, reason: str
    ) -> None:
        """Park an endpoint for non-quota reasons (auth failure, 5xx, timeout)."""
        with self._lock:
            now = self._now()
            bucket = self._bucket(endpoint)
            bucket.cooldown_until = max(
                bucket.cooldown_until, now + min(cooldown, MAX_COOLDOWN_SECONDS)
            )
            bucket.cooldown_reason = reason

    def sync_from_headers(
        self,
        endpoint: Endpoint,
        *,
        limit_requests: int | None = None,
        remaining_requests: int | None = None,
        limit_tokens: int | None = None,
        remaining_tokens: int | None = None,
        window: int = 60,
    ) -> None:
        """Correct the ledger from a provider's own rate-limit headers.

        Groq reports x-ratelimit-remaining-* on every response, but a single
        header pair does not say which window it is counting: in practice Groq
        sends the account's *daily* requests-limit under the same header name a
        per-minute limit would use, and its value (e.g. 1000) has nothing to do
        with a 60-second window. Assuming `window` blindly would then overwrite
        the correctly-configured per-minute counter with a daily figure - making
        the ledger think there is far more per-minute headroom than truly
        exists, which is the unsafe direction (under-counting risks a 429
        instead of costing a harmless fallback hop).

        Instead, the reported limit is matched against this endpoint's own
        configured limits (from limits.yaml) and assigned to whichever window's
        configured value it actually equals. `window` is only a fallback for
        endpoints with no configured limit of that kind at all.
        """
        with self._lock:
            now = self._now()
            bucket = self._bucket(endpoint)
            pairs = (
                (REQUESTS, limit_requests, remaining_requests),
                (TOKENS, limit_tokens, remaining_tokens),
            )
            for kind, limit_value, remaining in pairs:
                if limit_value is None or remaining is None:
                    continue
                target_window = _match_window(endpoint, kind, int(limit_value), window)
                counter = bucket.counters.get((kind, target_window))
                if counter is None:
                    counter = _Counter(window=target_window)
                    bucket.counters[(kind, target_window)] = counter
                counter.display_limit = max(counter.display_limit, int(limit_value))
                counter.asserted_used = max(0, int(limit_value) - int(remaining))
                counter.asserted_until = now + target_window

    # -- persistence & inspection ------------------------------------------- #

    def snapshot(self) -> dict[str, Any]:
        """Human-readable view of every bucket. For logging and debugging."""
        with self._lock:
            now = self._now()
            out: dict[str, Any] = {}
            for key, bucket in self._buckets.items():
                counters = {}
                for (kind, window), counter in bucket.counters.items():
                    counters[f"{kind}/{window_name(window)}"] = {
                        "used": counter.used(now),
                        "limit": counter.display_limit,
                        "remaining": counter.remaining(now, counter.display_limit),
                    }
                out[key] = {
                    "counters": counters,
                    "calls": bucket.calls,
                    "rate_limit_events": bucket.rate_limit_events,
                    "cooling_down_for": max(0.0, bucket.cooldown_until - now) or None,
                    "cooldown_reason": (
                        bucket.cooldown_reason if now < bucket.cooldown_until else None
                    ),
                }
            return out

    def to_dict(self) -> dict[str, Any]:
        """Serialisable state, for surviving a process restart.

        Daily and weekly counters are the reason this exists: a fresh process
        that forgets it already spent today's 20 Gemini requests will spend them
        again as 429s.
        """
        with self._lock:
            now = self._now()
            return {
                "version": 1,
                "saved_at": now,
                "buckets": {
                    key: {
                        "cooldown_until": bucket.cooldown_until,
                        "cooldown_reason": bucket.cooldown_reason,
                        "last_call_at": bucket.last_call_at,
                        "calls": bucket.calls,
                        "rate_limit_events": bucket.rate_limit_events,
                        "counters": {
                            f"{kind}|{window}": {
                                "limit": counter.display_limit,
                                "events": list(counter.events),
                            }
                            for (kind, window), counter in bucket.counters.items()
                        },
                    }
                    for key, bucket in self._buckets.items()
                },
            }

    def load_dict(self, data: Mapping[str, Any]) -> None:
        """Restore state written by to_dict(). Expired events are dropped."""
        if not data or data.get("version") != 1:
            return
        with self._lock:
            now = self._now()
            for key, raw in (data.get("buckets") or {}).items():
                bucket = self._buckets.setdefault(key, _Bucket())
                bucket.cooldown_until = max(
                    bucket.cooldown_until, float(raw.get("cooldown_until") or 0.0)
                )
                bucket.cooldown_reason = raw.get("cooldown_reason")
                bucket.last_call_at = max(
                    bucket.last_call_at, float(raw.get("last_call_at") or 0.0)
                )
                bucket.calls += int(raw.get("calls") or 0)
                bucket.rate_limit_events += int(raw.get("rate_limit_events") or 0)
                for counter_key, raw_counter in (raw.get("counters") or {}).items():
                    kind, _, window_text = counter_key.partition("|")
                    window = int(window_text)
                    counter = bucket.counters.get((kind, window))
                    if counter is None:
                        counter = _Counter(
                            window=window,
                            display_limit=int(raw_counter.get("limit", 0)),
                        )
                        bucket.counters[(kind, window)] = counter
                    for timestamp, amount in raw_counter.get("events") or ():
                        if now - float(timestamp) < window:
                            counter.add(float(timestamp), int(amount))
                    counter.prune(now)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict()), encoding="utf-8")
        tmp.replace(path)

    def load(self, path: str | Path) -> None:
        path = Path(path)
        if not path.exists():
            return
        try:
            self.load_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError) as exc:
            logger.warning("could not restore ledger from %s: %s", path, exc)

    def reset(self, endpoints: Iterable[Endpoint] | None = None) -> None:
        """Clear everything, or just the given endpoints' buckets."""
        with self._lock:
            if endpoints is None:
                self._buckets.clear()
                return
            for endpoint in endpoints:
                self._buckets.pop(endpoint.ledger_key, None)
