"""Quota windows, persistence and multi-process safety.

Window arithmetic is where a quota tracker quietly goes wrong: a daily tier that
never resets, or a monthly one that resets on the wrong day, silently costs the
user their free capacity.
"""

from __future__ import annotations

import json
import multiprocessing
import os

import pytest
from freezegun import freeze_time

from searchroute.ledger import JSONFileStore, MemoryStore
from searchroute.quota import (
    Anchor,
    Period,
    QuotaPolicy,
    QuotaState,
    QuotaTracker,
)


def tracker(policy: QuotaPolicy, name="p", store=None, reserve=0.0) -> QuotaTracker:
    t = QuotaTracker(store=store or MemoryStore(), reserve_pct=reserve)
    t.register(name, policy)
    return t


class TestWindows:
    def test_daily_window_resets_at_utc_midnight(self):
        policy = QuotaPolicy(limit=100, period=Period.DAILY)
        t = tracker(policy)

        with freeze_time("2026-09-07 23:00:00"):
            t.debit("p", 100)
            assert t.remaining("p") == 0

        with freeze_time("2026-09-08 00:30:00"):
            assert t.remaining("p") == 100, "a daily tier must refill after midnight"

    def test_daily_window_does_not_reset_within_the_day(self):
        policy = QuotaPolicy(limit=100, period=Period.DAILY)
        t = tracker(policy)

        with freeze_time("2026-09-07 01:00:00"):
            t.debit("p", 40)
        with freeze_time("2026-09-07 23:59:00"):
            assert t.remaining("p") == 60

    def test_monthly_calendar_window(self):
        policy = QuotaPolicy(limit=1000, period=Period.MONTHLY, anchor=Anchor.CALENDAR)
        t = tracker(policy)

        with freeze_time("2026-09-20 12:00:00"):
            t.debit("p", 1000)
            assert t.remaining("p") == 0

        with freeze_time("2026-10-01 00:01:00"):
            assert t.remaining("p") == 1000

    def test_monthly_signup_day_anchor(self):
        """A tier that renews on the 15th must not reset on the 1st."""
        policy = QuotaPolicy(
            limit=1000, period=Period.MONTHLY, anchor=Anchor.SIGNUP_DAY, anchor_day=15
        )
        t = tracker(policy)

        with freeze_time("2026-09-20 12:00:00"):
            t.debit("p", 1000)
            assert t.remaining("p") == 0

        with freeze_time("2026-10-01 12:00:00"):
            assert t.remaining("p") == 0, "the 1st is not this tier's reset day"

        with freeze_time("2026-10-15 12:00:00"):
            assert t.remaining("p") == 1000

    def test_signup_anchor_before_anchor_day_uses_previous_month(self):
        policy = QuotaPolicy(
            limit=100, period=Period.MONTHLY, anchor=Anchor.SIGNUP_DAY, anchor_day=20
        )
        with freeze_time("2026-09-05 12:00:00"):
            start = policy.window_start()
            assert start.month == 8 and start.day == 20

    def test_signup_anchor_clamps_to_short_months(self):
        """Signing up on the 31st must not break in February."""
        policy = QuotaPolicy(
            limit=100, period=Period.MONTHLY, anchor=Anchor.SIGNUP_DAY, anchor_day=31
        )
        with freeze_time("2026-02-15 12:00:00"):
            start = policy.window_start()
            assert start.month == 1 and start.day == 31

    def test_one_time_grant_never_refills(self):
        policy = QuotaPolicy(limit=2500, period=Period.ONE_TIME)
        t = tracker(policy)

        with freeze_time("2026-09-07 12:00:00"):
            t.debit("p", 2500)
        with freeze_time("2027-01-01 12:00:00"):
            assert t.remaining("p") == 0, "a one-time grant does not renew"

    def test_unmetered_policy_is_never_limited(self):
        t = QuotaTracker(store=MemoryStore())
        t.register("free", None)
        assert t.remaining("free") is None
        assert t.can_afford("free", 1_000_000)


class TestReserve:
    def test_reserve_reduces_spendable_capacity(self):
        t = tracker(QuotaPolicy(limit=1000, period=Period.MONTHLY), reserve=0.05)
        assert t.remaining("p") == 950

    def test_provider_is_skipped_once_reserve_is_reached(self):
        t = tracker(QuotaPolicy(limit=100, period=Period.MONTHLY), reserve=0.1)
        t.debit("p", 90)
        assert t.remaining("p") == 0
        assert t.can_afford("p", 1) is False


class TestProviderTruthWins:
    def test_mark_exhausted_overrides_local_count(self):
        """The provider's own 402 beats whatever our ledger believed."""
        t = tracker(QuotaPolicy(limit=1000, period=Period.MONTHLY))
        assert t.remaining("p") == 1000

        t.mark_exhausted("p")

        assert t.remaining("p") == 0

    def test_unmetered_provider_refusing_us_gets_a_cooldown(self):
        t = QuotaTracker(store=MemoryStore())
        t.register("free", None)

        with freeze_time("2026-09-07 12:00:00"):
            t.mark_exhausted("free")
            assert t.is_disabled("free") is True
        with freeze_time("2026-09-07 13:30:00"):
            assert t.is_disabled("free") is False, "cooldown should expire"

    def test_fraction_remaining_compares_tiers_fairly(self):
        """A 100/day tier and a 1000/month tier are compared by proportion."""
        t = QuotaTracker(store=MemoryStore(), reserve_pct=0.0)
        t.register("small", QuotaPolicy(limit=100, period=Period.DAILY))
        t.register("large", QuotaPolicy(limit=1000, period=Period.MONTHLY))

        t.debit("small", 50)
        t.debit("large", 900)

        assert t.fraction_remaining("small") == pytest.approx(0.5)
        assert t.fraction_remaining("large") == pytest.approx(0.1)


class TestJSONFileStore:
    def test_round_trips_state(self, tmp_path):
        store = JSONFileStore(tmp_path / "quota.json")
        store.set("exa", QuotaState(used=42, window_start="2026-09-01T00:00:00+00:00"))

        reloaded = JSONFileStore(tmp_path / "quota.json")
        assert reloaded.get("exa").used == 42

    def test_survives_a_corrupt_file(self, tmp_path):
        """Quota tracking is advisory: a mangled ledger must not take the app down."""
        path = tmp_path / "quota.json"
        path.write_text("{not valid json")

        store = JSONFileStore(path)
        assert store.get("exa").used == 0

        store.set("exa", QuotaState(used=5))
        assert store.get("exa").used == 5

    def test_writes_are_atomic(self, tmp_path):
        path = tmp_path / "quota.json"
        store = JSONFileStore(path)
        store.set("a", QuotaState(used=1))
        store.set("b", QuotaState(used=2))

        data = json.loads(path.read_text())
        assert set(data["providers"]) == {"a", "b"}
        assert not list(tmp_path.glob("*.tmp")), "temp files should be cleaned up"

    def test_persists_across_tracker_instances(self, tmp_path):
        path = tmp_path / "quota.json"
        policy = QuotaPolicy(limit=100, period=Period.MONTHLY)

        with freeze_time("2026-09-07 12:00:00"):
            first = tracker(policy, store=JSONFileStore(path))
            first.debit("p", 30)

            # A fresh process would build a new tracker over the same file.
            second = tracker(policy, store=JSONFileStore(path))
            assert second.remaining("p") == 70


def _hammer(args):
    """Debit once from a separate process."""
    path, n = args
    from searchroute.ledger import JSONFileStore
    from searchroute.quota import Period, QuotaPolicy, QuotaTracker

    t = QuotaTracker(store=JSONFileStore(path), reserve_pct=0.0)
    t.register("p", QuotaPolicy(limit=10_000, period=Period.MONTHLY))
    for _ in range(n):
        t.debit("p", 1)
    return True


class TestConcurrency:
    @pytest.mark.skipif(os.name == "nt", reason="advisory locking is POSIX-only here")
    def test_concurrent_processes_do_not_lose_debits(self, tmp_path):
        """Several workers sharing one machine must converge on one count."""
        path = str(tmp_path / "quota.json")
        processes, per_process = 4, 25

        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(processes) as pool:
            pool.map(_hammer, [(path, per_process)] * processes)

        t = QuotaTracker(store=JSONFileStore(path), reserve_pct=0.0)
        t.register("p", QuotaPolicy(limit=10_000, period=Period.MONTHLY))
        used = 10_000 - t.remaining("p")
        assert used == processes * per_process, f"lost debits: recorded {used}"
