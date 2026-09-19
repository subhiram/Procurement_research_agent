"""UsageLedger tests, checked against the real numbers in config/limits.yaml."""

import pytest

from llm_router.ledger import UsageLedger
from llm_router.registry import REQUESTS, TOKENS, Endpoint, Limit


def endpoint_for(registry, logical_model, provider):
    for endpoint in registry.get_endpoints(logical_model):
        if endpoint.provider == provider:
            return endpoint
    raise AssertionError(f"no {provider} endpoint for {logical_model}")


@pytest.fixture
def groq_120b(registry):
    return endpoint_for(registry, "gpt-oss-120b", "groq")


@pytest.fixture
def gemini_flash(registry):
    """Google's newest plain Flash: the tightest allowance in the config."""
    return endpoint_for(registry, "gemini-flash", "google_ai_studio")


# --------------------------------------------------------------------------- #
# config wiring
# --------------------------------------------------------------------------- #

def test_limits_come_from_yaml(groq_120b, gemini_flash):
    groq_limits = {(l.kind, l.window): l.value for l in groq_120b.limits}
    assert groq_limits[(REQUESTS, 60)] == 30      # rpm: 30
    assert groq_limits[(REQUESTS, 86400)] == 1000  # rpd: 1000
    assert groq_limits[(TOKENS, 60)] == 8000       # tpm: 8000
    assert groq_limits[(TOKENS, 86400)] == 200000  # tpd: 200000

    flash_limits = {(l.kind, l.window): l.value for l in gemini_flash.limits}
    assert gemini_flash.model_id == "gemini-3.8-flash"
    assert flash_limits[(REQUESTS, 60)] == 5         # rpm: 5
    assert flash_limits[(REQUESTS, 86400)] == 20     # rpd: 20
    assert flash_limits[(TOKENS, 60)] == 250000      # tpm: 250000


def test_ledger_key_follows_quota_scope(registry):
    """per_model providers get one bucket each; per_project pools them."""
    groq_a = endpoint_for(registry, "gpt-oss-120b", "groq")
    groq_b = endpoint_for(registry, "gpt-oss-20b", "groq")
    assert groq_a.ledger_key != groq_b.ledger_key

    gemini_flash = endpoint_for(registry, "gemini-flash", "google_ai_studio")
    gemma_31b = endpoint_for(registry, "gemma-31b", "google_ai_studio")
    assert gemini_flash.ledger_key == gemma_31b.ledger_key == "google_ai_studio:*"


def test_google_quota_is_shared_across_models(registry, clock):
    """Spending Google quota on one model really does count against another -
    it is per project, not per model.

    gemini-flash's own ceiling (rpd 20) is far tighter than gemma-31b's
    (rpd 14400), so calls only need to be driven far enough to prove the
    *usage* is shared, not all the way to gemma's own limit.
    """
    ledger = UsageLedger(time_fn=clock)
    flash = endpoint_for(registry, "gemini-flash", "google_ai_studio")
    gemma = endpoint_for(registry, "gemma-31b", "google_ai_studio")

    for i in range(5):
        if i:
            clock.advance(13)               # clear of flash's 12s min_interval
        assert ledger.try_acquire(flash).ok, f"blocked at call {i}"

    clock.advance(13)
    gemma_snapshot = ledger.snapshot()[gemma.ledger_key]["counters"]["requests/day"]
    assert gemma_snapshot["used"] == 5      # gemma's own counter sees flash's spend
    assert ledger.can_call(gemma)           # ...but 5 is nowhere near its 14400 ceiling


# --------------------------------------------------------------------------- #
# rolling windows
# --------------------------------------------------------------------------- #

def test_requests_per_minute_is_a_rolling_window(groq_120b, clock):
    ledger = UsageLedger(time_fn=clock)
    for _ in range(30):                     # groq rpm: 30
        assert ledger.try_acquire(groq_120b).ok
    assert not ledger.can_call(groq_120b)

    verdict = ledger.availability(groq_120b)
    assert verdict.retry_after == pytest.approx(60.0, abs=0.01)

    clock.advance(59.9)
    assert not ledger.can_call(groq_120b)
    clock.advance(0.2)                      # oldest call ages out
    assert ledger.can_call(groq_120b)


def test_daily_window_survives_minute_rollover(registry, clock):
    """Exhausting rpd must keep blocking long after the minute window clears."""
    ledger = UsageLedger(time_fn=clock)
    gemini = endpoint_for(registry, "gemini-flash-lite", "google_ai_studio")
    # gemini-3.5-flash-lite: rpm 15, rpd 500, min_interval 4.0s
    for i in range(500):
        assert ledger.try_acquire(gemini).ok, f"blocked at call {i}"
        clock.advance(4.1)                  # respect rpm and min_interval

    assert not ledger.can_call(gemini)
    verdict = ledger.availability(gemini)
    assert verdict.reason and "day" in verdict.reason
    clock.advance(3600)                     # an hour later, still out of daily quota
    assert not ledger.can_call(gemini)


def test_token_counter_blocks_and_recovers(groq_120b, clock):
    ledger = UsageLedger(time_fn=clock)
    ledger.record_call(groq_120b, tokens_used=7900)   # tpm: 8000
    assert ledger.can_call(groq_120b, estimated_tokens=100)
    assert not ledger.can_call(groq_120b, estimated_tokens=101)

    clock.advance(61)
    assert ledger.can_call(groq_120b, estimated_tokens=8000)


def test_record_tokens_after_acquire(groq_120b, clock):
    """try_acquire reserves the request; the token spend lands afterwards."""
    ledger = UsageLedger(time_fn=clock)
    assert ledger.try_acquire(groq_120b).ok
    ledger.record_tokens(groq_120b, 8000)
    assert not ledger.can_call(groq_120b, estimated_tokens=1)
    snapshot = ledger.snapshot()[groq_120b.ledger_key]
    assert snapshot["counters"]["tokens/minute"]["used"] == 8000
    assert snapshot["counters"]["requests/minute"]["used"] == 1


# --------------------------------------------------------------------------- #
# pre-emptive spacing
# --------------------------------------------------------------------------- #

def test_min_interval_spaces_tight_providers(gemini_flash, clock):
    """gemini-3.8-flash is 5 rpm, so calls are spaced 12s apart."""
    assert gemini_flash.min_interval == 12.0
    ledger = UsageLedger(time_fn=clock)
    assert ledger.try_acquire(gemini_flash).ok

    verdict = ledger.availability(gemini_flash)
    assert not verdict.ok and "spacing" in verdict.reason
    assert verdict.retry_after == pytest.approx(12.0)

    clock.advance(12.1)
    assert ledger.can_call(gemini_flash)


# --------------------------------------------------------------------------- #
# 429 handling
# --------------------------------------------------------------------------- #

def test_record_rate_limited_honours_retry_after(groq_120b, clock):
    ledger = UsageLedger(time_fn=clock)
    ledger.record_rate_limited(groq_120b, retry_after=8.0)
    assert not ledger.can_call(groq_120b)
    assert ledger.availability(groq_120b).retry_after == pytest.approx(8.0)
    clock.advance(8.1)
    assert ledger.can_call(groq_120b)


def test_record_rate_limited_without_retry_after_uses_provider_cooldown(groq_120b, clock):
    ledger = UsageLedger(time_fn=clock)
    cooldown = ledger.record_rate_limited(groq_120b)
    assert cooldown == groq_120b.cooldown_seconds == 60.0
    clock.advance(59)
    assert not ledger.can_call(groq_120b)
    clock.advance(2)
    assert ledger.can_call(groq_120b)


def test_absurd_retry_after_is_clamped(groq_120b, clock):
    ledger = UsageLedger(time_fn=clock)
    assert ledger.record_rate_limited(groq_120b, retry_after=10**9) == 3600.0


def test_rate_limit_corrects_optimistic_config(groq_120b, clock):
    """limits.yaml says 30 rpm; if the provider 429s at 3, the ledger believes it."""
    ledger = UsageLedger(time_fn=clock)
    for _ in range(3):
        assert ledger.try_acquire(groq_120b).ok
    ledger.record_rate_limited(groq_120b, retry_after=30.0)
    assert not ledger.can_call(groq_120b)


# --------------------------------------------------------------------------- #
# header sync
# --------------------------------------------------------------------------- #

def test_sync_from_headers_outranks_local_count(groq_120b, clock):
    """Another process shares this key; Groq's headers are the truth."""
    ledger = UsageLedger(time_fn=clock)
    assert ledger.try_acquire(groq_120b).ok            # local count says 1 used
    ledger.sync_from_headers(
        groq_120b, limit_requests=30, remaining_requests=0, window=60
    )
    assert not ledger.can_call(groq_120b)

    clock.advance(61)                                   # assertion expires with its window
    assert ledger.can_call(groq_120b)


def test_sync_from_headers_still_counts_new_calls(groq_120b, clock):
    ledger = UsageLedger(time_fn=clock)
    ledger.sync_from_headers(
        groq_120b, limit_requests=30, remaining_requests=2, window=60
    )
    assert ledger.try_acquire(groq_120b).ok
    assert ledger.try_acquire(groq_120b).ok
    assert not ledger.can_call(groq_120b)


# --------------------------------------------------------------------------- #
# concurrency & persistence
# --------------------------------------------------------------------------- #

def test_try_acquire_is_atomic_under_threads(clock):
    """20 threads racing for 5 rpm must not all win."""
    import threading

    endpoint = Endpoint(
        provider="test", model_id="m", logical_model="m", tier="A",
        limits=(Limit(kind=REQUESTS, window=60, value=5),),
    )
    ledger = UsageLedger(time_fn=clock)
    wins: list[bool] = []
    lock = threading.Lock()
    barrier = threading.Barrier(20)

    def attempt() -> None:
        barrier.wait()
        ok = ledger.try_acquire(endpoint).ok
        with lock:
            wins.append(ok)

    threads = [threading.Thread(target=attempt) for _ in range(20)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sum(wins) == 5


def test_state_survives_a_restart(tmp_path, registry, clock):
    """A fresh process must not respend today's Gemini quota."""
    gemini = endpoint_for(registry, "gemini-flash", "google_ai_studio")
    first = UsageLedger(time_fn=clock)
    for i in range(5):
        if i:
            clock.advance(12.1)
        assert first.try_acquire(gemini).ok
    path = tmp_path / "ledger.json"
    first.save(path)

    second = UsageLedger(time_fn=clock)
    assert second.can_call(gemini)      # empty ledger would happily spend again
    second.load(path)
    assert not second.can_call(gemini)


def test_load_drops_expired_events(tmp_path, groq_120b, clock):
    ledger = UsageLedger(time_fn=clock)
    for _ in range(30):
        ledger.try_acquire(groq_120b)
    path = tmp_path / "ledger.json"
    ledger.save(path)

    clock.advance(120)                  # the minute window has long passed
    restored = UsageLedger(time_fn=clock)
    restored.load(path)
    assert restored.can_call(groq_120b)
    used = restored.snapshot()[groq_120b.ledger_key]["counters"]
    assert used["requests/minute"]["used"] == 0
    assert used["requests/day"]["used"] == 30   # but the daily count survives


def test_a_shared_bucket_is_judged_by_each_models_own_limit(registry, clock):
    """Usage is shared; ceilings are not.

    Google pools quota per project, but flash-lite is allowed 500 requests a day
    where plain flash gets 20. Enforcing one ceiling across the bucket would
    throw away most of flash-lite's free tier, so each endpoint is checked
    against its own limits over the shared usage.
    """
    ledger = UsageLedger(time_fn=clock)
    lite = endpoint_for(registry, "gemini-flash-lite", "google_ai_studio")
    flash = endpoint_for(registry, "gemini-flash", "google_ai_studio")
    assert lite.ledger_key == flash.ledger_key

    for i in range(20):                 # spends flash's entire daily allowance
        if i:
            clock.advance(15)
        assert ledger.try_acquire(lite).ok

    clock.advance(15)
    assert not ledger.can_call(flash)   # flash is done for the day...
    assert ledger.can_call(lite)        # ...but flash-lite has 480 left


# --------------------------------------------------------------------------- #
# header window matching (regression: Groq reports its DAILY limit under the
# same header name a per-minute limit would use)
# --------------------------------------------------------------------------- #

def test_header_sync_matches_the_daily_limit_by_value_not_by_guessing(groq_120b, clock):
    """Groq's real x-ratelimit-limit-requests on this model is 1000 - which is
    the endpoint's configured *daily* limit (rpd), not a per-minute figure.
    Blindly writing it into a 60s bucket would make the ledger think the
    correctly-configured 30 rpm counter has 1000 rpm of headroom - the unsafe,
    under-counting direction. It must land on the daily counter instead."""
    ledger = UsageLedger(time_fn=clock)
    ledger.record_call(groq_120b)   # one real local call this minute
    ledger.sync_from_headers(
        groq_120b, limit_requests=1000, remaining_requests=992, window=60
    )

    counters = ledger.snapshot()[groq_120b.ledger_key]["counters"]
    assert counters["requests/minute"]["limit"] == 30       # untouched
    assert counters["requests/minute"]["used"] == 1         # local count, not 8
    assert counters["requests/day"]["limit"] == 1000
    assert counters["requests/day"]["used"] == 8             # 1000 - 992


def test_header_sync_still_uses_the_window_argument_with_no_configured_limits(clock):
    """An endpoint with no limits.yaml entry for this kind has nothing to match
    against, so the explicit window argument is the only signal available."""
    endpoint = Endpoint(
        provider="test", model_id="m", logical_model="m", tier="A", limits=(),
    )
    ledger = UsageLedger(time_fn=clock)
    ledger.sync_from_headers(endpoint, limit_requests=500, remaining_requests=10, window=60)
    counters = ledger.snapshot()[endpoint.ledger_key]["counters"]
    assert counters["requests/minute"]["limit"] == 500


def test_header_sync_falls_back_to_the_loosest_window_when_value_is_unrecognised(
    groq_120b, clock
):
    """The account's real limit no longer matches limits.yaml at all - the
    safe guess is the endpoint's largest configured window, not its tightest,
    since applying a big number to a short window is what caused a false
    all-clear in the first place."""
    ledger = UsageLedger(time_fn=clock)
    ledger.sync_from_headers(
        groq_120b, limit_requests=5000, remaining_requests=4990, window=60
    )
    counters = ledger.snapshot()[groq_120b.ledger_key]["counters"]
    assert counters["requests/minute"]["limit"] == 30        # untouched
    assert counters["requests/day"]["limit"] == 5000          # landed on rpd
