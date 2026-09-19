"""Fallback ladder: order, tier discipline, and downgrade flagging."""

import pytest

from llm_router.ladder import (
    STEP_SAME_MODEL,
    STEP_SAME_TIER_ANY_PROVIDER,
    STEP_SAME_TIER_SAME_PROVIDER,
    STEP_TIER_DOWNGRADE,
    resolve_candidates,
    soonest_retry,
)
from llm_router.ledger import UsageLedger

pytestmark = pytest.mark.usefixtures("all_keys")


def keys(candidates):
    return [c.endpoint.key for c in candidates]


def exhaust(ledger, endpoint, *, requests=1000):
    """Burn an endpoint's minute quota without waiting for the clock."""
    ledger.record_rate_limited(endpoint, retry_after=45.0)


def endpoint_for(registry, model, provider):
    return next(e for e in registry.get_endpoints(model) if e.provider == provider)


# --------------------------------------------------------------------------- #
# ladder shape
# --------------------------------------------------------------------------- #

def test_step_1_is_the_requested_model_on_every_provider(registry):
    """gemma-31b is the most widely hosted model in the config: three providers."""
    candidates = resolve_candidates("gemma-31b", registry=registry)
    step_1 = [c for c in candidates if c.step == STEP_SAME_MODEL]
    assert keys(step_1) == [
        "nvidia_nim:google/gemma-4-31b-it",
        "google_ai_studio:gemma-4-31b-it",
        "openrouter:google/gemma-4-31b-it:free",
    ]
    assert all(not c.tier_downgraded for c in step_1)


def test_step_1_respects_configured_priority(registry):
    """nvidia_nim is priority 40, google 50, openrouter 60 - in that order."""
    candidates = resolve_candidates("gemma-31b", registry=registry)
    assert [c.endpoint.provider for c in candidates[:3]] == [
        "nvidia_nim", "google_ai_studio", "openrouter",
    ]


def test_step_2_prefers_a_provider_that_hosts_the_requested_model(registry):
    """gemma-31b lives on nvidia_nim; nvidia_nim also has tier-A nemotron-super."""
    candidates = resolve_candidates("gemma-31b", registry=registry)
    step_2 = [c for c in candidates if c.step == STEP_SAME_TIER_SAME_PROVIDER]
    assert "nvidia_nim:nvidia/nemotron-3-super-120b-a12b" in keys(step_2)
    # and it comes before the tier-A models on providers that do not host qwen
    step_3 = [c for c in candidates if c.step == STEP_SAME_TIER_ANY_PROVIDER]
    assert keys(step_2) and keys(step_3)
    assert candidates.index(step_2[0]) < candidates.index(step_3[0])


def test_same_tier_steps_are_never_marked_as_downgrades(registry):
    candidates = resolve_candidates("qwen-27b", registry=registry)
    for candidate in candidates:
        if candidate.step in (STEP_SAME_TIER_SAME_PROVIDER, STEP_SAME_TIER_ANY_PROVIDER):
            assert candidate.endpoint.tier == "A"
            assert not candidate.tier_downgraded


def test_downgrades_are_flagged_and_come_last(registry):
    candidates = resolve_candidates("gpt-oss-120b", registry=registry)
    downgrades = [c for c in candidates if c.tier_downgraded]
    assert downgrades, "expected tier-A/B fallbacks below tier S"
    assert all(c.step == STEP_TIER_DOWNGRADE for c in downgrades)
    assert all(c.endpoint.tier in ("A", "B") for c in downgrades)
    first_downgrade = candidates.index(downgrades[0])
    assert all(not c.tier_downgraded for c in candidates[:first_downgrade])


def test_downgrade_order_is_a_before_b(registry):
    candidates = resolve_candidates("gpt-oss-120b", registry=registry)
    tiers = [c.endpoint.tier for c in candidates if c.tier_downgraded]
    assert tiers == sorted(tiers)      # "A" before "B"


def test_no_endpoint_appears_twice(registry):
    candidates = resolve_candidates("qwen-27b", registry=registry)
    assert len(keys(candidates)) == len(set(keys(candidates)))


def test_candidate_reason_names_the_downgrade(registry):
    candidates = resolve_candidates("gpt-oss-120b", registry=registry)
    downgrade = next(c for c in candidates if c.tier_downgraded)
    assert "S -> " + downgrade.endpoint.tier in downgrade.reason


# --------------------------------------------------------------------------- #
# the scenario from the build spec
# --------------------------------------------------------------------------- #

def test_the_requested_model_dying_holds_the_tier_rather_than_downgrading(
    registry, clock
):
    """gpt-oss-120b gone -> a tier-S *peer*, not a downgrade.

    Losing every host of the requested model is not the same as losing the
    tier. Tier S has several single-host models across NVIDIA NIM and
    OpenRouter, so the ladder must spend all of those at step 3 before it is
    entitled to flag anything as a downgrade.
    """
    ledger = UsageLedger(time_fn=clock)
    exhaust(ledger, endpoint_for(registry, "gpt-oss-120b", "groq"))

    candidates = resolve_candidates("gpt-oss-120b", ledger=ledger, registry=registry)

    assert "groq:openai/gpt-oss-120b" not in keys(candidates)
    assert candidates[0].endpoint.tier == "S"
    assert candidates[0].step == STEP_SAME_TIER_ANY_PROVIDER
    assert not candidates[0].tier_downgraded
    # every remaining tier-S option is offered before any downgrade is
    first_downgrade = next(i for i, c in enumerate(candidates) if c.tier_downgraded)
    assert all(c.endpoint.tier == "S" for c in candidates[:first_downgrade])


def test_all_tier_s_exhausted_yields_only_flagged_downgrades(registry, clock):
    ledger = UsageLedger(time_fn=clock)
    for model in registry.models_in_tier("S"):
        for endpoint in registry.get_endpoints(model):
            exhaust(ledger, endpoint)

    candidates = resolve_candidates("gpt-oss-120b", ledger=ledger, registry=registry)
    assert candidates, "tier A and B should still be reachable"
    assert all(c.tier_downgraded for c in candidates)
    assert all(c.endpoint.tier != "S" for c in candidates)


def test_everything_exhausted_returns_nothing(registry, clock):
    ledger = UsageLedger(time_fn=clock)
    for endpoint in registry.all_endpoints():
        exhaust(ledger, endpoint)
    assert resolve_candidates("gpt-oss-120b", ledger=ledger, registry=registry) == []


def test_recovery_after_cooldown(registry, clock):
    ledger = UsageLedger(time_fn=clock)
    groq = endpoint_for(registry, "gpt-oss-120b", "groq")
    ledger.record_rate_limited(groq, retry_after=45.0)
    assert "groq:openai/gpt-oss-120b" not in keys(
        resolve_candidates("gpt-oss-120b", ledger=ledger, registry=registry)
    )
    clock.advance(46)
    assert keys(resolve_candidates("gpt-oss-120b", ledger=ledger, registry=registry))[0] == (
        "groq:openai/gpt-oss-120b"
    )


# --------------------------------------------------------------------------- #
# diagnostics
# --------------------------------------------------------------------------- #

def test_include_unavailable_keeps_blocked_candidates_with_reasons(registry, clock):
    ledger = UsageLedger(time_fn=clock)
    groq = endpoint_for(registry, "gpt-oss-120b", "groq")
    ledger.record_rate_limited(groq, retry_after=45.0)

    candidates = resolve_candidates(
        "gpt-oss-120b", ledger=ledger, registry=registry, include_unavailable=True
    )
    blocked = next(c for c in candidates if c.endpoint.key == "groq:openai/gpt-oss-120b")
    assert not blocked.available
    assert blocked.availability.retry_after == pytest.approx(45.0)
    assert soonest_retry(candidates) == pytest.approx(45.0)


# --------------------------------------------------------------------------- #
# forcing and filtering
# --------------------------------------------------------------------------- #

def test_forcing_a_provider_restricts_the_whole_ladder(registry):
    candidates = resolve_candidates("qwen-27b", registry=registry, providers=["groq"])
    assert {c.endpoint.provider for c in candidates} == {"groq"}


def test_tier_floor_without_a_model_searches_that_tier_first(registry):
    candidates = resolve_candidates(None, "S", registry=registry)
    assert all(c.endpoint.tier == "S" for c in candidates if not c.tier_downgraded)
    assert candidates[0].endpoint.tier == "S"
    assert candidates[0].step == STEP_SAME_TIER_ANY_PROVIDER


def test_tier_floor_can_refuse_to_downgrade(registry):
    candidates = resolve_candidates(
        None, "S", registry=registry, allow_tier_downgrade=False
    )
    assert {c.endpoint.tier for c in candidates} == {"S"}


def test_explicit_tier_overrides_the_models_own_tier(registry):
    """route(model='gpt-oss-20b', tier='S') means: tier S is the floor."""
    candidates = resolve_candidates("gpt-oss-20b", "S", registry=registry)
    first = candidates[0]
    assert first.endpoint.key == "groq:openai/gpt-oss-20b"
    # ...but asking for a tier-B model under an S floor is honestly a downgrade
    assert first.tier_downgraded and first.requested_tier == "S"


def test_no_model_and_no_tier_walks_tiers_without_crying_downgrade(registry):
    """With nothing requested, walk S -> A -> B; no quality was ever promised."""
    from llm_router.registry import tier_rank

    candidates = resolve_candidates(registry=registry)
    ranks = [tier_rank(c.endpoint.tier) for c in candidates]
    assert ranks == sorted(ranks)
    assert candidates[0].endpoint.tier == "S"
    assert not any(c.tier_downgraded for c in candidates)


def test_streaming_filters_out_non_streaming_endpoints(registry, monkeypatch):
    """Cloudflare's wrapper cannot stream; such endpoints must never be chosen."""
    endpoint = next(iter(registry.get_endpoints("gpt-oss-120b")))
    object.__setattr__(endpoint, "supports_streaming", False)
    try:
        streamed = resolve_candidates("gpt-oss-120b", registry=registry, streaming=True)
        assert endpoint.key not in keys(streamed)
        assert endpoint.key in keys(resolve_candidates("gpt-oss-120b", registry=registry))
    finally:
        object.__setattr__(endpoint, "supports_streaming", True)


def test_endpoints_without_credentials_are_skipped(registry, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    candidates = resolve_candidates("gpt-oss-20b", registry=registry)
    assert "groq:openai/gpt-oss-20b" not in keys(candidates)
    assert "nvidia_nim:openai/gpt-oss-20b" in keys(candidates)


def test_a_disabled_provider_is_invisible_even_with_a_key(registry, monkeypatch):
    """Cloudflare is `enabled: false`; a key must not bring it back."""
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "test-key")
    assert not registry.is_usable("cloudflare")
    assert all(
        endpoint.provider != "cloudflare" for endpoint in registry.all_endpoints()
    )


def test_a_model_that_cannot_do_structured_output_is_hidden_from_those_calls(
    registry, all_keys
):
    """Groq's allam-2-7b rejects both json_schema and tool calling with a 400.

    For a caller whose every request carries a schema it is not a fallback but a
    guaranteed failure, so it must leave the ladder rather than be discovered by
    burning an attempt on it.
    """
    plain = registry.enabled_endpoints(tier="B")
    structured = registry.enabled_endpoints(tier="B", structured=True)

    assert any(e.model_id == "allam-2-7b" for e in plain)
    assert not any(e.model_id == "allam-2-7b" for e in structured)


def test_structured_filtering_keeps_the_rest_of_the_tier(registry, all_keys):
    """The filter must be surgical: hiding one endpoint, not emptying the tier."""
    structured = registry.enabled_endpoints(tier="B", structured=True)
    assert len(structured) > 5


def test_ollama_is_routable_with_no_credentials_at_all(registry, monkeypatch):
    """The keyless backstop: no env var can be missing, so it never drops out.

    Every other provider disappears from the ladder when its key is unset, which
    on a bare machine would leave nothing to route to.
    """
    for name in (
        "GROQ_API_KEY", "MISTRAL_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY",
        "NVIDIA_API_KEY", "OPENROUTER_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    assert registry.is_usable("ollama")
    endpoints = registry.enabled_endpoints(tier="B")
    assert {e.provider for e in endpoints} == {"ollama"}


def test_unknown_model_raises(registry):
    with pytest.raises(KeyError, match="unknown model"):
        resolve_candidates("no-such-model", registry=registry)


def test_unknown_tier_raises(registry):
    with pytest.raises(Exception):
        resolve_candidates(None, "Z", registry=registry)
