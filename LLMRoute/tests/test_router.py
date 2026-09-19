"""route() end to end, against scripted fake providers."""

import asyncio
import logging

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from llm_router import (
    TIERS,
    AllCandidatesExhausted,
    LLMRouter,
    ProviderError,
    RateLimited,
    aroute,
    route,
    route_stream,
)
from llm_router.router import DEFAULT_MAX_ATTEMPTS

pytestmark = pytest.mark.usefixtures("fake_providers")


@pytest.fixture
def call(registry, ledger, sessions):
    """route() with the test registry, ledger and session state wired in."""

    def _call(messages="hello", **kwargs):
        kwargs.setdefault("registry", registry)
        kwargs.setdefault("ledger", ledger)
        kwargs.setdefault("session_state", sessions)
        return route(messages, **kwargs)

    return _call


def meta(message):
    return message.response_metadata["llm_router"]


def rate_limited(provider, retry_after=30.0):
    return RateLimited(provider, "m", retry_after=retry_after)


# --------------------------------------------------------------------------- #
# happy path
# --------------------------------------------------------------------------- #

def test_returns_an_ai_message_with_routing_metadata(call):
    response = call("hello", model="gpt-oss-120b")
    assert isinstance(response, AIMessage)
    assert "hello from groq" in response.content
    assert meta(response)["provider"] == "groq"
    assert meta(response)["logical_model"] == "gpt-oss-120b"
    assert meta(response)["tier"] == "S"
    assert meta(response)["tier_downgraded"] is False


def test_accepts_strings_tuples_and_message_objects(call):
    assert call("hi").content
    assert call([("system", "be brief"), ("human", "hi")]).content
    assert call([HumanMessage("hi")]).content


def test_return_route_exposes_the_full_decision(call):
    result = call("hi", model="qwen-27b", return_route=True)
    assert result.endpoint.logical_model == "qwen-27b"
    assert result.tokens == 42
    assert result.attempts[-1].outcome == "success"
    assert result.content == result.message.content


def test_kwargs_reach_the_provider(call, fake_providers):
    from llm_router.providers import get_provider

    call("hi", model="qwen-27b", temperature=0.1, max_tokens=64)
    assert get_provider("groq").last_kwargs["temperature"] == 0.1


def test_tokens_are_charged_to_the_ledger(call, ledger, registry):
    endpoint = registry.get_endpoints("gpt-oss-120b")[0]
    call("hi", model="gpt-oss-120b")
    counters = ledger.snapshot()[endpoint.ledger_key]["counters"]
    assert counters["requests/minute"]["used"] == 1
    assert counters["tokens/minute"]["used"] == 42


# --------------------------------------------------------------------------- #
# falling down the ladder
# --------------------------------------------------------------------------- #

def test_rate_limit_falls_through_to_the_next_provider(call, fake_providers, ledger, registry):
    """gemma-31b is hosted on three providers, so step 1 alone can absorb this."""
    fake_providers.fail["nvidia_nim:google/gemma-4-31b-it"] = rate_limited("nvidia_nim", 45.0)

    response = call("hi", model="gemma-31b")
    assert meta(response)["provider"] == "google_ai_studio"
    assert meta(response)["ladder_step"] == 1          # same model, other provider
    assert meta(response)["tier_downgraded"] is False

    # ...and the failed endpoint is now parked for the retry_after it gave us
    nvidia = registry.get_endpoints("gemma-31b")[0]
    assert not ledger.can_call(nvidia)
    assert ledger.availability(nvidia).retry_after == pytest.approx(45.0)


def test_groq_498_is_treated_as_a_rate_limit(call, fake_providers):
    """Flex-tier capacity must fall back, not raise.

    Groq is the only host of gpt-oss-120b, so the fallback is a tier-S peer at
    step 3 rather than another host at step 1 - the tier still holds either way.
    """
    fake_providers.fail["groq:openai/gpt-oss-120b"] = RateLimited(
        "groq", "openai/gpt-oss-120b", status=498, message="flex tier capacity exceeded"
    )
    response = call("hi", model="gpt-oss-120b")
    assert meta(response)["provider"] == "nvidia_nim"
    assert meta(response)["tier"] == "S"
    assert meta(response)["tier_downgraded"] is False


def tier_s_endpoints(registry):
    return [
        endpoint
        for model in registry.models_in_tier("S")
        for endpoint in registry.get_endpoints(model)
    ]


def kill_tier_s(fake_providers, registry):
    """Fail every tier-S endpoint, and return the attempt budget to get past them.

    The budget is returned explicitly rather than relying on the default, so a
    test about what happens *below* tier S says how far it expects to walk. That
    matters more than it looks: the default has twice been raised to keep a tier
    reachable, and a test that silently depended on the old margin would have
    changed meaning without failing.
    """
    endpoints = tier_s_endpoints(registry)
    for endpoint in endpoints:
        fake_providers.fail[endpoint.key] = rate_limited(endpoint.provider)
    return len(endpoints) + 1


def test_exhausting_a_tier_downgrades_and_flags_it(call, fake_providers, registry):
    budget = kill_tier_s(fake_providers, registry)

    response = call("hi", model="gpt-oss-120b", max_attempts=budget)
    routing = meta(response)
    assert routing["tier"] in ("A", "B")
    assert routing["tier_downgraded"] is True
    assert routing["requested_tier"] == "S"
    assert "S -> " in routing["reason"]


def test_downgrade_is_logged_under_sticky(call, fake_providers, registry, caplog):
    budget = kill_tier_s(fake_providers, registry)

    with caplog.at_level(logging.WARNING, logger="llm_router"):
        call("hi", model="gpt-oss-120b", strategy="sticky", session_id="run-42",
             max_attempts=budget)
    assert any("tier downgrade" in record.message for record in caplog.records)


def test_downgrade_is_silent_under_free_first(call, fake_providers, registry, caplog):
    """free_first asked for whatever works; it does not need warning about it."""
    budget = kill_tier_s(fake_providers, registry)

    with caplog.at_level(logging.WARNING, logger="llm_router"):
        response = call("hi", model="gpt-oss-120b", strategy="free_first",
                        max_attempts=budget)
    assert not any("tier downgrade" in r.message for r in caplog.records)
    # ...but it is still recorded, so nothing is actually hidden
    assert meta(response)["tier_downgraded"] is True


def test_attempts_record_every_hop(call, fake_providers):
    fake_providers.fail["groq:openai/gpt-oss-120b"] = rate_limited("groq")
    result = call("hi", model="gpt-oss-120b", return_route=True)
    outcomes = [(a.endpoint, a.outcome) for a in result.attempts]
    assert outcomes[0] == ("groq:openai/gpt-oss-120b", "rate_limited")
    assert outcomes[-1] == ("nvidia_nim:deepseek-ai/deepseek-v4-pro-0813", "success")


def test_max_attempts_bounds_the_walk(call, fake_providers, registry):
    for endpoint in registry.all_endpoints():
        fake_providers.fail[endpoint.key] = rate_limited(endpoint.provider)
    with pytest.raises(AllCandidatesExhausted):
        call("hi", model="gpt-oss-120b", max_attempts=2)
    assert len(fake_providers.calls) == 2


def test_max_attempts_clears_every_tier(registry, all_keys):
    """The cap must not be smaller than a tier, or that tier loses its tail.

    This has bitten twice. At 6 it was one short of tier S. At 8 it was six
    short of tier B, which had grown to 14 by gaining a local Ollama endpoint at
    priority 950 - putting the free, unmetered backstop beyond reach in exactly
    the situation it exists for.

    The cap is applied to the ordered list *before* availability is checked, so
    parked endpoints consume slots too and the tail is lost even when most
    candidates are merely rate-limited rather than failing.
    """
    for tier in TIERS:
        endpoints = registry.enabled_endpoints(tier=tier)
        assert len(endpoints) <= DEFAULT_MAX_ATTEMPTS, (
            f"tier {tier} has {len(endpoints)} endpoints but the ladder only "
            f"tries {DEFAULT_MAX_ATTEMPTS}; its last "
            f"{len(endpoints) - DEFAULT_MAX_ATTEMPTS} would be unreachable. "
            f"Raise DEFAULT_MAX_ATTEMPTS."
        )


def test_the_local_backstop_is_reached_when_every_hosted_endpoint_is_spent(
    call, fake_providers, registry, all_keys
):
    """The scenario Ollama exists for, and the one that was silently broken.

    `contact_extraction` runs on tier B, once per vendor, and is meant to land
    on the free local model when the hosted free tiers are exhausted. That is
    precisely when every candidate ahead of it is rate-limited - so if the cap
    cannot reach position 14, the backstop is decorative.
    """
    hosted = [
        e for e in registry.enabled_endpoints(tier="B") if e.provider != "ollama"
    ]
    for endpoint in hosted:
        fake_providers.fail[endpoint.key] = rate_limited(endpoint.provider)

    result = call("hi", tier="B", return_route=True)

    assert result.endpoint.provider == "ollama"
    assert len(hosted) > DEFAULT_MAX_ATTEMPTS - 4  # i.e. the tail really is deep


def test_ledger_skips_are_not_provider_calls(call, fake_providers, ledger, registry):
    """A candidate the ledger already knows is exhausted must not be called."""
    groq = registry.get_endpoints("gpt-oss-120b")[0]
    ledger.record_rate_limited(groq, retry_after=60.0)
    call("hi", model="gpt-oss-120b")
    assert groq.key not in fake_providers.calls


# --------------------------------------------------------------------------- #
# error handling
# --------------------------------------------------------------------------- #

def test_a_bad_request_fails_fast_instead_of_burning_the_ladder(call, fake_providers):
    fake_providers.fail["groq:openai/gpt-oss-120b"] = ProviderError(
        "groq", "openai/gpt-oss-120b", message="invalid tool schema", status=400
    )
    with pytest.raises(ProviderError, match="invalid tool schema"):
        call("hi", model="gpt-oss-120b")
    assert len(fake_providers.calls) == 1


def test_a_transient_5xx_moves_on(call, fake_providers):
    fake_providers.fail["groq:openai/gpt-oss-120b"] = ProviderError(
        "groq", "openai/gpt-oss-120b", message="overloaded", status=503, transient=True
    )
    assert meta(call("hi", model="gpt-oss-120b"))["provider"] == "nvidia_nim"


def test_an_unknown_model_id_parks_that_endpoint(call, fake_providers, ledger, registry):
    """A stale id in models.yaml should not take the whole call down."""
    fake_providers.fail["groq:openai/gpt-oss-120b"] = ProviderError(
        "groq", "openai/gpt-oss-120b", message="model not found", status=404
    )
    assert meta(call("hi", model="gpt-oss-120b"))["provider"] == "nvidia_nim"
    groq = registry.get_endpoints("gpt-oss-120b")[0]
    assert ledger.availability(groq).retry_after > 60


def test_exhaustion_error_explains_itself(call, fake_providers, registry):
    for endpoint in registry.all_endpoints():
        fake_providers.fail[endpoint.key] = rate_limited(endpoint.provider, 30.0)
    with pytest.raises(AllCandidatesExhausted) as excinfo:
        call("hi", model="gpt-oss-120b")

    error = excinfo.value
    assert "gpt-oss-120b" in str(error)
    assert "soonest retry" in str(error)
    assert error.retry_after == pytest.approx(30.0, abs=1.0)
    assert error.attempts


def test_max_wait_retries_once_quota_frees_up(registry, ledger, sessions, clock, monkeypatch):
    """A short, explicit wait is allowed; the default is to fail immediately."""
    for endpoint in registry.all_endpoints():
        ledger.record_rate_limited(endpoint, retry_after=5.0)

    monkeypatch.setattr("time.sleep", lambda seconds: clock.advance(seconds))
    response = route(
        "hi", model="gpt-oss-120b", max_wait=10.0,
        registry=registry, ledger=ledger, session_state=sessions,
    )
    assert response.content


def test_without_max_wait_it_fails_immediately(registry, ledger, sessions):
    for endpoint in registry.all_endpoints():
        ledger.record_rate_limited(endpoint, retry_after=5.0)
    with pytest.raises(AllCandidatesExhausted):
        route("hi", model="gpt-oss-120b",
              registry=registry, ledger=ledger, session_state=sessions)


# --------------------------------------------------------------------------- #
# forcing
# --------------------------------------------------------------------------- #

def test_forcing_a_provider_and_model_pins_that_exact_endpoint(call, fake_providers):
    """gemma-31b's default host is nvidia_nim; naming a provider overrides that."""
    response = call("hi", provider="openrouter", model="gemma-31b")
    assert meta(response)["provider"] == "openrouter"


def test_a_pinned_endpoint_does_not_silently_fall_back(call, fake_providers):
    """Forcing an endpoint means that one; answering from elsewhere would lie."""
    fake_providers.fail["openrouter:google/gemma-4-31b-it:free"] = rate_limited("openrouter")
    with pytest.raises(AllCandidatesExhausted):
        call("hi", provider="openrouter", model="gemma-31b")
    assert fake_providers.calls == ["openrouter:google/gemma-4-31b-it:free"]


def test_forcing_only_a_provider_still_uses_its_ladder(call, fake_providers):
    fake_providers.fail["groq:openai/gpt-oss-120b"] = rate_limited("groq")
    response = call("hi", provider="groq", model=None, tier="S")
    assert meta(response)["provider"] == "groq"
    assert meta(response)["tier_downgraded"] is True     # only groq's A/B left


def test_tier_floor_without_a_model(call):
    response = call("hi", tier="S")
    assert meta(response)["tier"] == "S"
    assert meta(response)["requested_tier"] == "S"


def test_downgrade_can_be_forbidden(call, fake_providers, registry):
    budget = kill_tier_s(fake_providers, registry)
    with pytest.raises(AllCandidatesExhausted):
        call("hi", tier="S", allow_tier_downgrade=False, max_attempts=budget)


# --------------------------------------------------------------------------- #
# sticky sessions
# --------------------------------------------------------------------------- #

def test_sticky_keeps_a_session_on_one_endpoint(call, fake_providers):
    first = call("hi", model="qwen-27b", strategy="sticky", session_id="run-42")
    fake_providers.calls.clear()
    for _ in range(3):
        again = call("hi", model="qwen-27b", strategy="sticky", session_id="run-42")
        assert meta(again)["provider"] == meta(first)["provider"]
    assert set(fake_providers.calls) == {meta(first)["provider"] + ":qwen/qwen3.8-27b"}


def test_sticky_sessions_are_independent(call, fake_providers, registry, sessions):
    from llm_router.ladder import resolve_candidates

    openrouter = next(
        c for c in resolve_candidates("gemma-31b", registry=registry)
        if c.endpoint.provider == "openrouter"
    )
    sessions.remember("b", openrouter)
    assert meta(call("hi", model="gemma-31b", strategy="sticky", session_id="a"))["provider"] == "nvidia_nim"
    assert meta(call("hi", model="gemma-31b", strategy="sticky", session_id="b"))["provider"] == "openrouter"


def test_sticky_moves_on_when_its_endpoint_dies_then_returns(call, fake_providers, ledger, registry, clock):
    """Groq hosts qwen-27b under two model ids (3.8 and 3.6); ratelimiting one
    should move to the other before ever leaving Groq for a same-tier peer."""
    session = "research-run-42"
    first = call("hi", model="qwen-27b", strategy="sticky", session_id=session)
    assert meta(first)["provider"] == "groq"
    assert meta(first)["model_id"] == "qwen/qwen3.8-27b"

    groq_38 = registry.get_endpoints("qwen-27b")[0]
    ledger.record_rate_limited(groq_38, retry_after=60.0)
    second = call("hi", model="qwen-27b", strategy="sticky", session_id=session)
    assert meta(second)["provider"] == "groq"
    assert meta(second)["model_id"] == "qwen/qwen3.6-27b"

    groq_36 = registry.get_endpoints("qwen-27b")[1]
    ledger.record_rate_limited(groq_36, retry_after=60.0)
    third = call("hi", model="qwen-27b", strategy="sticky", session_id=session)
    assert meta(third)["provider"] == "mistral"
    assert meta(third)["tier_downgraded"] is False     # still tier A

    clock.advance(61)
    # It re-pinned to mistral (a same-tier peer), and stays there.
    assert meta(call("hi", model="qwen-27b", strategy="sticky", session_id=session))["provider"] == "mistral"


def test_a_downgrade_does_not_become_the_new_pin(call, fake_providers, registry, clock, ledger):
    session = "run"
    for endpoint in tier_s_endpoints(registry):
        ledger.record_rate_limited(endpoint, retry_after=60.0)

    downgraded = call("hi", model="gpt-oss-120b", strategy="sticky", session_id=session)
    assert meta(downgraded)["tier_downgraded"] is True

    clock.advance(61)
    recovered = call("hi", model="gpt-oss-120b", strategy="sticky", session_id=session)
    assert meta(recovered)["tier"] == "S"
    assert meta(recovered)["tier_downgraded"] is False


# --------------------------------------------------------------------------- #
# header sync
# --------------------------------------------------------------------------- #

def test_groq_headers_correct_the_ledger(call, fake_providers, ledger, registry):
    """Groq says the key has nothing left, even though we only spent one call."""
    groq = registry.get_endpoints("gpt-oss-120b")[0]
    fake_providers.headers[groq.key] = {
        "limit_requests": 30, "remaining_requests": 0,
        "limit_tokens": 8000, "remaining_tokens": 0,
    }
    call("hi", model="gpt-oss-120b")
    assert not ledger.can_call(groq)


# --------------------------------------------------------------------------- #
# async and streaming
# --------------------------------------------------------------------------- #

def test_aroute_matches_route(registry, ledger, sessions, fake_providers):
    response = asyncio.run(aroute(
        "hi", model="gpt-oss-120b",
        registry=registry, ledger=ledger, session_state=sessions,
    ))
    assert meta(response)["provider"] == "groq"


def test_aroute_falls_back_too(registry, ledger, sessions, fake_providers):
    fake_providers.fail["groq:openai/gpt-oss-120b"] = rate_limited("groq")
    response = asyncio.run(aroute(
        "hi", model="gpt-oss-120b",
        registry=registry, ledger=ledger, session_state=sessions,
    ))
    assert meta(response)["provider"] == "nvidia_nim"


def test_streaming_yields_chunks(registry, ledger, sessions, fake_providers):
    chunks = list(route_stream(
        "hi", model="gpt-oss-120b",
        registry=registry, ledger=ledger, session_state=sessions,
    ))
    assert "".join(str(c.content) for c in chunks).startswith("hello from groq")


def test_streaming_falls_back_before_the_first_chunk(registry, ledger, sessions, fake_providers):
    """Switching model mid-answer would be nonsense, so it must fail early."""
    fake_providers.fail["groq:openai/gpt-oss-120b"] = rate_limited("groq")
    chunks = list(route_stream(
        "hi", model="gpt-oss-120b",
        registry=registry, ledger=ledger, session_state=sessions,
    ))
    assert "nvidia_nim" in "".join(str(c.content) for c in chunks)


def test_streaming_charges_tokens(registry, ledger, sessions, fake_providers):
    endpoint = registry.get_endpoints("gpt-oss-120b")[0]
    list(route_stream(
        "hi", model="gpt-oss-120b",
        registry=registry, ledger=ledger, session_state=sessions,
    ))
    counters = ledger.snapshot()[endpoint.ledger_key]["counters"]
    assert counters["tokens/minute"]["used"] == 42


# --------------------------------------------------------------------------- #
# LLMRouter
# --------------------------------------------------------------------------- #

def test_llmrouter_invoke(registry, ledger, sessions, fake_providers):
    router = LLMRouter(model="gpt-oss-120b", registry=registry, ledger=ledger,
                       session_state=sessions)
    response = router.invoke("hi")
    assert isinstance(response, AIMessage)
    assert meta(response)["provider"] == "groq"
    assert router.last_route.endpoint.provider == "groq"


def test_llmrouter_falls_back_like_route(registry, ledger, sessions, fake_providers):
    fake_providers.fail["groq:openai/gpt-oss-120b"] = rate_limited("groq")
    router = LLMRouter(model="gpt-oss-120b", registry=registry, ledger=ledger,
                       session_state=sessions)
    assert meta(router.invoke("hi"))["provider"] == "nvidia_nim"


def test_llmrouter_sticky_session(registry, ledger, sessions, fake_providers):
    router = LLMRouter(strategy="sticky", session_id="agent-1", model="qwen-27b",
                       registry=registry, ledger=ledger, session_state=sessions)
    first = router.invoke("hi")
    assert meta(router.invoke("again"))["provider"] == meta(first)["provider"]


def test_llmrouter_model_kwargs_are_forwarded(registry, ledger, sessions, fake_providers):
    from llm_router.providers import get_provider

    router = LLMRouter(model="qwen-27b", model_kwargs={"temperature": 0.0},
                       registry=registry, ledger=ledger, session_state=sessions)
    router.invoke("hi")
    assert get_provider("groq").last_kwargs["temperature"] == 0.0


def test_llmrouter_bind_tools_reaches_the_provider(registry, ledger, sessions, fake_providers):
    from langchain_core.tools import tool

    from llm_router.providers import get_provider

    @tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    router = LLMRouter(model="qwen-27b", registry=registry, ledger=ledger,
                       session_state=sessions)
    router.bind_tools([add]).invoke("what is 2+2")
    assert get_provider("groq").last_kwargs["tools"] == [add]


def test_llmrouter_streams(registry, ledger, sessions, fake_providers):
    router = LLMRouter(model="gpt-oss-120b", registry=registry, ledger=ledger,
                       session_state=sessions)
    text = "".join(str(chunk.content) for chunk in router.stream("hi"))
    assert text.startswith("hello from groq")


def test_llmrouter_ainvoke(registry, ledger, sessions, fake_providers):
    router = LLMRouter(model="gpt-oss-120b", registry=registry, ledger=ledger,
                       session_state=sessions)
    response = asyncio.run(router.ainvoke("hi"))
    assert meta(response)["provider"] == "groq"


def test_llmrouter_batch_shares_one_ledger(registry, ledger, sessions, fake_providers):
    router = LLMRouter(model="gpt-oss-120b", registry=registry, ledger=ledger,
                       session_state=sessions)
    responses = router.batch(["a", "b", "c"])
    assert len(responses) == 3
    total = sum(
        bucket["counters"]["requests/minute"]["used"]
        for bucket in ledger.snapshot().values()
    )
    assert total == 3


def test_llmrouter_reports_its_identity(registry, ledger, sessions, fake_providers):
    router = LLMRouter(strategy="sticky", model="qwen-27b", tier="A")
    assert router._llm_type == "llm_router"
    assert router._identifying_params["model"] == "qwen-27b"


def test_a_bad_request_does_not_park_a_healthy_endpoint(call, fake_providers, ledger, registry):
    """The request was wrong, not the endpoint - it must stay callable."""
    groq = registry.get_endpoints("gpt-oss-120b")[0]
    fake_providers.fail[groq.key] = ProviderError(
        "groq", groq.model_id, message="invalid tool schema", status=400
    )
    with pytest.raises(ProviderError):
        call("hi", model="gpt-oss-120b")

    fake_providers.fail.clear()
    assert ledger.can_call(groq)
    assert meta(call("hi", model="gpt-oss-120b"))["provider"] == "groq"
