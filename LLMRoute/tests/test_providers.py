"""Error normalisation: every provider's failure shape -> RateLimited / ProviderError."""

import httpx
import pytest

from llm_router.providers import (
    AuthError,
    ProviderError,
    RateLimited,
    get_provider,
    parse_duration,
    rate_limit_snapshot,
    tokens_used,
)
from llm_router.registry import Endpoint


def make_endpoint(provider, model_id="m"):
    return Endpoint(
        provider=provider, model_id=model_id, logical_model="m", tier="A"
    )


def http_error(status, headers=None, body="rate limited"):
    """An httpx.HTTPStatusError, which is what langchain_mistralai raises."""
    request = httpx.Request("POST", "https://example.test/v1/chat")
    response = httpx.Response(status, headers=headers or {}, text=body, request=request)
    return httpx.HTTPStatusError(f"Error response {status}", request=request, response=response)


class SDKStatusError(Exception):
    """Shaped like groq.APIStatusError / openai.APIStatusError."""

    def __init__(self, status_code, headers=None, message="error"):
        self.status_code = status_code
        self.response = httpx.Response(
            status_code,
            headers=headers or {},
            request=httpx.Request("POST", "https://example.test"),
        )
        super().__init__(message)


class GoogleAPIError(Exception):
    """Shaped like google.genai.errors.APIError."""

    def __init__(self, code, details, message="error"):
        self.code = code
        self.details = details
        self.status = "RESOURCE_EXHAUSTED"
        super().__init__(message)


# --------------------------------------------------------------------------- #
# generic
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "provider_name",
    ["groq", "ollama", "mistral", "google_ai_studio", "nvidia_nim", "openrouter"],
)
def test_429_becomes_rate_limited(provider_name):
    provider = get_provider(provider_name)
    endpoint = make_endpoint(provider_name)
    error = provider.interpret(SDKStatusError(429, {"retry-after": "17"}), endpoint)
    assert isinstance(error, RateLimited)
    assert error.retry_after == 17.0


@pytest.mark.parametrize("status", [500, 502, 503, 504, 529])
def test_server_errors_are_transient_not_rate_limits(status):
    provider = get_provider("mistral")
    error = provider.interpret(SDKStatusError(status), make_endpoint("mistral"))
    assert isinstance(error, ProviderError) and error.transient
    assert not isinstance(error, RateLimited)


def test_unsupported_response_format_is_skippable_not_fatal():
    """The one 400 another endpoint can serve.

    Groq supports json_schema on some models only - allam-2-7b answers
    "This model does not support response format `json_schema`" with a 400.
    Treated as an ordinary bad request that would end the whole call, which is
    wrong: the request is fine, this model just cannot serve it.
    """
    from llm_router.router import _is_fatal

    provider = get_provider("groq")
    exc = SDKStatusError(
        400,
        message="This model does not support response format `json_schema`.",
    )
    error = provider.interpret(exc, make_endpoint("groq", "allam-2-7b"))

    assert isinstance(error, ProviderError)
    assert not _is_fatal(error)
    # The message names the remedy: declare the capability in config.
    assert "supports_structured_output" in str(error)


class GroqBadRequest(Exception):
    """Shaped like groq.BadRequestError, which carries a structured body."""

    def __init__(self, code, message="bad request"):
        self.status_code = 400
        self.body = {"error": {"message": message, "code": code}}
        super().__init__(f"Error code: 400 - {self.body}")


def test_a_model_failing_to_produce_valid_json_is_skippable():
    """Observed live, mid-run, on Groq's gpt-oss-20b under json_schema:

        code: json_validate_failed, failed_generation: ''

    The model produced nothing that matched the schema. That is a capability
    limit of that model on that prompt, not a defect in the request - another
    endpoint answers it fine. Treated as fatal it ended the whole graph run and
    surfaced to the client as an `error` event.
    """
    from llm_router.router import _is_fatal

    provider = get_provider("groq")
    error = provider.interpret(
        GroqBadRequest("json_validate_failed", "Failed to validate JSON."),
        make_endpoint("groq", "openai/gpt-oss-20b"),
    )

    assert isinstance(error, ProviderError)
    assert not _is_fatal(error)


def test_the_code_is_read_from_the_body_not_the_message():
    """Message wording changes without notice; the code does not."""
    from llm_router.router import _is_fatal

    provider = get_provider("groq")
    error = provider.interpret(
        GroqBadRequest("json_validate_failed", "some future rewording"),
        make_endpoint("groq"),
    )
    assert not _is_fatal(error)


def test_an_ordinary_bad_request_is_still_fatal():
    """The general rule the case above is an exception to: a request that is
    wrong everywhere should fail once, not once per candidate."""
    from llm_router.router import _is_fatal

    provider = get_provider("groq")
    error = provider.interpret(
        SDKStatusError(400, message="messages: field required"), make_endpoint("groq")
    )
    assert _is_fatal(error)


def test_a_bad_request_with_an_unrelated_code_is_still_fatal():
    """The exemption is narrow on purpose: only failures to produce the
    requested shape. A malformed request must still fail once, not fourteen
    times."""
    from llm_router.router import _is_fatal

    provider = get_provider("groq")
    error = provider.interpret(
        GroqBadRequest("invalid_api_param", "temperature must be <= 2"),
        make_endpoint("groq"),
    )
    assert _is_fatal(error)


@pytest.mark.parametrize("status", [401, 403])
def test_auth_failures_are_auth_errors(status):
    provider = get_provider("groq")
    error = provider.interpret(SDKStatusError(status), make_endpoint("groq"))
    assert isinstance(error, AuthError)


def test_bad_request_is_a_hard_failure():
    """A malformed request must not be retried around the ladder forever."""
    provider = get_provider("groq")
    error = provider.interpret(SDKStatusError(400), make_endpoint("groq"))
    assert isinstance(error, ProviderError)
    assert not error.transient and not isinstance(error, RateLimited)


def test_timeouts_are_transient():
    provider = get_provider("groq")
    error = provider.interpret(httpx.ReadTimeout("timed out"), make_endpoint("groq"))
    assert isinstance(error, ProviderError) and error.transient


def test_missing_api_key_raises_auth_error(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    endpoint = Endpoint(
        provider="groq", model_id="openai/gpt-oss-120b", logical_model="gpt-oss-120b",
        tier="S", api_key_env=("GROQ_API_KEY",),
    )
    with pytest.raises(AuthError, match="GROQ_API_KEY"):
        get_provider("groq").chat_model(endpoint)


# --------------------------------------------------------------------------- #
# groq specifics
# --------------------------------------------------------------------------- #

def test_groq_498_is_a_rate_limit_not_a_failure():
    """Groq's non-standard flex-tier signal must fall back, not blow up."""
    provider = get_provider("groq")
    error = provider.interpret(SDKStatusError(498), make_endpoint("groq"))
    assert isinstance(error, RateLimited) and error.status == 498


def test_groq_498_detected_from_message_when_status_is_lost():
    provider = get_provider("groq")
    error = provider.interpret(
        Exception("Error code: 498 - flex tier capacity exceeded"), make_endpoint("groq")
    )
    assert isinstance(error, RateLimited) and error.status == 498


def test_groq_compound_retry_after_is_parsed():
    provider = get_provider("groq")
    error = provider.interpret(
        SDKStatusError(429, {"retry-after": "2m59.56s"}), make_endpoint("groq")
    )
    assert error.retry_after == pytest.approx(179.56)


def test_groq_reset_header_used_when_retry_after_absent():
    provider = get_provider("groq")
    error = provider.interpret(
        SDKStatusError(429, {"x-ratelimit-reset-requests": "7.66s"}), make_endpoint("groq")
    )
    assert error.retry_after == pytest.approx(7.66)


def test_rate_limit_headers_are_extracted():
    snapshot = rate_limit_snapshot({
        "x-ratelimit-limit-requests": "30",
        "x-ratelimit-remaining-requests": "4",
        "x-ratelimit-limit-tokens": "8000",
        "x-ratelimit-remaining-tokens": "125",
        "unrelated": "x",
    })
    assert snapshot == {
        "limit_requests": 30, "remaining_requests": 4,
        "limit_tokens": 8000, "remaining_tokens": 125,
    }


# --------------------------------------------------------------------------- #
# mistral specifics
# --------------------------------------------------------------------------- #

def test_mistral_httpx_status_error_is_read():
    """langchain_mistralai raises raw httpx errors, with no status attribute."""
    provider = get_provider("mistral")
    error = provider.interpret(http_error(429, {"retry-after": "12"}), make_endpoint("mistral"))
    assert isinstance(error, RateLimited) and error.retry_after == 12.0


def test_mistral_429_without_retry_after_leaves_it_unknown():
    """No header means the ledger falls back to the configured cooldown."""
    provider = get_provider("mistral")
    error = provider.interpret(http_error(429), make_endpoint("mistral"))
    assert isinstance(error, RateLimited) and error.retry_after is None


# --------------------------------------------------------------------------- #
# google specifics
# --------------------------------------------------------------------------- #

GOOGLE_QUOTA_BODY = {
    "error": {
        "code": 429,
        "status": "RESOURCE_EXHAUSTED",
        "details": [
            {
                "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                "violations": [{
                    "quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier",
                    "quotaMetric": "generativelanguage.googleapis.com/generate_requests",
                }],
            },
            {
                "@type": "type.googleapis.com/google.rpc.RetryInfo",
                "retryDelay": "21s",
            },
        ],
    }
}


def test_google_quota_id_and_retry_delay_are_surfaced():
    provider = get_provider("google_ai_studio")
    error = provider.interpret(
        GoogleAPIError(429, GOOGLE_QUOTA_BODY), make_endpoint("google_ai_studio")
    )
    assert isinstance(error, RateLimited)
    assert error.retry_after == 21.0
    assert error.quota_id == "GenerateRequestsPerDayPerProjectPerModel-FreeTier"
    assert "PerDay" in str(error)      # a daily cap is worth seeing in the log


def test_google_resource_exhausted_without_status_code():
    """Older wrappers surface the status only in the message text."""
    provider = get_provider("google_ai_studio")
    error = provider.interpret(
        Exception("429 RESOURCE_EXHAUSTED: quota exceeded"), make_endpoint("google_ai_studio")
    )
    assert isinstance(error, RateLimited)


def test_google_non_quota_error_is_not_a_rate_limit():
    provider = get_provider("google_ai_studio")
    error = provider.interpret(GoogleAPIError(400, {}), make_endpoint("google_ai_studio"))
    assert isinstance(error, ProviderError) and not isinstance(error, RateLimited)


# --------------------------------------------------------------------------- #
# token accounting
# --------------------------------------------------------------------------- #

def test_tokens_from_usage_metadata():
    from langchain_core.messages import AIMessage

    message = AIMessage(
        content="hi",
        usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    )
    assert tokens_used(message) == 15


def test_tokens_from_legacy_response_metadata():
    from langchain_core.messages import AIMessage

    message = AIMessage(
        content="hi",
        response_metadata={"token_usage": {"prompt_tokens": 28, "completion_tokens": 38}},
    )
    assert tokens_used(message) == 66


def test_tokens_unknown_returns_none():
    from langchain_core.messages import AIMessage

    assert tokens_used(AIMessage(content="hi")) is None


def test_parse_duration_formats():
    assert parse_duration("30") == 30.0
    assert parse_duration("21s") == 21.0
    assert parse_duration("2m59.56s") == pytest.approx(179.56)
    assert parse_duration("500ms") == 0.5
    assert parse_duration(None) is None
    assert parse_duration("garbage") is None


# --------------------------------------------------------------------------- #
# openrouter specifics
# --------------------------------------------------------------------------- #

def test_openrouter_402_is_quota_not_a_fatal_bad_request():
    """Out of credit must park OpenRouter, not abort the whole call.

    route() treats a non-transient 4xx as "wrong everywhere" and stops walking
    the ladder, so a 402 left as a ProviderError would take down a request the
    rest of the ladder could still serve.
    """
    from llm_router.providers import OpenRouterProvider

    provider = get_provider("openrouter")
    error = provider.interpret(SDKStatusError(402), make_endpoint("openrouter"))
    assert isinstance(error, RateLimited) and error.status == 402
    assert error.retry_after == OpenRouterProvider.NO_CREDIT_COOLDOWN


def test_openrouter_reset_header_is_an_absolute_timestamp():
    """OpenRouter sends when the window resets, in epoch ms - not a duration."""
    import time

    provider = get_provider("openrouter")
    reset_at_ms = int((time.time() + 42) * 1000)
    error = provider.interpret(
        SDKStatusError(429, {"x-ratelimit-reset": str(reset_at_ms)}),
        make_endpoint("openrouter"),
    )
    assert isinstance(error, RateLimited)
    assert error.retry_after == pytest.approx(42.0, abs=2.0)


def test_openrouter_stale_reset_header_is_ignored():
    """A reset already in the past would otherwise become a negative cooldown."""
    import time

    provider = get_provider("openrouter")
    stale = int((time.time() - 500) * 1000)
    error = provider.interpret(
        SDKStatusError(429, {"x-ratelimit-reset": str(stale)}),
        make_endpoint("openrouter"),
    )
    # No usable hint, so the ledger falls back to the configured cooldown.
    assert error.retry_after is None


def test_openrouter_retry_after_still_wins_over_the_reset_header():
    provider = get_provider("openrouter")
    error = provider.interpret(
        SDKStatusError(429, {"retry-after": "9", "x-ratelimit-reset": "1"}),
        make_endpoint("openrouter"),
    )
    assert error.retry_after == pytest.approx(9.0)


def test_bare_ratelimit_headers_are_read():
    """OpenRouter reports one request budget without the -requests suffix."""
    snapshot = rate_limit_snapshot(
        {"X-RateLimit-Limit": "50", "X-RateLimit-Remaining": "3"}
    )
    assert snapshot == {"limit_requests": 50, "remaining_requests": 3}


def test_qualified_ratelimit_headers_win_over_bare_ones():
    """A qualified counter names what it counts; the bare one does not."""
    snapshot = rate_limit_snapshot({
        "x-ratelimit-limit-requests": "30",
        "x-ratelimit-remaining-requests": "26",
        "x-ratelimit-limit": "50",
        "x-ratelimit-remaining": "3",
    })
    assert snapshot["limit_requests"] == 30
    assert snapshot["remaining_requests"] == 26


# --------------------------------------------------------------------------- #
# nvidia nim specifics
# --------------------------------------------------------------------------- #

def test_nvidia_nim_waits_longer_than_everyone_else():
    """NIM is shared public infra and is documented as slow under load."""
    from llm_router.providers import DEFAULT_HTTP_TIMEOUT

    assert get_provider("nvidia_nim").http_timeout > DEFAULT_HTTP_TIMEOUT


def test_nvidia_nim_timeouts_are_transient_so_the_ladder_steps_past():
    provider = get_provider("nvidia_nim")
    error = provider.interpret(
        httpx.ReadTimeout("timed out"), make_endpoint("nvidia_nim")
    )
    assert isinstance(error, ProviderError) and error.transient
    assert not isinstance(error, RateLimited)


@pytest.mark.parametrize("provider_name", ["openrouter", "nvidia_nim"])
def test_openai_compatible_providers_point_at_their_own_host(provider_name):
    provider = get_provider(provider_name)
    assert provider.base_url.startswith("https://")
    assert provider.captures_headers
