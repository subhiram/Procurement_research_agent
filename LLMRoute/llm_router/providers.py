"""Thin per-provider wrappers around the real LangChain chat models.

This is the only module that knows what a given provider's failures look like.
Everything above it sees exactly two failure shapes:

    RateLimited     - out of quota, try somewhere else (retry_after if known)
    ProviderError   - anything else that went wrong at the provider

Providers are constructed lazily and cached, so importing llm_router does not
require every provider package to be installed - only the ones you actually
route to.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Iterator, Mapping, Sequence

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage

from .registry import Endpoint

logger = logging.getLogger("llm_router.providers")

# Statuses that mean "you are out of quota, come back later" rather than
# "your request was wrong". 498 is Groq's non-standard flex-tier capacity
# signal, which is a capacity limit and must not be treated as a hard failure.
RATE_LIMIT_STATUSES = frozenset({429, 498})

# Transient server-side failures. Not quota, but worth stepping to the next
# candidate rather than failing the call.
TRANSIENT_STATUSES = frozenset({500, 502, 503, 504, 529})

AUTH_STATUSES = frozenset({401, 403})


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #

class RouterError(Exception):
    """Base class for everything this package raises."""


class RateLimited(RouterError):
    """The provider refused the call for quota reasons.

    `retry_after` is seconds, when the provider told us; None when it did not
    and the ledger should fall back to the configured cooldown.
    """

    def __init__(
        self,
        provider: str,
        model_id: str | None = None,
        retry_after: float | None = None,
        *,
        message: str | None = None,
        status: int | None = None,
        quota_id: str | None = None,
    ) -> None:
        self.provider = provider
        self.model_id = model_id
        self.retry_after = retry_after
        self.status = status
        # Google names the exact limit that was hit; worth keeping for logs.
        self.quota_id = quota_id
        target = f"{provider}/{model_id}" if model_id else provider
        detail = message or "rate limited"
        if retry_after is not None:
            detail += f" (retry after {retry_after:.0f}s)"
        if quota_id:
            detail += f" [quota: {quota_id}]"
        super().__init__(f"{target}: {detail}")


class ProviderError(RouterError):
    """A non-quota failure from a provider."""

    def __init__(
        self,
        provider: str,
        model_id: str | None = None,
        *,
        message: str | None = None,
        status: int | None = None,
        transient: bool = False,
        cause: BaseException | None = None,
    ) -> None:
        self.provider = provider
        self.model_id = model_id
        self.status = status
        self.transient = transient
        self.cause = cause
        target = f"{provider}/{model_id}" if model_id else provider
        super().__init__(f"{target}: {message or cause or 'provider call failed'}")


class AuthError(ProviderError):
    """Missing, invalid or unauthorised API key. Not worth retrying this run."""


class ProviderNotInstalled(ProviderError):
    """The provider's LangChain package is not installed."""


class NoCandidates(RouterError):
    """Nothing in the registry matches the request at all."""


class AllCandidatesExhausted(RouterError):
    """Every candidate on the ladder was out of quota or failed.

    `retry_after` is the shortest wait across all of them, so a caller that
    wants to back off knows how long is worth waiting.
    """

    def __init__(
        self,
        message: str,
        *,
        attempts: Sequence[Mapping[str, Any]] = (),
        retry_after: float | None = None,
    ) -> None:
        self.attempts = list(attempts)
        self.retry_after = retry_after
        super().__init__(message)


# --------------------------------------------------------------------------- #
# Parsing helpers
# --------------------------------------------------------------------------- #

_DURATION_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(ms|s|m|h)")
_DURATION_UNITS = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}


def parse_duration(value: Any) -> float | None:
    """Parse the duration formats providers actually send.

    Handles a bare number of seconds ("30", 30), Google's protobuf style
    ("21s", "1.5s") and Groq's compound style ("2m59.56s", "500ms").
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        pass
    total = 0.0
    matched = False
    for amount, unit in _DURATION_RE.findall(text):
        total += float(amount) * _DURATION_UNITS[unit]
        matched = True
    return total if matched else None


def _as_int(value: Any) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def status_of(exc: BaseException) -> int | None:
    """Best-effort HTTP status from any provider SDK's exception.

    Structural rather than isinstance-based on purpose: these SDKs rename and
    reshuffle their exception classes between releases, but they all keep a
    status code somewhere reachable.
    """
    for attr in ("status_code", "http_status", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
    response = getattr(exc, "response", None)
    if response is not None:
        value = getattr(response, "status_code", None)
        if isinstance(value, int):
            return value
    return None


def headers_of(exc: BaseException) -> Mapping[str, str]:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if isinstance(headers, Mapping):
        return headers
    return {}


def retry_after_from_headers(headers: Mapping[str, str]) -> float | None:
    lowered = {str(k).lower(): v for k, v in headers.items()}
    for key in ("retry-after", "x-ratelimit-reset-requests", "x-ratelimit-reset-tokens"):
        parsed = parse_duration(lowered.get(key))
        if parsed is not None:
            return parsed
    return None


def rate_limit_snapshot(headers: Mapping[str, str]) -> dict[str, int]:
    """Pull x-ratelimit-* counters out of a response's headers.

    Two spellings are in the wild. OpenAI-shaped providers (Groq, NVIDIA NIM)
    qualify the counter they mean - `x-ratelimit-remaining-requests` - while
    OpenRouter reports a single request budget under the bare
    `x-ratelimit-remaining`. The qualified form wins where both appear, since
    the bare one cannot say whether it is counting requests or tokens.
    """
    lowered = {str(k).lower(): v for k, v in headers.items()}
    out: dict[str, int] = {}
    fields = {
        "limit_requests": ("x-ratelimit-limit-requests", "x-ratelimit-limit"),
        "remaining_requests": (
            "x-ratelimit-remaining-requests", "x-ratelimit-remaining",
        ),
        "limit_tokens": ("x-ratelimit-limit-tokens",),
        "remaining_tokens": ("x-ratelimit-remaining-tokens",),
    }
    for name, candidates in fields.items():
        for header in candidates:
            value = _as_int(lowered.get(header))
            if value is not None:
                out[name] = value
                break
    return out


def tokens_used(message: BaseMessage) -> int | None:
    """Total tokens for a response, however the provider chose to report it."""
    usage = getattr(message, "usage_metadata", None)
    if isinstance(usage, Mapping):
        total = usage.get("total_tokens")
        if total is None:
            total = (usage.get("input_tokens") or 0) + (usage.get("output_tokens") or 0)
        if total:
            return int(total)
    metadata = getattr(message, "response_metadata", None) or {}
    token_usage = metadata.get("token_usage") or metadata.get("usage") or {}
    if isinstance(token_usage, Mapping):
        total = token_usage.get("total_tokens")
        if total is None:
            total = (
                (token_usage.get("prompt_tokens") or 0)
                + (token_usage.get("completion_tokens") or 0)
            )
        if total:
            return int(total)
    return None


# --------------------------------------------------------------------------- #
# Header capture
# --------------------------------------------------------------------------- #

# langchain_groq drops response headers, but the underlying SDK accepts a custom
# httpx client. An event hook on that client stashes the rate-limit headers of
# the most recent response for the current thread / task, which the wrapper
# reads back after invoke() returns. A ContextVar rather than a plain dict so
# concurrent calls do not read each other's counters.
_last_headers: ContextVar[dict[str, str] | None] = ContextVar(
    "llm_router_last_headers", default=None
)


def _capture_headers(response: Any) -> None:
    try:
        _last_headers.set(dict(response.headers))
    except Exception:  # pragma: no cover - never break a call over telemetry
        pass


DEFAULT_HTTP_TIMEOUT = 60.0


def _build_http_clients(timeout: float = DEFAULT_HTTP_TIMEOUT) -> tuple[Any, Any]:
    import httpx

    hooks = {"response": [_capture_headers]}
    return (
        httpx.Client(event_hooks=hooks, timeout=httpx.Timeout(timeout)),
        httpx.AsyncClient(
            event_hooks={"response": [_a_capture_headers]},
            timeout=httpx.Timeout(timeout),
        ),
    )


async def _a_capture_headers(response: Any) -> None:
    _capture_headers(response)


@dataclass
class ProviderResponse:
    """A completed call, normalised."""

    message: AIMessage
    endpoint: Endpoint
    tokens: int | None = None
    rate_limit: Mapping[str, int] = field(default_factory=dict)
    #: The schema instance, when the call asked for structured output. The raw
    #: AIMessage is kept alongside it rather than replaced, so token accounting
    #: and rate-limit headers work identically on both paths.
    parsed: Any = None


# --------------------------------------------------------------------------- #
# Providers
# --------------------------------------------------------------------------- #

#: Provider error codes meaning "this model could not produce the requested
#: shape". Distinct from a malformed request, which is wrong everywhere.
_STRUCTURED_OUTPUT_FAILURE_CODES = frozenset({"json_validate_failed"})


def _endpoint_cannot_serve(exc: BaseException) -> bool:
    """Whether a 400 means *this model* cannot serve the call, not that the
    request is wrong.

    The router treats a non-transient 4xx as fatal, because a bad request should
    fail once rather than be retried against every endpoint in turn. These are
    the exception: the request is fine and another model will answer it, so
    failing the whole call throws away a ladder that would have worked.

    Three real cases, all observed live against Groq:

    - ``code: json_validate_failed`` - the model emitted output that did not
      match the schema (gpt-oss-20b, mid-run, with an empty ``failed_generation``
      so it produced nothing usable at all). A capability limit of that model on
      that prompt, not a defect in the request.
    - "This model does not support response format `json_schema`" (allam-2-7b).
    - "`tool calling` is not supported with this model" (allam-2-7b again).

    The code is checked before the text: it is exact, whereas message wording
    changes without notice.
    """
    if status_of(exc) != 400:
        return False
    if _error_code(exc) in _STRUCTURED_OUTPUT_FAILURE_CODES:
        return True
    text = str(exc).lower()
    return (
        "response format" in text
        or "response_format" in text
        or "tool calling" in text
        or "tool_calling" in text
        or "failed to validate json" in text
    )


def _error_code(exc: BaseException) -> str:
    """The provider's own error code, when the SDK exposes one."""
    body = getattr(exc, "body", None)
    if isinstance(body, Mapping):
        error = body.get("error")
        if isinstance(error, Mapping):
            return str(error.get("code") or "")
    return str(getattr(exc, "code", "") or "")


#: Reserved call kwarg carrying a structured-output schema down to whichever
#: endpoint ends up serving the call. Travels as a kwarg rather than as router
#: state so it flows through route(), aroute() and the LangChain adapter by the
#: same path as every other per-call setting.
STRUCTURED_OUTPUT_KWARG = "structured_output_schema"


class BaseProvider:
    """One provider. Subclasses supply construction and error interpretation."""

    name: str = ""
    package: str = ""
    #: kwargs that must not be forwarded to the chat model constructor
    captures_headers: bool = False

    def __init__(self) -> None:
        self._models: dict[Any, Any] = {}
        self._lock = threading.Lock()

    # -- construction ------------------------------------------------------- #

    def _model_class(self) -> type:
        """The LangChain chat model class this provider builds. Imported lazily."""
        raise NotImplementedError

    def model_class(self) -> type | None:
        """_model_class(), or None when the provider package is not installed."""
        try:
            return self._model_class()
        except ImportError:
            return None

    def _construct(self, endpoint: Endpoint, **kwargs: Any) -> Any:
        raise NotImplementedError

    def _import_error(self, exc: ImportError) -> ProviderNotInstalled:
        return ProviderNotInstalled(
            self.name,
            message=(
                f"{self.package} is not installed; "
                f"run `pip install {self.package}` to route to {self.name}"
            ),
            cause=exc,
        )

    def chat_model(self, endpoint: Endpoint, **kwargs: Any) -> Any:
        """Cached LangChain chat model for this endpoint and these settings."""
        cache_key = (endpoint.model_id, _freeze(kwargs))
        with self._lock:
            model = self._models.get(cache_key)
            if model is None:
                if endpoint.api_key_env and endpoint.api_key is None:
                    raise AuthError(
                        self.name,
                        endpoint.model_id,
                        message=(
                            "no API key found; set "
                            + " or ".join(endpoint.api_key_env)
                        ),
                        status=401,
                    )
                try:
                    model = self._construct(endpoint, **kwargs)
                except ImportError as exc:
                    raise self._import_error(exc) from exc
                self._models[cache_key] = model
            return model

    # -- errors ------------------------------------------------------------- #

    def interpret(self, exc: BaseException, endpoint: Endpoint) -> RouterError:
        """Map any provider exception onto RateLimited / ProviderError."""
        status = status_of(exc)
        headers = headers_of(exc)
        if _endpoint_cannot_serve(exc):
            # A 400 normally means the request is wrong everywhere and the
            # router fails it once rather than twelve times. These are the
            # exception: they say *this model* could not produce the requested
            # shape, which another endpoint in the ladder usually can. Reported
            # as a missing-model 404 so it is skipped rather than fatal, and
            # parked long enough not to be retried all run.
            return ProviderError(
                self.name,
                endpoint.model_id,
                message=(
                    f"{endpoint.model_id!r} could not produce "
                    f"{endpoint.structured_output_method!r} structured output "
                    f"for this call ({_short(exc)}); if it never can, set "
                    f"supports_structured_output: false on this endpoint in "
                    f"models.yaml"
                ),
                status=404,
                cause=exc,
            )
        if status in RATE_LIMIT_STATUSES:
            return RateLimited(
                self.name,
                endpoint.model_id,
                retry_after=retry_after_from_headers(headers),
                message=_short(exc),
                status=status,
            )
        if status in AUTH_STATUSES:
            return AuthError(
                self.name, endpoint.model_id,
                message=_short(exc), status=status, cause=exc,
            )
        if status in TRANSIENT_STATUSES:
            return ProviderError(
                self.name, endpoint.model_id, message=_short(exc),
                status=status, transient=True, cause=exc,
            )
        if _looks_transient(exc):
            return ProviderError(
                self.name, endpoint.model_id, message=_short(exc),
                status=status, transient=True, cause=exc,
            )
        return ProviderError(
            self.name, endpoint.model_id, message=_short(exc),
            status=status, cause=exc,
        )

    # -- calling ------------------------------------------------------------ #

    def _bind(self, model: Any, kwargs: dict[str, Any]) -> Any:
        """Apply tool binding through the provider's own converter.

        Tools are kept in their original form all the way down so each provider
        translates them itself, rather than the router guessing a wire format
        that happens to suit one provider.
        """
        tools = kwargs.pop("tools", None)
        if not tools:
            return model
        bind_kwargs = {}
        for key in ("tool_choice", "parallel_tool_calls", "strict"):
            if key in kwargs:
                bind_kwargs[key] = kwargs.pop(key)
        return model.bind_tools(tools, **bind_kwargs)

    def pop_rate_limit(self) -> Mapping[str, int]:
        """Rate-limit counters from the most recent response, consumed once.

        Only providers that capture headers return anything. Reading clears the
        slot so a later call cannot be charged with an earlier one's numbers.
        """
        if not self.captures_headers:
            return {}
        headers = _last_headers.get()
        if not headers:
            return {}
        _last_headers.set(None)
        return rate_limit_snapshot(headers)

    def _begin(self) -> None:
        """Drop any headers left over from a previous call on this thread."""
        if self.captures_headers:
            _last_headers.set(None)

    def _finish(
        self, message: AIMessage, endpoint: Endpoint, parsed: Any = None
    ) -> ProviderResponse:
        return ProviderResponse(
            message=message,
            endpoint=endpoint,
            tokens=tokens_used(message),
            rate_limit=self.pop_rate_limit(),
            parsed=parsed,
        )

    def _structure(self, model: Any, endpoint: Endpoint, schema: Any) -> Any:
        """Bind a schema using the method this endpoint actually supports.

        The method is a property of the model, not of the request: Groq's
        gpt-oss models need `json_schema` because they answer in prose under
        function calling, and Gemma on Ollama needs it because its tool calling
        is too weak to rely on. Since the ladder only learns which endpoint it
        is using at call time, this has to happen here rather than once at the
        top - binding a single method for a whole ladder gets it wrong for at
        least one provider in that ladder.

        `include_raw=True` keeps the AIMessage alongside the parsed object, so
        token counts and rate-limit headers are read from the same place on both
        paths. It also turns a parse failure into a value rather than an
        exception, which is why `_parsed_or_raise` re-raises it below: for a
        router a schema violation is a failed candidate, not a failed call.
        """
        try:
            return model.with_structured_output(
                schema,
                method=endpoint.structured_output_method,
                include_raw=True,
            )
        except (NotImplementedError, ValueError, TypeError) as exc:
            # A wrapper that does not implement the configured method would
            # otherwise fail identically on every retry. Say which endpoint and
            # which method, since the fix is a one-line config edit.
            raise ProviderError(
                self.name,
                endpoint.model_id,
                message=(
                    f"structured output method "
                    f"{endpoint.structured_output_method!r} is not supported by "
                    f"{self.package}; set structured_output_method for provider "
                    f"{self.name!r} in limits.yaml"
                ),
                cause=exc,
            ) from exc

    @staticmethod
    def _parsed_or_raise(result: Any) -> tuple[AIMessage, Any]:
        """Split an include_raw result, treating a parse failure as an error."""
        if not isinstance(result, Mapping):
            # A wrapper that ignored include_raw returned the object directly.
            return _as_ai_message(result), result
        error = result.get("parsing_error")
        if error is not None:
            raise error if isinstance(error, BaseException) else ValueError(str(error))
        return _as_ai_message(result.get("raw")), result.get("parsed")

    def invoke(
        self, endpoint: Endpoint, messages: Sequence[BaseMessage], **kwargs: Any
    ) -> ProviderResponse:
        self._begin()
        call_kwargs = dict(kwargs)
        schema = call_kwargs.pop(STRUCTURED_OUTPUT_KWARG, None)
        model = self.chat_model(
            endpoint, **_constructor_kwargs(call_kwargs, self.model_class())
        )
        runnable = self._bind(model, call_kwargs)
        if schema is not None:
            runnable = self._structure(runnable, endpoint, schema)
        try:
            result = runnable.invoke(list(messages), **call_kwargs)
        except RouterError:
            raise
        except Exception as exc:
            raise self.interpret(exc, endpoint) from exc
        if schema is None:
            return self._finish(_as_ai_message(result), endpoint)
        message, parsed = self._parsed_or_raise(result)
        return self._finish(message, endpoint, parsed)

    async def ainvoke(
        self, endpoint: Endpoint, messages: Sequence[BaseMessage], **kwargs: Any
    ) -> ProviderResponse:
        self._begin()
        call_kwargs = dict(kwargs)
        schema = call_kwargs.pop(STRUCTURED_OUTPUT_KWARG, None)
        model = self.chat_model(
            endpoint, **_constructor_kwargs(call_kwargs, self.model_class())
        )
        runnable = self._bind(model, call_kwargs)
        if schema is not None:
            runnable = self._structure(runnable, endpoint, schema)
        try:
            result = await runnable.ainvoke(list(messages), **call_kwargs)
        except RouterError:
            raise
        except Exception as exc:
            raise self.interpret(exc, endpoint) from exc
        if schema is None:
            return self._finish(_as_ai_message(result), endpoint)
        message, parsed = self._parsed_or_raise(result)
        return self._finish(message, endpoint, parsed)

    def stream(
        self, endpoint: Endpoint, messages: Sequence[BaseMessage], **kwargs: Any
    ) -> Iterator[AIMessageChunk]:
        """Stream chunks.

        The first chunk is pulled eagerly so a rate limit surfaces before the
        router has committed to this endpoint and can still fall back.
        """
        self._begin()
        call_kwargs = dict(kwargs)
        # Streaming a schema yields one object at the end, not a token stream,
        # so there is nothing to stream. Refuse rather than silently returning
        # unstructured chunks the caller will try to parse.
        if call_kwargs.pop(STRUCTURED_OUTPUT_KWARG, None) is not None:
            raise ProviderError(
                self.name,
                endpoint.model_id,
                message="structured output cannot be streamed; use invoke instead",
            )
        model = self.chat_model(
            endpoint, **_constructor_kwargs(call_kwargs, self.model_class())
        )
        runnable = self._bind(model, call_kwargs)
        try:
            iterator = iter(runnable.stream(list(messages), **call_kwargs))
            first = next(iterator, None)
        except RouterError:
            raise
        except Exception as exc:
            raise self.interpret(exc, endpoint) from exc

        def generate() -> Iterator[AIMessageChunk]:
            if first is not None:
                yield first
            try:
                yield from iterator
            except RouterError:
                raise
            except Exception as exc:
                raise self.interpret(exc, endpoint) from exc

        return generate()


class GroqProvider(BaseProvider):
    name = "groq"
    package = "langchain-groq"
    captures_headers = True

    def _model_class(self) -> type:
        from langchain_groq import ChatGroq

        return ChatGroq

    def _construct(self, endpoint: Endpoint, **kwargs: Any) -> Any:
        ChatGroq = self._model_class()

        http_client, http_async_client = _build_http_clients()
        return ChatGroq(
            model=endpoint.model_id,
            api_key=endpoint.api_key,
            http_client=http_client,
            http_async_client=http_async_client,
            **kwargs,
        )

    def interpret(self, exc: BaseException, endpoint: Endpoint) -> RouterError:
        # Groq's 498 "flex tier capacity exceeded" is not in any HTTP registry
        # and some SDK layers surface it only in the message text, so it is
        # matched on text as well as status. It is capacity, not a hard failure.
        if status_of(exc) not in RATE_LIMIT_STATUSES:
            text = str(exc).lower()
            if "498" in text and ("capacity" in text or "flex" in text):
                return RateLimited(
                    self.name, endpoint.model_id,
                    retry_after=retry_after_from_headers(headers_of(exc)),
                    message="flex tier capacity exceeded", status=498,
                )
        return super().interpret(exc, endpoint)


class OllamaProvider(BaseProvider):
    """A local Ollama daemon. The only unmetered endpoint in the ladder.

    Deliberately keyless: its limits.yaml block declares no `api_key_env`, so
    `Endpoint.has_credentials` is true and `chat_model()` skips the auth check
    that every hosted provider goes through. That is what lets it stay in the
    ladder on a machine with no API keys configured at all.

    Its value is not speed - it is roughly an order of magnitude slower than
    Groq - but that it costs nothing and cannot rate-limit. High-volume, low-
    judgment work belongs here so the hosted free tiers are spent on the calls
    that genuinely need a better model.
    """

    name = "ollama"
    package = "langchain-ollama"

    #: Matches Ollama's own default. Overridden by OLLAMA_BASE_URL.
    default_base_url = "http://localhost:11434"

    def _model_class(self) -> type:
        from langchain_ollama import ChatOllama

        return ChatOllama

    def _construct(self, endpoint: Endpoint, **kwargs: Any) -> Any:
        ChatOllama = self._model_class()

        # A caller-supplied base_url wins, so a test server or a remote daemon
        # can be pointed at without touching the environment.
        kwargs.setdefault(
            "base_url", os.environ.get("OLLAMA_BASE_URL") or self.default_base_url
        )
        return ChatOllama(model=endpoint.model_id, **kwargs)

    def interpret(self, exc: BaseException, endpoint: Endpoint) -> RouterError:
        """A daemon that is not running is transient, not fatal.

        Ollama is a local process the operator may simply not have started. That
        surfaces as a connection refusal, which `_looks_transient` already
        catches - but a 404 from a model that was never pulled would otherwise
        be read as a permanently missing model and park the endpoint for an
        hour. It is one `ollama pull` away, so it is reported as transient too.
        """
        if status_of(exc) == 404:
            return ProviderError(
                self.name,
                endpoint.model_id,
                message=(
                    f"{endpoint.model_id!r} is not pulled on this Ollama daemon; "
                    f"run `ollama pull {endpoint.model_id}`"
                ),
                status=404,
                transient=True,
                cause=exc,
            )
        return super().interpret(exc, endpoint)


class MistralProvider(BaseProvider):
    name = "mistral"
    package = "langchain-mistralai"

    def _model_class(self) -> type:
        from langchain_mistralai import ChatMistralAI

        return ChatMistralAI

    def _construct(self, endpoint: Endpoint, **kwargs: Any) -> Any:
        ChatMistralAI = self._model_class()

        return ChatMistralAI(
            model=endpoint.model_id, api_key=endpoint.api_key, **kwargs
        )


class GoogleAIStudioProvider(BaseProvider):
    name = "google_ai_studio"
    package = "langchain-google-genai"

    def _model_class(self) -> type:
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI

    def _construct(self, endpoint: Endpoint, **kwargs: Any) -> Any:
        ChatGoogleGenerativeAI = self._model_class()

        return ChatGoogleGenerativeAI(
            model=endpoint.model_id, google_api_key=endpoint.api_key, **kwargs
        )

    def interpret(self, exc: BaseException, endpoint: Endpoint) -> RouterError:
        """Read Google's QuotaFailure / RetryInfo out of the error body.

        A 429 here is RESOURCE_EXHAUSTED and the body names the exact limit that
        was hit in QuotaFailure.violations[].quotaId, sometimes with a
        retryDelay. Both are worth surfacing: the quota is per project, so
        knowing which limit tripped is the difference between "wait a minute"
        and "you are done for the day".
        """
        status = status_of(exc)
        is_quota = status in RATE_LIMIT_STATUSES or "RESOURCE_EXHAUSTED" in str(exc)
        if not is_quota:
            return super().interpret(exc, endpoint)

        retry_after = retry_after_from_headers(headers_of(exc))
        quota_id = None
        for detail in _google_error_details(exc):
            if not isinstance(detail, Mapping):
                continue
            type_url = str(detail.get("@type", ""))
            if "QuotaFailure" in type_url:
                violations = detail.get("violations") or []
                if violations and isinstance(violations[0], Mapping):
                    quota_id = violations[0].get("quotaId") or violations[0].get("subject")
            elif "RetryInfo" in type_url:
                retry_after = parse_duration(detail.get("retryDelay")) or retry_after
        return RateLimited(
            self.name, endpoint.model_id, retry_after=retry_after,
            message="resource exhausted", status=status or 429, quota_id=quota_id,
        )


class OpenAICompatibleProvider(BaseProvider):
    """Any provider that speaks the OpenAI chat-completions wire format.

    OpenRouter and NVIDIA NIM both expose an OpenAI-compatible endpoint, so both
    are `langchain-openai`'s ChatOpenAI pointed at a different `base_url` rather
    than a bespoke SDK each. That also means both get header capture for free:
    ChatOpenAI accepts a custom httpx client, so the same event hook the Groq
    wrapper uses can stash x-ratelimit-* for the ledger to read back.

    Subclasses set `base_url`, and `http_timeout` when the default is too short.
    """

    package = "langchain-openai"
    captures_headers = True
    base_url: str = ""
    http_timeout: float = DEFAULT_HTTP_TIMEOUT

    def _model_class(self) -> type:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI

    def _construct(self, endpoint: Endpoint, **kwargs: Any) -> Any:
        ChatOpenAI = self._model_class()

        # base_url is a default, not a fixture: a caller pointing at a proxy,
        # a self-hosted NIM or a test server passes their own and it must win
        # rather than collide with this class's.
        kwargs.setdefault("base_url", self.base_url)
        http_client, http_async_client = _build_http_clients(self.http_timeout)
        return ChatOpenAI(
            model=endpoint.model_id,
            api_key=endpoint.api_key,
            http_client=http_client,
            http_async_client=http_async_client,
            **kwargs,
        )


class OpenRouterProvider(OpenAICompatibleProvider):
    name = "openrouter"
    base_url = "https://openrouter.ai/api/v1"

    #: How long to park the account after a 402. Out of credit is a real state
    #: change, not a busy minute, so retrying on the normal 60s cooldown would
    #: just spend the ladder's attempt budget on a certain failure every minute.
    NO_CREDIT_COOLDOWN = 3600.0

    def interpret(self, exc: BaseException, endpoint: Endpoint) -> RouterError:
        """Normalise the two OpenRouter-specific quota shapes.

        402 "insufficient credits" is quota, not a bad request - but the router
        treats a non-transient 4xx as fatal and aborts the whole call, so
        leaving it as a ProviderError would take down a request the rest of the
        ladder could still serve. It comes back as RateLimited with a long
        retry instead, which parks OpenRouter and steps to the next candidate.

        429s carry `x-ratelimit-reset` as an absolute epoch, not a duration, so
        the generic header parser reads it as a nonsensical number of seconds.
        """
        status = status_of(exc)
        headers = headers_of(exc)
        if status == 402:
            return RateLimited(
                self.name, endpoint.model_id,
                retry_after=self.NO_CREDIT_COOLDOWN,
                message="out of credits", status=402,
            )
        if status in RATE_LIMIT_STATUSES:
            retry_after = (
                retry_after_from_headers(headers) or _reset_at_to_delay(headers)
            )
            return RateLimited(
                self.name, endpoint.model_id, retry_after=retry_after,
                message=_short(exc), status=status,
            )
        return super().interpret(exc, endpoint)


class NvidiaNimProvider(OpenAICompatibleProvider):
    """NIM via ChatOpenAI rather than langchain-nvidia-ai-endpoints' ChatNVIDIA.

    ChatNVIDIA is the official wrapper and posts to this same base_url with the
    same OpenAI-shaped payload, but it costs two things this router needs:

    - It builds its own `requests.Session` and takes no `http_client`, so the
      httpx event hook that feeds `x-ratelimit-*` into the ledger cannot be
      installed at all.
    - Its `_finalize()` falls back to fetching the live catalogue for any model
      absent from the static table it ships, which puts a network round trip and
      a UserWarning inside model construction - on the routing hot path, for
      exactly the recent models this config uses.

    The one thing ChatNVIDIA offers that a plain base_url cannot is per-model
    custom endpoints, and every model that needs one is a vision, embedding or
    reranking model. None are chat models, so none are routable here.
    """

    name = "nvidia_nim"
    base_url = "https://integrate.api.nvidia.com/v1"

    #: NIM's build.nvidia.com endpoints are shared public infrastructure and
    #: their own docs warn calls can be slow or time out under load, so this is
    #: deliberately more patient than the 60s every other provider gets. A
    #: timeout still reads as transient, so the ladder steps past it either way -
    #: this only stops a slow-but-working answer being thrown away.
    http_timeout = 120.0


def _reset_at_to_delay(headers: Mapping[str, str]) -> float | None:
    """Seconds until an absolute `x-ratelimit-reset` timestamp.

    OpenRouter reports when the window resets rather than how long to wait, in
    epoch milliseconds. Values are sanity-checked against the clock: a header
    that is already in the past, or absurdly far ahead, is discarded so the
    caller falls back to the configured cooldown rather than trusting garbage.
    """
    lowered = {str(k).lower(): v for k, v in headers.items()}
    raw = _as_int(lowered.get("x-ratelimit-reset"))
    if raw is None:
        return None
    now = time.time()
    # Epoch seconds until ~2286; anything larger is milliseconds.
    reset_at = raw / 1000.0 if raw > 1e11 else float(raw)
    delay = reset_at - now
    if delay <= 0 or delay > 86400:
        return None
    return delay


def _google_error_details(exc: BaseException) -> list[Any]:
    body = getattr(exc, "details", None)
    if isinstance(body, Mapping):
        error = body.get("error")
        if isinstance(error, Mapping):
            details = error.get("details")
            if isinstance(details, list):
                return details
        details = body.get("details")
        if isinstance(details, list):
            return details
    if isinstance(body, list):
        return body
    return []


# --------------------------------------------------------------------------- #
# Registry of providers
# --------------------------------------------------------------------------- #

_PROVIDER_CLASSES: dict[str, type[BaseProvider]] = {
    "groq": GroqProvider,
    "ollama": OllamaProvider,
    "mistral": MistralProvider,
    "google_ai_studio": GoogleAIStudioProvider,
    "nvidia_nim": NvidiaNimProvider,
    "openrouter": OpenRouterProvider,
}

_instances: dict[str, BaseProvider] = {}
_instances_lock = threading.Lock()


def register_provider(name: str, provider_class: type[BaseProvider]) -> None:
    """Add a provider at runtime. Adding one to v0 is this plus config."""
    with _instances_lock:
        _PROVIDER_CLASSES[name] = provider_class
        _instances.pop(name, None)


def supported_providers() -> tuple[str, ...]:
    return tuple(_PROVIDER_CLASSES)


def get_provider(name: str) -> BaseProvider:
    with _instances_lock:
        provider = _instances.get(name)
        if provider is None:
            try:
                provider_class = _PROVIDER_CLASSES[name]
            except KeyError:
                raise ProviderError(
                    name,
                    message=(
                        f"no wrapper for provider {name!r}; "
                        f"known providers: {', '.join(sorted(_PROVIDER_CLASSES))}"
                    ),
                ) from None
            provider = provider_class()
            _instances[name] = provider
        return provider


def reset_providers() -> None:
    """Drop cached chat models. Used by tests and after credentials change."""
    with _instances_lock:
        _instances.clear()


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #

#: Always sent with the call, never to the constructor, even when the model
#: class happens to have a field of the same name - these describe one request.
_ALWAYS_PER_CALL = frozenset({
    "stop", "tools", "tool_choice", "functions", "function_call",
    "response_format", "parallel_tool_calls", "strict", "stream_usage",
    "config", "callbacks", "run_name", "run_id", "metadata", "tags",
    STRUCTURED_OUTPUT_KWARG,
})


@lru_cache(maxsize=None)
def _constructor_names(model_class: type) -> frozenset[str]:
    """Every name a chat model class accepts in its constructor.

    Read off the pydantic model rather than hardcoded, because the split is
    different for every provider: `base_url` and `top_k` are constructor
    settings on some classes and unknown on others, and guessing wrong sends a
    setting into the request body where it becomes a 400.
    """
    names: set[str] = set()
    for field_name, model_field in getattr(model_class, "model_fields", {}).items():
        names.add(field_name)
        alias = getattr(model_field, "alias", None)
        if isinstance(alias, str):
            names.add(alias)
        validation_alias = getattr(model_field, "validation_alias", None)
        if isinstance(validation_alias, str):
            names.add(validation_alias)
    return frozenset(names)


def _constructor_kwargs(
    call_kwargs: dict[str, Any], model_class: type | None
) -> dict[str, Any]:
    """Split constructor settings out of the per-call kwargs, in place."""
    if model_class is None:
        return {}
    accepted = _constructor_names(model_class)
    return {
        key: call_kwargs.pop(key)
        for key in list(call_kwargs)
        if key in accepted and key not in _ALWAYS_PER_CALL
    }


def _freeze(value: Any) -> Any:
    """Hashable form of arbitrary kwargs, for the model cache key."""
    if isinstance(value, Mapping):
        return tuple(sorted((k, _freeze(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple, set)):
        return tuple(_freeze(v) for v in value)
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def _as_ai_message(value: Any) -> AIMessage:
    if isinstance(value, AIMessage):
        return value
    return AIMessage(content=str(getattr(value, "content", value)))


def _short(exc: BaseException, limit: int = 240) -> str:
    text = " ".join(str(exc).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


_TRANSIENT_HINTS = ("timeout", "timed out", "connection", "temporarily", "overloaded")


def _looks_transient(exc: BaseException) -> bool:
    name = type(exc).__name__.lower()
    if "timeout" in name or "connection" in name:
        return True
    text = str(exc).lower()
    return any(hint in text for hint in _TRANSIENT_HINTS)
