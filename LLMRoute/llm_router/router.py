"""route() - the single callable - and LLMRouter, its LangChain adapter.

    from llm_router import route
    response = route(messages, strategy="free_first")

    from llm_router import LLMRouter
    agent = create_react_agent(model=LLMRouter(strategy="sticky"), tools=tools)

LLMRouter is a thin BaseChatModel over route(): all the routing lives in one
place, and the class only adapts it to the Runnable interface.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Iterator, Mapping, Sequence

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.messages.utils import convert_to_messages
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import RunnableLambda
from pydantic import ConfigDict, Field

from .ladder import Candidate, explain, resolve_candidates, soonest_retry
from .ledger import UsageLedger
from .policies import SessionState, get_policy, signals_downgrade
from .providers import (
    STRUCTURED_OUTPUT_KWARG,
    AllCandidatesExhausted,
    AuthError,
    ProviderError,
    ProviderResponse,
    RateLimited,
    get_provider,
    tokens_used,
)
from .registry import Endpoint, Registry, default_registry

logger = logging.getLogger("llm_router")

#: How many endpoints one call will try before giving up. Bounds worst-case
#: latency: without it a bad moment could walk a dozen providers in series.
#:
#: **It has to clear the biggest tier**, or the tail of the ladder becomes
#: unreachable: the cap is applied to the *ordered candidate list* before
#: availability is checked, so endpoints the ledger has already parked still
#: consume a slot. A tier holding more endpoints than this silently loses its
#: last candidates, and the downgrade it is entitled to never happens.
#:
#: This has now bitten twice. First at 6, one short of tier S. Then at 8, when
#: tier B grew to 14 by gaining a local Ollama endpoint at priority 950 - which
#: put the free, unmetered backstop at position 14 of 14, six places beyond
#: reach. It failed silently and in exactly the circumstance it exists for:
#: every hosted endpoint rate-limited is precisely when the local model should
#: take over.
#:
#: 16 clears the largest tier today (A, at 15). The worst case is cheaper than
#: it looks - a parked candidate is skipped without a network call, so the
#: expensive path is fifteen genuine failures, which is pathological rather than
#: routine. `test_max_attempts_clears_every_tier` fails the suite if a tier
#: grows past this again, so the next provider added cannot repeat it.
DEFAULT_MAX_ATTEMPTS = 16

#: Assumed output length when the caller did not set max_tokens. Only used to
#: pre-check token quotas, never sent to the provider.
ASSUMED_OUTPUT_TOKENS = 1024

#: Cooldowns applied to an endpoint after non-quota failures.
TRANSIENT_COOLDOWN = 20.0
AUTH_COOLDOWN = 900.0
MISSING_MODEL_COOLDOWN = 3600.0


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #

@dataclass
class Attempt:
    """One endpoint the router tried or skipped, and what happened."""

    endpoint: str
    tier: str
    step: int
    outcome: str            # "success" | "rate_limited" | "error" | "skipped"
    detail: str | None = None
    retry_after: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "endpoint": self.endpoint, "tier": self.tier, "step": self.step,
            "outcome": self.outcome, "detail": self.detail,
            "retry_after": self.retry_after,
        }


@dataclass
class RouteResult:
    """A successful call plus everything about how it was routed."""

    message: AIMessage
    endpoint: Endpoint
    candidate: Candidate
    requested_model: str | None
    requested_tier: str | None
    strategy: str
    tokens: int | None = None
    latency: float = 0.0
    attempts: list[Attempt] = field(default_factory=list)
    #: The schema instance when structured output was requested, else None.
    parsed: Any = None

    @property
    def tier_downgraded(self) -> bool:
        return self.candidate.tier_downgraded

    @property
    def content(self) -> Any:
        return self.message.content

    def as_metadata(self) -> dict[str, Any]:
        return {
            "provider": self.endpoint.provider,
            "model_id": self.endpoint.model_id,
            "logical_model": self.endpoint.logical_model,
            "tier": self.endpoint.tier,
            "requested_model": self.requested_model,
            # The floor that was actually in force: an explicit tier= if given,
            # otherwise the requested model's own tier. Reporting the bare
            # argument would say "requested_tier: None" on exactly the calls
            # where a downgrade happened, which is when it matters most.
            "requested_tier": self.candidate.requested_tier or self.requested_tier,
            "tier_downgraded": self.tier_downgraded,
            "ladder_step": self.candidate.step,
            "reason": self.candidate.reason,
            "strategy": self.strategy,
            "tokens": self.tokens,
            "latency": round(self.latency, 3),
            "attempts": [a.as_dict() for a in self.attempts],
        }


# --------------------------------------------------------------------------- #
# Process-wide defaults
# --------------------------------------------------------------------------- #

_default_ledger: UsageLedger | None = None
_default_sessions: SessionState | None = None
_defaults_lock = threading.Lock()
_ledger_path: str | None = None


def default_ledger() -> UsageLedger:
    """The ledger route() uses when none is passed.

    One per process, so every call site shares a single view of what has been
    spent - which is the whole point of tracking quota that the provider will
    not tell us about.
    """
    global _default_ledger
    if _default_ledger is None:
        with _defaults_lock:
            if _default_ledger is None:
                ledger = UsageLedger()
                if _ledger_path:
                    ledger.load(_ledger_path)
                _default_ledger = ledger
    return _default_ledger


def default_sessions() -> SessionState:
    global _default_sessions
    if _default_sessions is None:
        with _defaults_lock:
            if _default_sessions is None:
                _default_sessions = SessionState()
    return _default_sessions


def configure(*, ledger_path: str | None = None) -> None:
    """Persist the shared ledger across restarts.

    Worth doing for daily and weekly caps: a fresh process with an empty ledger
    will happily respend a quota it already used, and only find out via 429s.
    """
    global _ledger_path
    _ledger_path = ledger_path
    if ledger_path and _default_ledger is not None:
        _default_ledger.load(ledger_path)


def reset_state() -> None:
    """Drop the shared ledger and session state. For tests."""
    global _default_ledger, _default_sessions
    with _defaults_lock:
        _default_ledger = None
        _default_sessions = None


def _persist(ledger: UsageLedger) -> None:
    if _ledger_path and ledger is _default_ledger:
        try:
            ledger.save(_ledger_path)
        except OSError as exc:  # pragma: no cover - never fail a call over this
            logger.warning("could not persist ledger: %s", exc)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

MessagesLike = str | Sequence[Any] | BaseMessage


def normalize_messages(messages: MessagesLike) -> list[BaseMessage]:
    """Accept a string, one message, or any LangChain-ish message sequence."""
    if isinstance(messages, str):
        return convert_to_messages([("human", messages)])
    if isinstance(messages, BaseMessage):
        return [messages]
    return list(convert_to_messages(list(messages)))


def estimate_tokens(messages: Sequence[BaseMessage], max_tokens: int | None) -> int:
    """Rough token estimate, used only to pre-check token quotas.

    Deliberately crude and deliberately high: the cost of over-estimating is one
    extra hop down the ladder, the cost of under-estimating is a 429.
    """
    characters = 0
    for message in messages:
        content = message.content
        if isinstance(content, str):
            characters += len(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, str):
                    characters += len(part)
                elif isinstance(part, Mapping):
                    characters += len(str(part.get("text", "")))
    return characters // 4 + int(max_tokens or ASSUMED_OUTPUT_TOKENS)


def _providers_filter(provider: str | Sequence[str] | None) -> list[str] | None:
    if provider is None:
        return None
    if isinstance(provider, str):
        return [provider]
    return list(provider)


# --------------------------------------------------------------------------- #
# The router
# --------------------------------------------------------------------------- #

class _Walk:
    """Drives one routing call: plan the ladder, hand out candidates, record
    what happened.

    route(), aroute() and route_stream() differ only in how they make the call -
    sync, await, or open a stream - so everything else lives here. Keeping one
    copy matters: the bookkeeping (reserve a slot, decide whether a failure is
    fatal, record the outcome, work out how long to wait) is exactly the part
    that silently drifts apart when it is written out three times.
    """

    def __init__(
        self,
        *,
        messages: Sequence[BaseMessage],
        model: str | None,
        provider: str | Sequence[str] | None,
        tier: str | None,
        strategy: str | Any,
        session_id: str | None,
        registry: Registry,
        ledger: UsageLedger,
        session_state: SessionState,
        streaming: bool,
        allow_tier_downgrade: bool,
        max_attempts: int,
        max_wait: float,
        call_kwargs: Mapping[str, Any],
    ) -> None:
        self.messages = messages
        self.model = model
        self.provider = provider
        self.tier = tier
        self.strategy = strategy
        self.session_id = session_id
        self.registry = registry
        self.ledger = ledger
        self.session_state = session_state
        self.streaming = streaming
        # Derived from the call rather than passed in: a structured request is
        # identified by carrying a schema, and an endpoint that cannot produce
        # schema-conforming output is not a fallback for one - it is a
        # guaranteed 400. Filtering it out of the ladder is cheaper and clearer
        # than discovering that per candidate.
        self.structured = call_kwargs.get(STRUCTURED_OUTPUT_KWARG) is not None
        self.allow_tier_downgrade = allow_tier_downgrade
        self.max_attempts = max_attempts
        self.max_tokens = call_kwargs.get("max_tokens")
        self.started = time.monotonic()
        self.deadline = self.started + max_wait

        self.strategy_name = ""
        self.ordered: list[Candidate] = []
        self.everything: list[Candidate] = []
        self.estimate = 0
        self.attempts: list[Attempt] = []

    def plan(self) -> None:
        """Resolve the ladder and put it in policy order."""
        strategy_name, policy = get_policy(self.strategy)
        self.strategy_name = strategy_name
        self.estimate = estimate_tokens(self.messages, self.max_tokens)

        # An explicit provider+model pins the exact endpoint: the caller asked
        # for that one, so answering from somewhere else would be a lie.
        pinned = self.provider is not None and self.model is not None

        everything = resolve_candidates(
            self.model,
            self.tier,
            self.ledger,
            registry=self.registry,
            providers=_providers_filter(self.provider),
            streaming=self.streaming,
            structured=self.structured,
            estimated_tokens=self.estimate,
            allow_tier_downgrade=self.allow_tier_downgrade and not pinned,
            include_unavailable=True,
        )
        if pinned:
            everything = [c for c in everything if c.step == 1]

        self.everything = everything
        live = [c for c in everything if c.available]
        self.ordered = policy(live, self.session_id, self.session_state)
        self.attempts = []

    def acquirable(self) -> Iterator[Candidate]:
        """Yield candidates that have quota, reserving a slot for each.

        The reservation is atomic, so two concurrent callers cannot both win the
        last free request on an endpoint.
        """
        for candidate in self.ordered[:self.max_attempts]:
            endpoint = candidate.endpoint
            verdict = self.ledger.try_acquire(
                endpoint, estimated_tokens=self.estimate
            )
            if not verdict.ok:
                self.attempts.append(Attempt(
                    endpoint.key, endpoint.tier, candidate.step, "skipped",
                    verdict.reason, verdict.retry_after,
                ))
                continue
            yield candidate

    def failed(self, candidate: Candidate, exc: Exception) -> None:
        """Record a failed attempt, or re-raise if retrying cannot help."""
        # A malformed request is the caller's bug, not the endpoint's: parking a
        # healthy endpoint over it would punish the wrong thing, and walking the
        # whole ladder would burn a dozen quotas on the same bad request.
        if _is_fatal(exc):
            raise exc
        endpoint = candidate.endpoint
        outcome, detail, wait = _record_failure(exc, endpoint, self.ledger)
        self.attempts.append(Attempt(
            endpoint.key, endpoint.tier, candidate.step, outcome, detail, wait
        ))
        logger.info("%s failed (%s), trying next candidate", endpoint.key, outcome)

    def succeeded(
        self, response: ProviderResponse, candidate: Candidate
    ) -> RouteResult:
        return _succeed(
            response, candidate, self.ledger, self.session_state, self.session_id,
            self.strategy_name, self.model, self.tier, self.attempts,
            time.monotonic() - self.started,
        )

    def should_wait(self) -> float | None:
        """Seconds worth waiting for the soonest endpoint, or None to give up."""
        wait = soonest_retry(self.everything)
        remaining = self.deadline - time.monotonic()
        if wait is None or remaining <= 0 or wait > remaining:
            return None
        return min(wait, remaining) + 0.05

    def give_up(self) -> AllCandidatesExhausted:
        return _exhausted(
            self.everything, self.attempts, self.model, self.tier, self.ledger
        )


def _walk(
    messages: Sequence[BaseMessage],
    *,
    strategy: str | Any,
    model: str | None,
    provider: str | Sequence[str] | None,
    tier: str | None,
    session_id: str | None,
    allow_tier_downgrade: bool,
    max_attempts: int,
    max_wait: float,
    registry: Registry | None,
    ledger: UsageLedger | None,
    session_state: SessionState | None,
    streaming: bool,
    call_kwargs: Mapping[str, Any],
) -> _Walk:
    return _Walk(
        messages=messages, model=model, provider=provider, tier=tier,
        strategy=strategy, session_id=session_id,
        registry=registry or default_registry(),
        ledger=ledger or default_ledger(),
        session_state=session_state or default_sessions(),
        streaming=streaming, allow_tier_downgrade=allow_tier_downgrade,
        max_attempts=max_attempts, max_wait=max_wait, call_kwargs=call_kwargs,
    )


def _record_failure(
    exc: Exception, endpoint: Endpoint, ledger: UsageLedger
) -> tuple[str, str, float | None]:
    """Update the ledger for a failed attempt. Returns (outcome, detail, wait)."""
    if isinstance(exc, RateLimited):
        wait = ledger.record_rate_limited(
            endpoint, exc.retry_after, reason=str(exc)
        )
        return "rate_limited", str(exc), wait
    if isinstance(exc, AuthError):
        # Bad credentials will not fix themselves inside one run.
        ledger.record_unavailable(endpoint, AUTH_COOLDOWN, reason=str(exc))
        return "error", str(exc), AUTH_COOLDOWN
    if isinstance(exc, ProviderError) and exc.status == 404:
        # Almost always a stale model id in models.yaml. Park this endpoint and
        # keep going, rather than failing a call the ladder can still serve.
        logger.warning(
            "%s rejected model id %r as unknown - check models.yaml against "
            "`python scripts/verify_models.py`", endpoint.provider, endpoint.model_id,
        )
        ledger.record_unavailable(endpoint, MISSING_MODEL_COOLDOWN, reason=str(exc))
        return "error", str(exc), MISSING_MODEL_COOLDOWN
    ledger.record_unavailable(endpoint, TRANSIENT_COOLDOWN, reason=str(exc))
    return "error", str(exc), TRANSIENT_COOLDOWN


def _is_fatal(exc: Exception) -> bool:
    """A request that is wrong everywhere should fail once, not twelve times."""
    return (
        isinstance(exc, ProviderError)
        and not exc.transient
        and not isinstance(exc, AuthError)
        and exc.status is not None
        and exc.status not in (404, 408, 409)
        and 400 <= exc.status < 500
    )


def _exhausted(
    everything: Sequence[Candidate],
    attempts: Sequence[Attempt],
    model: str | None,
    tier: str | None,
    ledger: UsageLedger,
) -> AllCandidatesExhausted:
    """Explain the failure using the ledger as it stands *now*.

    The candidate list was resolved before the call loop ran, so its
    availability is stale by the time everything has failed - it would cheerfully
    report a dozen "available" endpoints that had just 429'd. Re-reading the
    ledger here is what makes the message, and the retry_after a caller backs
    off on, actually true.
    """
    everything = [
        Candidate(
            endpoint=candidate.endpoint,
            step=candidate.step,
            tier_downgraded=candidate.tier_downgraded,
            requested_tier=candidate.requested_tier,
            availability=ledger.availability(candidate.endpoint),
        )
        for candidate in everything
    ]
    target = model or (f"tier {tier}" if tier else "any model")
    wait = soonest_retry(everything)
    lines = [f"no endpoint available for {target}."]
    if wait is not None:
        lines.append(f"soonest retry in {wait:.0f}s.")
    if everything:
        lines.append("candidates:\n" + explain(everything))
    else:
        lines.append(
            "no candidates at all - check that the provider is enabled in "
            "limits.yaml and its API key is set."
        )
    return AllCandidatesExhausted(
        " ".join(lines[:2]) + "\n" + "\n".join(lines[2:]),
        attempts=[a.as_dict() for a in attempts],
        retry_after=wait,
    )


def _succeed(
    response: ProviderResponse,
    candidate: Candidate,
    ledger: UsageLedger,
    session_state: SessionState,
    session_id: str | None,
    strategy_name: str,
    requested_model: str | None,
    requested_tier: str | None,
    attempts: list[Attempt],
    latency: float,
) -> RouteResult:
    """Charge the ledger, pin the session, and build the result."""
    endpoint = candidate.endpoint
    ledger.record_tokens(endpoint, response.tokens)
    if response.rate_limit:
        # Groq tells us what is actually left; that outranks our own count.
        ledger.sync_from_headers(endpoint, **response.rate_limit)
    session_state.remember(session_id, candidate)
    _persist(ledger)

    attempts.append(Attempt(
        endpoint.key, endpoint.tier, candidate.step, "success",
        detail=f"{response.tokens} tokens" if response.tokens else None,
    ))
    result = RouteResult(
        message=response.message, endpoint=endpoint, candidate=candidate,
        requested_model=requested_model, requested_tier=requested_tier,
        strategy=strategy_name, tokens=response.tokens, latency=latency,
        attempts=attempts, parsed=response.parsed,
    )

    if candidate.tier_downgraded and signals_downgrade(strategy_name):
        # A silent quality drop is exactly what a consistency-sensitive run
        # must not get, so sticky says so out loud as well as in the metadata.
        logger.warning(
            "tier downgrade: %s was requested at tier %s, served by %s at tier %s "
            "(session %s)",
            requested_model or "request", candidate.requested_tier,
            endpoint.key, endpoint.tier, session_id,
        )

    response.message.response_metadata["llm_router"] = result.as_metadata()
    return result


def route(
    messages: MessagesLike,
    *,
    strategy: str | Any = "free_first",
    model: str | None = None,
    provider: str | Sequence[str] | None = None,
    tier: str | None = None,
    session_id: str | None = None,
    return_route: bool = False,
    allow_tier_downgrade: bool = True,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    max_wait: float = 0.0,
    registry: Registry | None = None,
    ledger: UsageLedger | None = None,
    session_state: SessionState | None = None,
    **kwargs: Any,
) -> AIMessage | RouteResult:
    """Send `messages` to the best free endpoint available right now.

    Args:
        messages: a string, a message, or any LangChain message sequence.
        strategy: "free_first", "sticky", or a custom policy callable.
        model: logical model to prefer, e.g. "qwen-27b".
        provider: restrict to a provider. With `model`, pins that exact endpoint.
        tier: quality floor, "S" | "A" | "B".
        session_id: required by "sticky" to keep a run on one endpoint.
        return_route: return a RouteResult instead of the bare AIMessage.
        max_wait: if everything is briefly rate limited, wait up to this many
            seconds for the soonest endpoint rather than failing. Off by default.
        **kwargs: forwarded to the provider (temperature, max_tokens, tools...).

    Returns:
        The AIMessage, with routing details under
        `response.response_metadata["llm_router"]`, or a RouteResult when
        `return_route=True`.

    Raises:
        AllCandidatesExhausted: everything on the ladder is out of quota.
        ProviderError: the request itself is bad, so retrying elsewhere is futile.
    """
    prompt = normalize_messages(messages)
    walk = _walk(
        prompt, strategy=strategy, model=model, provider=provider, tier=tier,
        session_id=session_id, allow_tier_downgrade=allow_tier_downgrade,
        max_attempts=max_attempts, max_wait=max_wait, registry=registry,
        ledger=ledger, session_state=session_state, streaming=False,
        call_kwargs=kwargs,
    )

    while True:
        walk.plan()
        for candidate in walk.acquirable():
            try:
                response = get_provider(candidate.endpoint.provider).invoke(
                    candidate.endpoint, prompt, **kwargs
                )
            except Exception as exc:
                walk.failed(candidate, exc)
                continue
            result = walk.succeeded(response, candidate)
            return result if return_route else result.message

        wait = walk.should_wait()
        if wait is None:
            raise walk.give_up()
        logger.info("all candidates busy; waiting %.1fs for the soonest", wait)
        time.sleep(wait)


async def aroute(
    messages: MessagesLike,
    *,
    strategy: str | Any = "free_first",
    model: str | None = None,
    provider: str | Sequence[str] | None = None,
    tier: str | None = None,
    session_id: str | None = None,
    return_route: bool = False,
    allow_tier_downgrade: bool = True,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    max_wait: float = 0.0,
    registry: Registry | None = None,
    ledger: UsageLedger | None = None,
    session_state: SessionState | None = None,
    **kwargs: Any,
) -> AIMessage | RouteResult:
    """Async twin of route(). Same ladder, same ledger, same semantics."""
    import asyncio

    prompt = normalize_messages(messages)
    walk = _walk(
        prompt, strategy=strategy, model=model, provider=provider, tier=tier,
        session_id=session_id, allow_tier_downgrade=allow_tier_downgrade,
        max_attempts=max_attempts, max_wait=max_wait, registry=registry,
        ledger=ledger, session_state=session_state, streaming=False,
        call_kwargs=kwargs,
    )

    while True:
        walk.plan()
        for candidate in walk.acquirable():
            try:
                response = await get_provider(candidate.endpoint.provider).ainvoke(
                    candidate.endpoint, prompt, **kwargs
                )
            except Exception as exc:
                walk.failed(candidate, exc)
                continue
            result = walk.succeeded(response, candidate)
            return result if return_route else result.message

        wait = walk.should_wait()
        if wait is None:
            raise walk.give_up()
        await asyncio.sleep(wait)


def route_stream(
    messages: MessagesLike,
    *,
    strategy: str | Any = "free_first",
    model: str | None = None,
    provider: str | Sequence[str] | None = None,
    tier: str | None = None,
    session_id: str | None = None,
    allow_tier_downgrade: bool = True,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    registry: Registry | None = None,
    ledger: UsageLedger | None = None,
    session_state: SessionState | None = None,
    **kwargs: Any,
) -> Iterator[AIMessageChunk]:
    """Stream from the best available endpoint.

    Endpoints whose wrapper cannot stream are excluded from the ladder rather
    than discovered at call time. Fallback is only possible up to the first
    chunk - once tokens have been handed to the caller, switching model
    mid-answer would produce nonsense - so the provider wrapper pulls that first
    chunk eagerly and any rate limit surfaces before anything is yielded.
    """
    prompt = normalize_messages(messages)
    walk = _walk(
        prompt, strategy=strategy, model=model, provider=provider, tier=tier,
        session_id=session_id, allow_tier_downgrade=allow_tier_downgrade,
        max_attempts=max_attempts, max_wait=0.0, registry=registry,
        ledger=ledger, session_state=session_state, streaming=True,
        call_kwargs=kwargs,
    )
    walk.plan()

    for candidate in walk.acquirable():
        endpoint = candidate.endpoint
        provider_impl = get_provider(endpoint.provider)
        try:
            chunks = provider_impl.stream(endpoint, prompt, **kwargs)
        except Exception as exc:
            walk.failed(candidate, exc)
            continue

        # The response has begun, so Groq's headers for it are already captured.
        headers = provider_impl.pop_rate_limit()
        if headers:
            walk.ledger.sync_from_headers(endpoint, **headers)

        walk.session_state.remember(session_id, candidate)
        if candidate.tier_downgraded and signals_downgrade(walk.strategy_name):
            logger.warning(
                "tier downgrade while streaming: served by %s at tier %s",
                endpoint.key, endpoint.tier,
            )
        return _stream_and_account(chunks, endpoint, walk.ledger)

    raise walk.give_up()


def _stream_and_account(
    chunks: Iterator[AIMessageChunk], endpoint: Endpoint, ledger: UsageLedger
) -> Iterator[AIMessageChunk]:
    """Yield chunks, then charge the ledger with whatever usage arrived."""
    total = 0
    for chunk in chunks:
        counted = tokens_used(chunk)
        if counted:
            total = max(total, counted)
        yield chunk
    if total:
        ledger.record_tokens(endpoint, total)
    _persist(ledger)


# --------------------------------------------------------------------------- #
# LangChain adapter
# --------------------------------------------------------------------------- #

#: Where the parsed schema instance rides back from the provider. On the message
#: rather than in generation_info because LangChain hands callers the AIMessage
#: and drops the ChatResult around it.
_PARSED_KEY = "llm_router_parsed"


def _take_parsed(message: AIMessage) -> Any:
    """Pull the parsed object off the message, or explain why it is missing."""
    if not isinstance(message, AIMessage) or _PARSED_KEY not in message.additional_kwargs:
        raise ValueError(
            "structured output was requested but the provider returned none; "
            "this usually means the call did not go through LLMRouter"
        )
    return message.additional_kwargs.pop(_PARSED_KEY)


def _structured_only(message: AIMessage) -> Any:
    return _take_parsed(message)


def _structured_with_raw(message: AIMessage) -> dict[str, Any]:
    """The include_raw shape LangChain callers expect.

    `parsing_error` is always None here: a schema violation is raised inside the
    provider so the ladder can step past it, which means anything reaching this
    point already parsed.
    """
    return {"raw": message, "parsed": _take_parsed(message), "parsing_error": None}


class LLMRouter(BaseChatModel):
    """A LangChain chat model that routes each call through route().

        router = LLMRouter(strategy="free_first")
        agent = create_react_agent(model=router, tools=my_tools)

    Every LangChain and LangGraph entry point - invoke, ainvoke, stream, batch,
    bind_tools, with_structured_output - funnels into the same route() as the
    plain function, so behaviour cannot drift between the two.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)

    strategy: str = "free_first"
    router_model: str | None = Field(default=None, alias="model")
    provider: str | Sequence[str] | None = None
    tier: str | None = None
    session_id: str | None = None
    allow_tier_downgrade: bool = True
    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    max_wait: float = 0.0
    #: Forwarded to the provider on every call (temperature, max_tokens, ...).
    model_kwargs: dict[str, Any] = Field(default_factory=dict)

    registry: Registry | None = Field(default=None, exclude=True)
    ledger: UsageLedger | None = Field(default=None, exclude=True)
    session_state: SessionState | None = Field(default=None, exclude=True)

    #: Set on every response: how the last call was routed.
    last_route: RouteResult | None = Field(default=None, exclude=True)

    @property
    def _llm_type(self) -> str:
        return "llm_router"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy, "model": self.router_model,
            "provider": self.provider, "tier": self.tier,
        }

    def _route_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        merged = {**self.model_kwargs, **kwargs}
        if merged.get("stop") is None:
            merged.pop("stop", None)   # LangChain passes stop=None constantly
        return {
            "strategy": self.strategy,
            "model": self.router_model,
            "provider": self.provider,
            "tier": self.tier,
            "session_id": self.session_id,
            "allow_tier_downgrade": self.allow_tier_downgrade,
            "max_attempts": self.max_attempts,
            "registry": self.registry,
            "ledger": self.ledger,
            "session_state": self.session_state,
            **merged,
        }

    def _to_result(self, result: RouteResult) -> ChatResult:
        object.__setattr__(self, "last_route", result)
        if result.parsed is not None:
            # Ride back on the message so the parser downstream of the bound
            # runnable can reach it. LangChain hands callers the AIMessage and
            # discards the ChatResult, so generation_info would not survive.
            result.message.additional_kwargs[_PARSED_KEY] = result.parsed
        generation = ChatGeneration(
            message=result.message,
            generation_info={"llm_router": result.as_metadata()},
        )
        return ChatResult(
            generations=[generation],
            llm_output={
                "model_name": result.endpoint.model_id,
                "provider": result.endpoint.provider,
                "token_usage": {"total_tokens": result.tokens or 0},
            },
        )

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        result = route(
            messages, return_route=True, max_wait=self.max_wait,
            **self._route_kwargs({"stop": stop, **kwargs}),
        )
        return self._to_result(result)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        result = await aroute(
            messages, return_route=True, max_wait=self.max_wait,
            **self._route_kwargs({"stop": stop, **kwargs}),
        )
        return self._to_result(result)

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        route_kwargs = self._route_kwargs({"stop": stop, **kwargs})
        for chunk in route_stream(messages, **route_kwargs):
            generation = ChatGenerationChunk(message=chunk)
            if run_manager:
                run_manager.on_llm_new_token(chunk.text or "", chunk=generation)
            yield generation

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        # Sync streaming in a thread would block the loop; going through
        # _agenerate keeps the event loop free and still yields a valid stream,
        # just in one chunk.
        result = await self._agenerate(messages, stop, run_manager, **kwargs)
        message = result.generations[0].message
        chunk = AIMessageChunk(
            content=message.content,
            response_metadata=message.response_metadata,
            usage_metadata=getattr(message, "usage_metadata", None),
            tool_calls=getattr(message, "tool_calls", []) or [],
        )
        generation = ChatGenerationChunk(message=chunk)
        if run_manager:
            await run_manager.on_llm_new_token(chunk.text or "", chunk=generation)
        yield generation

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> Any:
        """Bind tools without committing to a wire format.

        The tools are carried through untouched and converted by whichever
        provider ends up serving the call, so a tool definition that routes to
        Groq today and Gemini tomorrow still works.
        """
        return self.bind(tools=list(tools), **kwargs)

    def with_structured_output(
        self, schema: Any, *, include_raw: bool = False, **kwargs: Any
    ) -> Any:
        """Constrain output to `schema`, per endpoint rather than per ladder.

        This overrides BaseChatModel's inherited implementation, which would
        route every provider through function calling. That is wrong here for
        two measured reasons: Groq's gpt-oss models answer in prose under
        function calling and Groq rejects it with a 400, and Gemma on Ollama has
        too little tool calling to rely on. Both need `json_schema` instead, and
        LangChain implements that on the concrete chat model classes rather than
        on BaseChatModel - so the choice has to be made once the ladder knows
        which endpoint it landed on, not up front.

        The schema therefore travels as a bound per-call kwarg and is applied by
        the serving provider, using the `structured_output_method` its
        limits.yaml block declares. A response that violates the schema raises,
        which the ladder treats as a failed candidate and steps past - so a
        provider that cannot hold the schema falls through to one that can
        instead of returning something unusable.

        Passing `method=` explicitly is refused rather than ignored: honouring
        it would reintroduce exactly the one-method-for-every-provider bug.
        """
        if "method" in kwargs:
            raise ValueError(
                "LLMRouter chooses the structured output method per endpoint, "
                "from `structured_output_method` in limits.yaml; passing "
                "method= here would apply one provider's answer to all of them"
            )
        bound = self.bind(**{STRUCTURED_OUTPUT_KWARG: schema}, **kwargs)
        if include_raw:
            return bound | RunnableLambda(_structured_with_raw, name="llm_router_parsed")
        return bound | RunnableLambda(_structured_only, name="llm_router_parsed")

    def snapshot(self) -> dict[str, Any]:
        """Current quota position, for logging or a health endpoint."""
        return (self.ledger or default_ledger()).snapshot()
