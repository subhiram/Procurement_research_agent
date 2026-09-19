"""What one run actually did, recorded as it happens.

The archive stores a run's inputs and outputs. This records the path between
them: which node ran when, which model served it, what the routing ladder tried
before something answered, which search provider replied and what it cost. That
is the information needed to tune tiers and budgets, and almost all of it was
already being produced and thrown away.

Nothing here is collected by hand. Two framework-native seams supply it:

- **LangChain's callback protocol**, which LangGraph already routes every node
  span and every model call through. Attaching one handler to the run config
  therefore captures node causality and LLM calls without a single node knowing
  that tracing exists.
- **SearchRoute's `hooks` parameter**, a `(event, payload)` callable it invokes
  per provider call. Used rather than wrapping `search()`, because it also sees
  the retries and fallbacks that happen *inside* SearchRoute - which is exactly
  where a search gets expensive.

LLMRoute supplies the interesting half of the LLM record for free: every reply
carries `response_metadata["llm_router"]` with the provider, model, tier, ladder
step, tokens, latency and the full list of attempts. The attempts matter most -
they show what was tried and why it fell through, which is invisible from the
outside.

Langfuse, when configured, is simply a second handler in the same list. The
local trace does not depend on it: this file is written to `runs/` regardless,
so the observability a run needs never requires an account or a container.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.outputs import LLMResult

from procurement_agent.config import Settings, get_settings

log = logging.getLogger(__name__)

#: The graph's own nodes — the only chain spans worth recording.
#:
#: An allowlist rather than a blocklist of framework internals, because the
#: blocklist lost. LangChain raises a chain event for every Runnable it
#: composes, so one real node produced five spans named `RunnableParallel<raw>`,
#: `PydanticOutputParser`, `RunnableAssign<parsed,parsing_error>` and so on -
#: the structured-output plumbing, which changes shape with the binding method
#: and would need the blocklist updated every time. Naming what we want is
#: stable; enumerating what we don't never will be.
TRACED_NODES = frozenset(
    {
        "intake_parser",
        "clarify_spec",
        "ask_clarification",
        "material_research",
        "research_sourcing",
        "vendor_search",
        "fan_out_to_extraction",
        "contact_extraction",
        "vendor_summary",
        "save_run",
        "draft_outreach_email",
    }
)

#: Cap on recorded events. A pathological run should not turn its archive file
#: into something nobody can open.
MAX_EVENTS = 2000


def _elapsed_ms(began: float | None) -> int | None:
    return int((time.monotonic() - began) * 1000) if began else None


def _swallowed(where: str, exc: BaseException) -> None:
    """Record that tracing failed, without letting it reach the caller.

    Logged rather than silently dropped: a bug in here is invisible otherwise -
    the trace simply comes back missing events, with nothing to say why. Debug
    level, because a broken tracer is not the operator's problem mid-run.
    """
    log.debug("trace: %s failed (%s); event dropped", where, exc)


@dataclass
class RunTrace:
    """Ordered record of one run. Append-only; never raises at a call site."""

    thread_id: str
    started_at: float = field(default_factory=time.monotonic)
    events: list[dict] = field(default_factory=list)
    _dropped: int = 0

    def add(self, kind: str, **fields: Any) -> None:
        if len(self.events) >= MAX_EVENTS:
            self._dropped += 1
            return
        self.events.append(
            {"kind": kind, "at": round(time.monotonic() - self.started_at, 3), **fields}
        )

    # -- summarising -------------------------------------------------------- #

    def totals(self) -> dict:
        """The headline numbers, so a reader does not have to sum the events."""
        llm = [e for e in self.events if e["kind"] == "llm"]
        searches = [e for e in self.events if e["kind"] == "search"]

        by_provider: dict[str, int] = {}
        for event in llm:
            by_provider[event.get("provider") or "unknown"] = (
                by_provider.get(event.get("provider") or "unknown", 0) + 1
            )
        search_by_provider: dict[str, int] = {}
        for event in searches:
            for name in event.get("providers") or ["none"]:
                search_by_provider[name] = search_by_provider.get(name, 0) + 1

        return {
            "wall_seconds": round(time.monotonic() - self.started_at, 2),
            "llm_calls": len(llm),
            "llm_calls_by_provider": by_provider,
            # The same calls as seen by the concrete chat model underneath the
            # router. Reported rather than dropped so the numbers can be
            # reconciled: this should track `llm_calls`, and a large gap means
            # something is calling a model outside LLMRoute.
            "llm_inner_calls": sum(1 for e in self.events if e["kind"] == "llm_inner"),
            "llm_tokens": sum(e.get("tokens") or 0 for e in llm),
            # A fallback is a call that something was tried for and failed
            # before it succeeded - so it is counted from the recorded failed
            # attempts, not from `ladder_step`.
            #
            # `ladder_step` looks like the obvious signal and is not: it says
            # *which rung* served the call, and its numbering depends on what
            # was pinned. A request pinned to a tier rather than a model has no
            # step-1 or step-2 rung at all, so every call reports step 3 and
            # `step > 1` marks all of them as fallbacks. Every request this app
            # makes is tier-pinned, so that read said "half of all calls are
            # falling back" when the true figure was none.
            "llm_fallbacks": sum(1 for e in llm if e.get("attempts")),
            "search_calls": len(searches),
            "search_calls_by_provider": search_by_provider,
            # Against the run ceiling: one per metered search, zero for keyless.
            "searches_charged": sum(e.get("charged") or 0 for e in searches),
            # The providers' own units. Not comparable between providers - a
            # content search is 2 on Tavily and 100 on Exa - so this is for
            # spotting an expensive provider, not for budgeting.
            "provider_cost_units": sum(e.get("provider_cost") or 0 for e in searches),
            "nodes_run": [e["node"] for e in self.events if e["kind"] == "node_end"],
            "events_dropped": self._dropped,
        }

    def as_dict(self) -> dict:
        return {"totals": self.totals(), "events": list(self.events)}


class TraceCallbackHandler(AsyncCallbackHandler):
    """Feeds a RunTrace from LangChain's own callbacks.

    Every method swallows its own errors. A tracer that breaks the run it was
    only meant to observe is worse than no tracer, and these fire inside the
    graph's execution path.
    """

    def __init__(self, trace: RunTrace) -> None:
        self.trace = trace
        self._llm_started: dict[UUID, float] = {}
        self._node_started: dict[UUID, tuple[str, float]] = {}

    # -- node spans --------------------------------------------------------- #

    async def on_chain_start(
        self, serialized: dict[str, Any], inputs: Any, *, run_id: UUID, **kwargs: Any
    ) -> None:
        try:
            name = (kwargs.get("name") or (serialized or {}).get("name") or "").strip()
            if name not in TRACED_NODES:
                return
            self._node_started[run_id] = (name, time.monotonic())
            self.trace.add("node_start", node=name)
        except Exception as exc:  # noqa: BLE001 - never break the run
            _swallowed("node span", exc)

    async def on_chain_end(self, outputs: Any, *, run_id: UUID, **kwargs: Any) -> None:
        try:
            started = self._node_started.pop(run_id, None)
            if started is None:
                return
            name, began = started
            self.trace.add(
                "node_end", node=name, ms=int((time.monotonic() - began) * 1000)
            )
        except Exception as exc:  # noqa: BLE001
            _swallowed("callback", exc)

    async def on_chain_error(
        self, error: BaseException, *, run_id: UUID, **kwargs: Any
    ) -> None:
        try:
            started = self._node_started.pop(run_id, None)
            if started is None:
                return
            self.trace.add("node_error", node=started[0], detail=str(error)[:300])
        except Exception as exc:  # noqa: BLE001
            _swallowed("callback", exc)

    # -- model calls -------------------------------------------------------- #

    async def on_chat_model_start(
        self, serialized: dict[str, Any], messages: Any, *, run_id: UUID, **kwargs: Any
    ) -> None:
        self._llm_started[run_id] = time.monotonic()

    async def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        try:
            began = self._llm_started.pop(run_id, None)
            routing = self._routing_of(response)
            if not routing:
                # Every model call in this app goes through LLMRoute, so a reply
                # with no routing metadata is the *inner* provider call, not a
                # separate one: `callbacks` is in LLMRoute's `_ALWAYS_PER_CALL`,
                # so the run config's handlers are forwarded down to the
                # concrete chat model, which then reports the same call again.
                #
                # Counting both doubled `llm_calls` and `llm_tokens` in every
                # archived run, and put the duplicates under "unknown" - which
                # looked like a provider attribution gap rather than the
                # double count it was. Those totals are what tier and budget
                # decisions get made from, so they have to be the real figure.
                self.trace.add("llm_inner", ms=_elapsed_ms(began))
                return
            self.trace.add(
                "llm",
                provider=routing.get("provider"),
                model=routing.get("model_id"),
                tier=routing.get("tier"),
                # >1 means the first-choice endpoint did not serve this call.
                ladder_step=routing.get("ladder_step"),
                tier_downgraded=routing.get("tier_downgraded"),
                tokens=routing.get("tokens"),
                ms=_elapsed_ms(began),
                # What was tried before this succeeded, and why each failed.
                attempts=[
                    {
                        "endpoint": a.get("endpoint"),
                        "outcome": a.get("outcome"),
                        "detail": a.get("detail"),
                    }
                    for a in (routing.get("attempts") or [])
                    if a.get("outcome") != "success"
                ],
            )
        except Exception as exc:  # noqa: BLE001
            _swallowed("callback", exc)

    async def on_llm_error(
        self, error: BaseException, *, run_id: UUID, **kwargs: Any
    ) -> None:
        try:
            self._llm_started.pop(run_id, None)
            self.trace.add("llm_error", detail=str(error)[:300])
        except Exception as exc:  # noqa: BLE001
            _swallowed("callback", exc)

    @staticmethod
    def _routing_of(response: LLMResult) -> dict:
        """LLMRoute's own account of how the call was routed."""
        for generations in response.generations or []:
            for generation in generations:
                info = getattr(generation, "generation_info", None) or {}
                if "llm_router" in info:
                    return info["llm_router"]
                message = getattr(generation, "message", None)
                metadata = getattr(message, "response_metadata", None) or {}
                if "llm_router" in metadata:
                    return metadata["llm_router"]
        return {}


def search_hook(trace: RunTrace):
    """A SearchRoute hook recording each provider *attempt* into `trace`.

    SearchRoute fires this per provider, so a query that fell through two
    providers before a third answered shows all three - which the call site
    cannot see, because by then the fallback has already happened.

    It carries only `provider`, `latency_ms`, `kind` and `error`; the query,
    capability and credit cost are recorded separately by `search()` itself,
    which is where they are known. The two are complementary: one `search`
    event per logical query, and a `search_attempt` event per provider tried.
    """

    def record(event: str, payload: dict) -> None:
        try:
            trace.add(
                "search_attempt",
                outcome=event,
                provider=payload.get("provider"),
                ms=round(payload["latency_ms"], 1)
                if payload.get("latency_ms") is not None
                else None,
                # `error_kind`, not `kind`: that name is taken by the event type
                # in `RunTrace.add`, and passing it here is a TypeError that the
                # catch below would swallow into silently missing events.
                error_kind=str(payload["kind"]) if payload.get("kind") else None,
                error=str(payload["error"])[:200] if payload.get("error") else None,
            )
        except Exception as exc:  # noqa: BLE001 - never break the search
            _swallowed("search hook", exc)

    return record


def build_callbacks(
    trace: RunTrace, thread_id: str, settings: Settings | None = None
) -> list:
    """Handlers for the run config: the local trace, plus Langfuse if configured.

    Langfuse is added only when both keys are present, so it is genuinely
    optional - the agent behaves identically without it, and the local trace is
    written either way. `LANGFUSE_HOST` chooses between Langfuse Cloud and a
    self-hosted instance, which is why that decision needs no code change.
    """
    settings = settings or get_settings()
    handlers: list = [TraceCallbackHandler(trace)]

    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        return handlers

    try:
        from langfuse import get_client
        from langfuse.langchain import CallbackHandler
    except ImportError:
        log.warning(
            "langfuse keys are set but the package is not installed; "
            "run `uv sync --extra observability`. Local tracing is unaffected."
        )
        return handlers

    try:
        # The client is configured explicitly rather than left to read ambient
        # environment variables. The SDK would find them in most cases, but
        # "most cases" is the wrong guarantee here: pointing this at a
        # self-hosted instance must be a single setting that visibly takes
        # effect, not one that silently falls back to the cloud endpoint if the
        # variable did not reach the SDK's process environment.
        get_client(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
        handlers.append(CallbackHandler())
        log.info("langfuse tracing enabled -> %s", settings.langfuse_host)
    except Exception as exc:  # noqa: BLE001 - observability is never fatal
        log.warning("langfuse handler could not be created (%s); continuing", exc)
    return handlers
