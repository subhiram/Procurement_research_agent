"""The model factory — the single seam between this app and its LLM provider.

Nodes call `get_model(task, schema=...)` and never import a provider class, so
the whole routing layer can be replaced without touching a node. That is exactly
what happened: retry, fallback, rate limiting and provider configuration used to
live in this package and now live in LLMRoute, which does the same jobs across
more providers and keeps a quota ledger that survives a restart.

What this module still owns is the *domain* question LLMRoute deliberately does
not answer: which quality tier each of this app's tasks deserves. LLMRoute knows
about tiers, ladders and quotas; it does not know that material research is the
one genuinely hard judgment call in the graph and that contact extraction is a
token hog that should land on a local model. That mapping is below.
"""

from __future__ import annotations

import logging
from typing import TypeVar

from langchain_core.runnables import Runnable
from llm_router import LLMRouter, configure, default_registry
from pydantic import BaseModel

from procurement_agent.config import Settings, TaskType, get_settings

log = logging.getLogger(__name__)

SchemaT = TypeVar("SchemaT", bound=BaseModel)

#: Quality floor per task, not a provider list.
#:
#: Ordering used to be an explicit per-task provider chain here. It is a tier
#: now because LLMRoute picks within a tier by live quota rather than by a fixed
#: order, which is strictly better on free tiers - a chain hardcodes an ordering
#: that is only correct while every provider still has budget.
#:
#: The tiers themselves are measured, from a bake-off on this project's real
#: schemas:
#:
#:   provider          intake parse                  contact extraction   speed
#:   groq/gpt-oss-20b  full (dims + qty + unit)      correct              0.8s
#:   gemma4:e4b        partial (name only)           correct              19-25s
#:   llama3.1 / 3.2    partial (name only)           ALL NULLS            2-8s
#:
#: `bulk` is the one that looks surprising. It is the heaviest node by volume -
#: one call per vendor - so it sits in tier B, where a local Ollama endpoint is
#: the last candidate and absorbs the overflow for free once the hosted free
#: tiers are spent. It is not pinned to Ollama: Groq is ~25x faster and the
#: regex pre-filter already cut the payload to roughly 360 tokens, which fits
#: inside Groq's budget comfortably.
TASK_TIERS: dict[TaskType, str] = {
    # Material research: low token volume, highest judgment. The one call worth
    # spending the scarcest quota on.
    "reasoning": "S",
    # Outreach emails are prose a human will read and edit.
    "drafting": "A",
    # A short structured parse of the buyer's request. 20b matched 120b here.
    "extraction": "B",
    # The buyer is waiting on this one, so latency matters more than depth.
    "clarification": "B",
    # One call per vendor, on a regex digest rather than the page.
    "bulk": "B",
}

#: Wait this long for a rate-limited endpoint before stepping down the ladder.
#:
#: LLMRoute defaults to 0.0 - fail over instantly. That is wrong here: a brief
#: Groq 429 is worth riding out, because stepping down means either a slower
#: local model or a provider with a tighter daily cap. Ten seconds is long
#: enough to clear a per-minute burst and short enough not to stall a run.
MAX_WAIT_SECONDS = 10.0

_configured = False


def _ensure_configured(settings: Settings) -> None:
    """Point LLMRoute's quota ledger at a file, once per process.

    Without this the ledger is per-process and in-memory, so a fresh CLI run
    starts believing the whole daily quota is unspent and rediscovers the truth
    through 429s. Daily and weekly caps are the ones that make this matter.
    """
    global _configured
    if _configured:
        return
    path = settings.llm_ledger_path
    path.parent.mkdir(parents=True, exist_ok=True)
    configure(ledger_path=str(path))
    _configured = True


def get_model(
    task: TaskType,
    schema: type[BaseModel] | None = None,
    *,
    session_id: str | None = None,
    settings: Settings | None = None,
) -> Runnable:
    """A model for `task`, with fallback, quota accounting and retries applied.

    Returns a LangChain runnable. With `schema`, invoking it yields an instance
    of that schema; without one, an `AIMessage`.

    `session_id` identifies the run. It only changes routing when
    `llm_sticky_session` is on, in which case the run pins to whichever endpoint
    served its first call.

    That is off by default, deliberately. On free tiers, spreading across
    providers as quota allows beats concentrating load on one provider's limit -
    and this graph's tasks span tiers on purpose (`reasoning` is S, `bulk` is B),
    so a single session pin does not apply cleanly across a run in any case.
    Turn it on when consistent model behaviour within one run matters more than
    spreading the load.
    """
    settings = settings or get_settings()
    _ensure_configured(settings)

    sticky = bool(session_id and settings.llm_sticky_session)
    router = LLMRouter(
        tier=TASK_TIERS[task],
        strategy="sticky" if sticky else "free_first",
        session_id=session_id,
        max_wait=MAX_WAIT_SECONDS,
        model_kwargs={"temperature": 0.0},
    )
    if schema is None:
        return router
    # The method (function calling vs json_schema) is chosen per endpoint by
    # LLMRoute, from limits.yaml. It has to be: Groq's gpt-oss models answer in
    # prose under function calling and get 400ed, and Gemma's tool calling is
    # too weak to rely on - so a single method for the whole ladder is wrong for
    # at least one provider in it.
    return router.with_structured_output(schema)


def describe_routing(settings: Settings | None = None) -> list[str]:
    """One line per task saying where it would actually go right now."""
    settings = settings or get_settings()
    _ensure_configured(settings)
    registry = default_registry()

    lines = []
    for task, tier in TASK_TIERS.items():
        endpoints = registry.enabled_endpoints(tier=tier)
        served = ", ".join(f"{e.provider}/{e.model_id}" for e in endpoints) or "nothing"
        lines.append(f"{task} (tier {tier}): {served}")
    return lines


async def validate_models(settings: Settings | None = None) -> list[str]:
    """Warn about tasks that currently have nowhere to run.

    Deliberately a configuration check, not a network one. Verifying that model
    IDs are still served is LLMRoute's job and belongs with its config -
    `python LLMRoute/scripts/verify_models.py` does it against every provider's
    live catalogue. What matters here is the thing specific to this app: whether
    each task's tier has any usable endpoint at all.

    Warns and returns the problems; never raises, so a misconfiguration cannot
    take the app down at boot.
    """
    settings = settings or get_settings()
    _ensure_configured(settings)
    registry = default_registry()
    problems: list[str] = []

    usable = [name for name in registry.providers() if registry.is_usable(name)]
    if not usable:
        msg = (
            "no LLM provider is usable: every hosted provider is missing its API "
            "key and the local Ollama daemon is disabled. Set one key in .env, "
            "or run Ollama locally."
        )
        log.warning(msg)
        return [msg]
    log.info("llm providers usable: %s", ", ".join(sorted(usable)))

    for task, tier in TASK_TIERS.items():
        if not registry.enabled_endpoints(tier=tier):
            msg = (
                f"task {task!r} needs tier {tier} and no tier-{tier} endpoint is "
                f"usable; it will fail at runtime. Usable providers: "
                f"{', '.join(sorted(usable))}"
            )
            log.warning(msg)
            problems.append(msg)

    return problems
