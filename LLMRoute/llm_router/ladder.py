"""resolve_candidates() - the fallback ladder.

Given "I want this model" (or "I want at least this tier"), produce the ordered
list of endpoints worth trying, cheapest deviation first:

    step 1  same model, another provider
    step 2  another model in the same tier, on a provider that hosts the
            requested model
    step 3  another model in the same tier, anywhere
    step 4  a lower tier, anywhere            <- flagged as a downgrade

Step 4 is the only one that changes the answer's quality, so candidates
reached that way carry `tier_downgraded=True`. Steps 2 and 3 swap the model but
hold the tier, and must never be reported as a downgrade - and equally, a
tier-A answer must never be passed off as the tier-S one that was asked for.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Sequence

from .ledger import Availability, UsageLedger
from .registry import TIERS, Endpoint, Registry, default_registry, tier_rank

logger = logging.getLogger("llm_router.ladder")

STEP_SAME_MODEL = 1
STEP_SAME_TIER_SAME_PROVIDER = 2
STEP_SAME_TIER_ANY_PROVIDER = 3
STEP_TIER_DOWNGRADE = 4

STEP_REASONS = {
    STEP_SAME_MODEL: "requested model",
    STEP_SAME_TIER_SAME_PROVIDER: "same tier, provider that hosts the requested model",
    STEP_SAME_TIER_ANY_PROVIDER: "same tier, different provider",
    STEP_TIER_DOWNGRADE: "lower tier",
}


@dataclass(frozen=True)
class Candidate:
    """One endpoint worth trying, and how far off the request it is."""

    endpoint: Endpoint
    step: int
    #: True only when the ladder had to accept a worse tier than was asked for
    tier_downgraded: bool = False
    requested_tier: str | None = None
    availability: Availability | None = None

    @property
    def available(self) -> bool:
        return self.availability is None or self.availability.ok

    @property
    def reason(self) -> str:
        if self.requested_tier is None:
            # Nothing was asked for, so the step numbers do not describe a
            # deviation from anything - it is just best-available order.
            return f"no preference given; best available (tier {self.endpoint.tier})"
        reason = STEP_REASONS.get(self.step, "candidate")
        if self.tier_downgraded:
            reason += f" ({self.requested_tier} -> {self.endpoint.tier})"
        return reason

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"{self.endpoint.provider}/{self.endpoint.model_id} [step {self.step}: {self.reason}]"


def _sort_key(endpoint: Endpoint) -> tuple[int, str, str]:
    # Deterministic order: configured priority, then a stable tiebreak. Ordering
    # by remaining quota was tempting but makes routing non-reproducible between
    # runs, which is miserable to debug.
    return (endpoint.priority, endpoint.provider, endpoint.model_id)


def resolve_candidates(
    logical_model: str | None = None,
    requested_tier: str | None = None,
    ledger: UsageLedger | None = None,
    *,
    registry: Registry | None = None,
    providers: Sequence[str] | None = None,
    streaming: bool = False,
    structured: bool = False,
    estimated_tokens: int = 0,
    allow_tier_downgrade: bool = True,
    include_unavailable: bool = False,
) -> list[Candidate]:
    """Build the fallback ladder for one request.

    By default only endpoints the ledger says are callable right now come back,
    in ladder order, so the caller can simply walk the list. Pass
    `include_unavailable=True` to get the blocked ones too, each carrying the
    reason and its retry_after - that is what the router uses to explain itself
    when everything is exhausted.

    `providers` pins the search to a subset (forcing `provider="groq"`), and
    `streaming` drops endpoints whose wrapper cannot stream, and `structured`
    drops those that cannot produce schema-conforming output at all.
    """
    registry = registry or default_registry()

    if logical_model is not None and not registry.has_model(logical_model):
        raise KeyError(
            f"unknown model {logical_model!r}; "
            f"known models: {', '.join(registry.models())}"
        )
    if requested_tier is not None:
        requested_tier = requested_tier.upper()
        tier_rank(requested_tier)  # validates

    # The tier to hold: what was asked for explicitly, else the requested
    # model's own tier. With neither, nothing has been promised, so no
    # candidate can count as a downgrade.
    target_tier = requested_tier or (
        registry.tier_of(logical_model) if logical_model else None
    )

    common = dict(providers=providers, streaming=streaming, structured=structured)
    seen: set[str] = set()
    candidates: list[Candidate] = []

    def take(endpoints: Iterable[Endpoint], step: int, tier: str | None) -> None:
        downgraded = bool(
            target_tier and tier and tier_rank(tier) > tier_rank(target_tier)
        )
        for endpoint in sorted(endpoints, key=_sort_key):
            if endpoint.key in seen:
                continue
            seen.add(endpoint.key)
            candidates.append(
                Candidate(
                    endpoint=endpoint,
                    step=step,
                    tier_downgraded=downgraded,
                    requested_tier=target_tier,
                )
            )

    # -- step 1: the model that was actually asked for ---------------------- #
    hosting_providers: list[str] = []
    if logical_model is not None:
        same_model = registry.enabled_endpoints(logical_model=logical_model, **common)
        # Preserve host order for step 2, before dedup mutates anything.
        hosting_providers = list(dict.fromkeys(e.provider for e in same_model))
        take(same_model, STEP_SAME_MODEL, registry.tier_of(logical_model))

    # -- steps 2 and 3: hold the tier, change the model --------------------- #
    if target_tier is not None:
        same_tier = registry.enabled_endpoints(tier=target_tier, **common)
        if hosting_providers:
            take(
                [e for e in same_tier if e.provider in hosting_providers],
                STEP_SAME_TIER_SAME_PROVIDER,
                target_tier,
            )
        take(same_tier, STEP_SAME_TIER_ANY_PROVIDER, target_tier)
    else:
        # Nothing specific was requested, so walk the tiers best-first. These
        # are not downgrades: no quality level was ever promised.
        for tier in TIERS:
            take(registry.enabled_endpoints(tier=tier, **common), STEP_SAME_TIER_ANY_PROVIDER, tier)

    # -- step 4: give up quality, last ------------------------------------- #
    if allow_tier_downgrade and target_tier is not None:
        for tier in TIERS[tier_rank(target_tier) + 1:]:
            take(registry.enabled_endpoints(tier=tier, **common), STEP_TIER_DOWNGRADE, tier)

    if ledger is None:
        return candidates

    # Attach live availability, and drop what cannot be called right now.
    resolved = [
        Candidate(
            endpoint=candidate.endpoint,
            step=candidate.step,
            tier_downgraded=candidate.tier_downgraded,
            requested_tier=candidate.requested_tier,
            availability=ledger.availability(
                candidate.endpoint, estimated_tokens=estimated_tokens
            ),
        )
        for candidate in candidates
    ]
    if include_unavailable:
        return resolved
    return [candidate for candidate in resolved if candidate.available]


def soonest_retry(candidates: Sequence[Candidate]) -> float | None:
    """Shortest wait across blocked candidates, for backing off sensibly."""
    waits = [
        candidate.availability.retry_after
        for candidate in candidates
        if candidate.availability is not None
        and not candidate.availability.ok
        and candidate.availability.retry_after is not None
    ]
    return min(waits) if waits else None


def explain(candidates: Sequence[Candidate]) -> str:
    """One line per candidate, for logs and exhaustion errors."""
    lines = []
    for candidate in candidates:
        status = "available"
        if candidate.availability is not None and not candidate.availability.ok:
            wait = candidate.availability.retry_after
            status = candidate.availability.reason or "unavailable"
            if wait is not None:
                status += f"; retry in {wait:.0f}s"
        lines.append(
            f"  step {candidate.step} {candidate.endpoint.provider}/"
            f"{candidate.endpoint.model_id} "
            f"[tier {candidate.endpoint.tier}"
            + (", DOWNGRADED" if candidate.tier_downgraded else "")
            + f"] - {status}"
        )
    return "\n".join(lines)
