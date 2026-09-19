"""Selection policies: free_first ordering and sticky session behaviour."""

import pytest

from llm_router.ladder import resolve_candidates
from llm_router.ledger import UsageLedger
from llm_router.policies import (
    SessionState,
    free_first,
    get_policy,
    signals_downgrade,
    sticky,
)

pytestmark = pytest.mark.usefixtures("all_keys")


def keys(candidates):
    return [c.endpoint.key for c in candidates]


@pytest.fixture
def candidates(registry):
    """gemma-31b: three hosts at step 1, so index 1 is a real second choice."""
    return resolve_candidates("gemma-31b", registry=registry)


@pytest.fixture
def state():
    return SessionState()


# --------------------------------------------------------------------------- #
# free_first
# --------------------------------------------------------------------------- #

def test_free_first_keeps_ladder_order(candidates):
    assert keys(free_first(candidates)) == keys(candidates)


def test_free_first_ignores_session_pins(candidates, state):
    state.remember("s1", candidates[1])
    assert keys(free_first(candidates, "s1", state)) == keys(candidates)


# --------------------------------------------------------------------------- #
# sticky
# --------------------------------------------------------------------------- #

def test_sticky_without_a_pin_is_just_the_ladder(candidates, state):
    assert keys(sticky(candidates, "fresh", state)) == keys(candidates)


def test_sticky_returns_to_the_same_endpoint(candidates, state):
    second = candidates[1]
    state.remember("research-run-42", second)
    ordered = sticky(candidates, "research-run-42", state)
    assert ordered[0].endpoint.key == second.endpoint.key
    # nothing is dropped, just reordered
    assert set(keys(ordered)) == set(keys(candidates))
    assert len(ordered) == len(candidates)


def test_sticky_sessions_do_not_leak_into_each_other(candidates, state):
    state.remember("a", candidates[1])
    assert sticky(candidates, "a", state)[0].endpoint.key == candidates[1].endpoint.key
    assert sticky(candidates, "b", state)[0].endpoint.key == candidates[0].endpoint.key


def test_sticky_falls_back_when_the_pinned_endpoint_is_exhausted(registry, clock):
    ledger = UsageLedger(time_fn=clock)
    state = SessionState()
    all_candidates = resolve_candidates("gemma-31b", registry=registry)
    pinned = all_candidates[1]                        # google_ai_studio
    state.remember("run", pinned)

    ledger.record_rate_limited(pinned.endpoint, retry_after=60.0)
    live = resolve_candidates("gemma-31b", ledger=ledger, registry=registry)
    ordered = sticky(live, "run", state)

    assert pinned.endpoint.key not in keys(ordered)
    assert ordered[0].endpoint.key == all_candidates[0].endpoint.key


def test_sticky_resumes_the_pin_once_quota_returns(registry, clock):
    ledger = UsageLedger(time_fn=clock)
    state = SessionState()
    pinned = resolve_candidates("gemma-31b", registry=registry)[1]
    state.remember("run", pinned)
    ledger.record_rate_limited(pinned.endpoint, retry_after=60.0)

    clock.advance(61)
    live = resolve_candidates("gemma-31b", ledger=ledger, registry=registry)
    assert sticky(live, "run", state)[0].endpoint.key == pinned.endpoint.key


# --------------------------------------------------------------------------- #
# downgrades must not become permanent
# --------------------------------------------------------------------------- #

def test_a_downgrade_is_never_pinned(registry, state):
    """Otherwise a session would sit at tier B long after tier S recovered."""
    candidates = resolve_candidates("gpt-oss-120b", registry=registry)
    downgrade = next(c for c in candidates if c.tier_downgraded)
    state.remember("run", downgrade)
    assert state.get("run") is None


def test_same_tier_swap_is_pinned(registry, state):
    candidates = resolve_candidates("qwen-27b", registry=registry)
    swap = next(c for c in candidates if c.step > 1 and not c.tier_downgraded)
    state.remember("run", swap)
    assert state.get("run").endpoint_key == swap.endpoint.key


def test_sticky_is_the_policy_that_signals_downgrades():
    assert signals_downgrade("sticky")
    assert not signals_downgrade("free_first")


# --------------------------------------------------------------------------- #
# session bookkeeping
# --------------------------------------------------------------------------- #

def test_forget_and_clear(candidates, state):
    state.remember("a", candidates[0])
    state.forget("a")
    assert state.get("a") is None
    state.remember("b", candidates[0])
    state.clear()
    assert len(state) == 0


def test_session_map_is_bounded(candidates):
    state = SessionState(max_sessions=3)
    for i in range(10):
        state.remember(f"s{i}", candidates[0])
    assert len(state) == 3
    assert state.get("s0") is None and state.get("s9") is not None


def test_no_session_id_is_a_no_op(candidates, state):
    state.remember(None, candidates[0])
    assert len(state) == 0
    assert state.get(None) is None


# --------------------------------------------------------------------------- #
# resolution
# --------------------------------------------------------------------------- #

def test_get_policy_by_name():
    assert get_policy("sticky")[1] is sticky
    assert get_policy("free_first")[1] is free_first


def test_get_policy_accepts_a_callable(candidates):
    def reverse_policy(candidates, session_id=None, session_state=None):
        return list(reversed(candidates))

    name, policy = get_policy(reverse_policy)
    assert name == "reverse_policy"
    assert keys(policy(candidates)) == list(reversed(keys(candidates)))


def test_unknown_strategy_is_rejected():
    with pytest.raises(ValueError, match="unknown strategy"):
        get_policy("cheapest_maybe")
