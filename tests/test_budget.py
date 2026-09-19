"""The per-run search budget.

This exists alongside SearchRoute's own quota ledger and is not the same thing.
The ledger is a monthly, per-provider balance shared by everything on the
machine; this is a ceiling on one run, so a single wide fan-out cannot consume
the month in one request. The tests below are about that ceiling holding under
concurrency, which is the only way it can fail in practice.
"""

from __future__ import annotations

import asyncio

import pytest

from procurement_agent.search.client import BudgetExhausted, SessionBudget, search
from tests.conftest import FakeHit


class _Response:
    """Enough of a searchroute SearchResponse for the client to charge against."""

    def __init__(self, results, cost=1):
        self.results = results
        self.degraded = False
        self.notes = []
        self.providers_used = ["fake"]

        class _Cost:
            total = cost

        self.cost = _Cost()

        class _Depth:
            name = "SNIPPETS"

        self.depth = self.requested_depth = _Depth()


class _FakeClient:
    def __init__(self, cost=1, error=None):
        self.cost = cost
        self.error = error
        self.calls = 0

    async def search(self, query, **kwargs):
        self.calls += 1
        if self.error is not None:
            raise self.error
        # A real provider suspends here. Without an await the whole "concurrent"
        # test would run to completion one coroutine at a time and pass against
        # a budget that races - which is exactly what it is meant to catch.
        await asyncio.sleep(0)
        return _Response([FakeHit(url=f"https://example.test/{self.calls}")], self.cost)


@pytest.fixture
def fake_client(monkeypatch):
    client = _FakeClient()

    async def _get_client(settings=None):
        return client

    monkeypatch.setattr("procurement_agent.search.client.get_client", _get_client)
    return client


class TestSessionBudget:
    async def test_spending_reduces_the_remaining_credits(self, fake_client):
        budget = SessionBudget(credits=5)
        await search("anything", budget)
        assert budget.credits_remaining == 4

    async def test_one_metered_search_costs_one_whatever_the_provider_bills(
        self, fake_client
    ):
        """The budget counts searches, not provider credits.

        Provider units are not comparable: the same content search bills 2 on
        Tavily, 100 on Exa and 0 on arXiv. Measured on a real run, three Exa
        queries reported 300 and exhausted a 25-credit budget before the vendor
        search had issued half its queries. The actual money is SearchRoute's
        monthly ledger's job; this ceiling bounds how much work a run does.
        """
        fake_client.cost = 100  # what Exa reports for one content search
        budget = SessionBudget(credits=10)

        await search("anything", budget)

        assert budget.credits_remaining == 9

    async def test_an_expensive_provider_does_not_exhaust_the_run(self, fake_client):
        """The regression. Before this, three Exa searches ended the run."""
        fake_client.cost = 100
        budget = SessionBudget(credits=6)

        for _ in range(6):
            await search("q", budget)

        assert budget.credits_remaining == 0
        assert budget.truncated is False

    async def test_refuses_to_spend_past_zero(self, fake_client):
        budget = SessionBudget(credits=1)
        await search("first", budget)
        with pytest.raises(BudgetExhausted):
            await search("second", budget)

    async def test_exhaustion_marks_the_run_truncated(self, fake_client):
        """The caller needs to be able to say the list is incomplete rather
        than presenting a budget-limited result as a complete one."""
        budget = SessionBudget(credits=0)
        with pytest.raises(BudgetExhausted):
            await search("anything", budget)
        assert budget.truncated is True

    async def test_the_ceiling_holds_under_concurrent_branches(self, fake_client):
        """The reason the budget lives here rather than in the nodes.

        Nodes fan out concurrently, so a check-then-spend inside each branch
        would race and overspend. Twenty simultaneous searches against five
        credits must produce five successes, not twenty.

        The fake provider awaits mid-call so the coroutines genuinely interleave.
        A check-then-act budget passes this test without that await and fails
        with it, which is the whole reason it is there.
        """
        budget = SessionBudget(credits=5)

        async def attempt():
            try:
                await search("q", budget)
                return True
            except BudgetExhausted:
                return False

        outcomes = await asyncio.gather(*(attempt() for _ in range(20)))

        assert sum(outcomes) == 5
        assert budget.credits_remaining == 0
        assert budget.truncated is True

    async def test_a_free_provider_costs_nothing(self, fake_client):
        """The bug that made every run truncate early.

        SearchRoute's keyless providers - arXiv, PubMed, Crossref, Wikipedia,
        DuckDuckGo - report a cost of 0 because they are genuinely free.
        Charging a flat credit for them meant the whole academic pass and both
        reference lookups consumed budget they never actually spent.
        """
        fake_client.cost = 0
        budget = SessionBudget(credits=5)

        await search("arxiv query", budget)

        assert budget.credits_remaining == 5

    async def test_many_free_searches_never_exhaust_the_budget(self, fake_client):
        fake_client.cost = 0
        budget = SessionBudget(credits=2)

        for _ in range(20):
            await search("free", budget)

        assert budget.credits_remaining == 2
        assert budget.truncated is False

    async def test_a_provider_failure_returns_nothing_rather_than_raising(
        self, monkeypatch, fake_client
    ):
        """SearchRoute has already tried the alternatives by the time it fails,
        so there is nothing left to fall back to and a failed enrichment query
        must not sink the run."""
        from searchroute import SearchRouteError

        fake_client.error = SearchRouteError("all providers down")
        budget = SessionBudget(credits=5)

        assert await search("anything", budget) == []
        # Still charged: the providers were called either way.
        assert budget.credits_remaining == 4


class TestSubAllocations:
    """One pool split between nodes, not several pools that ignore each other.

    Separate budgets per node are separate ceilings, and the run-wide ceiling
    silently stops existing — which is exactly what happened before this: five
    private pools let a run spend roughly 32 credits while reporting 12.
    """

    async def test_an_allocation_is_a_limit_not_a_transfer(self):
        """Taking the credits up front looks equivalent and is not.

        An allocation that goes unused would keep them, and the vendor search
        the run exists to perform would find the pool drained by nodes that
        never spent anything. Measured: four allocations totalling 12 removed 12
        of 25 credits from a run that made five charged searches.
        """
        run = SessionBudget(credits=25)
        child = await run.sub(4, "research_sourcing")

        assert child.credits_remaining == 4
        assert run.credits_remaining == 25

    async def test_an_allocation_cannot_outspend_its_share(self, fake_client):
        run = SessionBudget(credits=25)
        child = await run.sub(2, "material_research")

        await search("one", child)
        await search("two", child)
        with pytest.raises(BudgetExhausted):
            await search("three", child)

        # Only what was actually spent left the run's pool.
        assert run.credits_remaining == 23

    async def test_the_same_name_shares_one_allocation(self):
        """What makes a fan-out share a pool.

        `contact_extraction` runs once per candidate. Without memoising by name,
        twelve branches would each be handed a full allocation — reproducing the
        private-pool bug inside a single node.
        """
        run = SessionBudget(credits=25)
        first = await run.sub(4, "contact_extraction")
        second = await run.sub(4, "contact_extraction")

        assert first is second
        assert run.credits_remaining == 25

    async def test_a_fan_out_cannot_exceed_its_shared_allocation(self, fake_client):
        """Twelve concurrent candidates, four credits between them."""
        run = SessionBudget(credits=25)

        async def branch():
            budget = await run.sub(4, "contact_extraction")
            try:
                await search("confirm", budget)
                return True
            except BudgetExhausted:
                return False

        outcomes = await asyncio.gather(*(branch() for _ in range(12)))

        assert sum(outcomes) == 4
        assert run.credits_remaining == 21   # only the four that ran

    async def test_an_unused_allocation_costs_the_run_nothing(self, fake_client):
        """The academic pass on a material with no literature must not shrink
        the vendor search's budget. There is nothing to release: credits are
        only ever taken when actually spent."""
        run = SessionBudget(credits=25)
        await run.sub(4, "research_sourcing")
        await run.sub(4, "contact_extraction")
        await run.sub(3, "material_research")

        assert run.credits_remaining == 25

    async def test_allocations_still_share_one_run_ceiling(self, fake_client):
        """Limits may add up to more than the run has; the run still wins."""
        run = SessionBudget(credits=3)
        a = await run.sub(5, "material_research")
        b = await run.sub(5, "research_sourcing")

        await search("1", a)
        await search("2", a)
        await search("3", b)
        with pytest.raises(BudgetExhausted):
            await search("4", b)

        assert run.credits_remaining == 0

    async def test_an_allocation_is_capped_at_what_is_left(self):
        """A node that can do part of its work beats one that does none."""
        run = SessionBudget(credits=2)
        child = await run.sub(10, "material_research")

        # Capped by what the run actually has, not by the nominal limit.
        assert child.credits_remaining == 2
        assert run.credits_remaining == 2

    async def test_exhausting_an_allocation_marks_the_run_truncated(self):
        """The buyer needs to know the list is incomplete however it happened."""
        run = SessionBudget(credits=1)
        child = await run.sub(0, "clarify_spec")

        with pytest.raises(BudgetExhausted):
            await child._reserve()

        assert run.truncated is True
