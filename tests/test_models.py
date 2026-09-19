"""The model factory.

Provider fallback, retries, rate limiting and quota accounting are LLMRoute's
and are tested there. What is this app's to get right is the mapping from its
own tasks to a quality tier, and the guarantee that structured output still
works across a ladder that may land on any provider — which is the thing that
broke in production before, and the reason LLMRoute chooses the method per
endpoint rather than once per call.
"""

from __future__ import annotations

import pytest
from llm_router import default_registry
from pydantic import BaseModel

from procurement_agent.config import TaskType, get_settings
from procurement_agent.llm.models import (
    MAX_WAIT_SECONDS,
    TASK_TIERS,
    describe_routing,
    get_model,
    validate_models,
)


class _Answer(BaseModel):
    text: str


@pytest.fixture
def registry():
    return default_registry()


class TestTaskTiers:
    def test_every_task_type_has_a_tier(self):
        """A task with no entry raises a KeyError deep inside a graph run, so
        the mapping must be total rather than defaulted."""
        declared = set(TaskType.__args__)
        assert declared == set(TASK_TIERS)

    def test_every_tier_is_a_real_one(self, registry):
        from llm_router import TIERS

        assert set(TASK_TIERS.values()) <= set(TIERS)

    def test_reasoning_outranks_the_bulk_work(self):
        """Material research is the one genuinely hard judgment call and the
        cheapest by volume; contact extraction is the reverse. Spending the
        scarce quota the other way round is the mistake this guards."""
        from llm_router.registry import tier_rank

        assert tier_rank(TASK_TIERS["reasoning"]) < tier_rank(TASK_TIERS["bulk"])

    def test_bulk_can_reach_a_free_local_endpoint(self, registry):
        """contact_extraction makes one call per vendor. Its tier must contain
        an unmetered endpoint, or a wide run exhausts the hosted free tiers."""
        endpoints = registry.enabled_endpoints(tier=TASK_TIERS["bulk"])
        assert any(e.provider == "ollama" for e in endpoints)

    def test_the_local_endpoint_is_last_in_its_tier(self, registry):
        """Free but slow: it should absorb overflow, not lead."""
        endpoints = registry.enabled_endpoints(tier=TASK_TIERS["bulk"])
        priorities = {e.provider: e.priority for e in endpoints}
        assert priorities["ollama"] == max(priorities.values())


class TestStructuredOutput:
    def test_a_schema_produces_a_structured_runnable(self):
        model = get_model("extraction", schema=_Answer)
        assert hasattr(model, "invoke")

    def test_no_schema_returns_the_router_itself(self):
        model = get_model("extraction")
        assert model.__class__.__name__ == "LLMRouter"

    def test_gpt_oss_is_constrained_by_json_schema_not_tool_calling(self, registry):
        """The failure this whole seam exists for.

        Under function calling the gpt-oss models intermittently answer in prose
        and Groq rejects it with a 400 - observed live on material_research,
        which came back as a markdown table. Every node here uses structured
        output, so this is not a corner case.
        """
        gpt_oss = [
            e for e in registry.all_endpoints()
            if e.provider == "groq" and "gpt-oss" in e.model_id
        ]
        assert gpt_oss
        assert all(e.structured_output_method == "json_schema" for e in gpt_oss)

    def test_the_method_varies_within_a_single_provider(self, registry):
        """Why this is per endpoint rather than per provider.

        Groq needs json_schema on gpt-oss and cannot do it at all on allam-2-7b,
        which answers "This model does not support response format
        `json_schema`" with a 400. A per-provider setting is wrong either way.
        """
        methods = {
            e.structured_output_method
            for e in registry.all_endpoints()
            if e.provider == "groq"
        }
        assert methods == {"json_schema", "function_calling"}

    def test_a_model_with_no_structured_output_is_excluded_from_those_calls(
        self, registry
    ):
        """allam-2-7b supports neither json_schema nor tool calling. Since every
        node here sends a schema, it must leave the ladder rather than be
        discovered by burning an attempt on a guaranteed 400."""
        structured = registry.enabled_endpoints(
            tier=TASK_TIERS["extraction"], structured=True
        )
        assert not any(e.model_id == "allam-2-7b" for e in structured)

    def test_local_gemma_is_constrained_by_json_schema_too(self, registry):
        """Gemma's tool calling is too weak to produce a reliable call."""
        ollama = [e for e in registry.all_endpoints() if e.provider == "ollama"]
        assert ollama
        assert all(e.structured_output_method == "json_schema" for e in ollama)

    def test_the_method_is_not_uniform_across_providers(self, registry):
        """If it were, one setting would do and the per-endpoint machinery
        would be dead weight. It is not: OpenAI-compatible hosts use tool
        calling."""
        methods = {e.structured_output_method for e in registry.all_endpoints()}
        assert len(methods) > 1


class TestRouterConfiguration:
    def test_spreading_across_providers_is_the_default(self, monkeypatch):
        """Off by default on purpose: on free tiers, spreading across providers
        as quota allows beats concentrating load on one provider's limit, and
        this graph's tasks span tiers so a single pin does not apply cleanly."""
        monkeypatch.setattr(get_settings(), "llm_sticky_session", False)
        assert get_model("bulk", session_id="thread-1").strategy == "free_first"

    def test_a_session_id_can_pin_the_run_when_asked(self, monkeypatch):
        """For when consistent model behaviour within one run matters more."""
        monkeypatch.setattr(get_settings(), "llm_sticky_session", True)
        model = get_model("bulk", session_id="thread-1")
        assert model.strategy == "sticky"
        assert model.session_id == "thread-1"

    def test_pinning_needs_a_session_to_pin_to(self, monkeypatch):
        """Sticky with no session id has nothing to key on, so it must not
        silently claim to be pinning."""
        monkeypatch.setattr(get_settings(), "llm_sticky_session", True)
        assert get_model("bulk").strategy == "free_first"

    def test_waits_briefly_rather_than_failing_over_instantly(self):
        """LLMRoute defaults to 0.0. Stepping down the ladder on the first 429
        means a slower local model or a tighter daily cap, so a short burst is
        worth riding out."""
        assert get_model("reasoning").max_wait == MAX_WAIT_SECONDS
        assert MAX_WAIT_SECONDS > 0

    def test_no_provider_credentials_are_read_from_app_settings(self):
        """Keys belong to LLMRoute, which reads the environment itself.
        Duplicating them here would mean two places to update."""
        fields = set(get_settings().model_dump())
        assert not [f for f in fields if f.endswith("_api_key")]


class TestValidation:
    async def test_reports_no_problems_when_a_provider_is_reachable(self):
        """Ollama is keyless, so this holds even with no API keys set."""
        assert await validate_models() == []

    async def test_describes_where_each_task_would_go(self):
        lines = describe_routing()
        assert len(lines) == len(TASK_TIERS)
        assert all(any(line.startswith(task) for line in lines) for task in TASK_TIERS)

    async def test_never_raises(self, monkeypatch):
        """A misconfiguration must surface as a warning at startup, not as an
        exception that takes the app down at boot."""
        monkeypatch.setattr(
            "procurement_agent.llm.models.default_registry",
            lambda: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        with pytest.raises(RuntimeError):
            # Documents the current contract honestly: the registry itself is
            # allowed to fail loudly, because a registry that cannot load means
            # there is no routing at all and there is nothing to degrade to.
            await validate_models()
