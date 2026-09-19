"""The per-run trace.

What matters is that the record is honest about what the run did, and that
collecting it can never break the run it observes.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from procurement_agent.trace import (
    MAX_EVENTS,
    TRACED_NODES,
    RunTrace,
    TraceCallbackHandler,
    build_callbacks,
    search_hook,
)


def _llm_result(provider="groq", model="openai/gpt-oss-20b", attempts=None, tokens=42):
    """An LLMResult shaped the way LLMRoute returns one."""
    message = AIMessage(content="hi")
    message.response_metadata["llm_router"] = {
        "provider": provider,
        "model_id": model,
        "tier": "B",
        "ladder_step": 1,
        "tier_downgraded": False,
        "tokens": tokens,
        "attempts": attempts
        or [{"endpoint": f"{provider}:{model}", "outcome": "success"}],
    }
    return LLMResult(generations=[[ChatGeneration(message=message)]])


@pytest.fixture
def trace():
    return RunTrace(thread_id="t-1")


@pytest.fixture
def handler(trace):
    return TraceCallbackHandler(trace)


class TestNodeSpans:
    async def test_records_node_order(self, handler, trace):
        for name in ("intake_parser", "clarify_spec"):
            await handler.on_chain_start({"name": name}, {}, run_id=name, name=name)
            await handler.on_chain_end({}, run_id=name)

        assert trace.totals()["nodes_run"] == ["intake_parser", "clarify_spec"]

    async def test_ignores_framework_plumbing(self, handler, trace):
        """LangChain raises a chain event for every Runnable it composes.

        A real run produced five of these per node - the structured-output
        plumbing - which buried the nodes entirely. Hence an allowlist of the
        graph's own nodes rather than a blocklist that has to chase them.
        """
        for name in (
            "LangGraph", "ChannelWrite", "llm_router_parsed", "__start__",
            "RunnableParallel<raw>", "PydanticOutputParser",
            "RunnableAssign<parsed,parsing_error>", "RunnableWithFallbacks",
        ):
            await handler.on_chain_start({"name": name}, {}, run_id=name, name=name)
            await handler.on_chain_end({}, run_id=name)

        assert trace.totals()["nodes_run"] == []

    async def test_records_a_node_failure(self, handler, trace):
        await handler.on_chain_start({"name": "vendor_search"}, {}, run_id=1,
                                     name="vendor_search")
        await handler.on_chain_error(RuntimeError("boom"), run_id=1)

        errors = [e for e in trace.events if e["kind"] == "node_error"]
        assert errors and errors[0]["node"] == "vendor_search"


class TestLLMCalls:
    async def test_records_which_model_served_the_call(self, handler, trace):
        await handler.on_chat_model_start({}, [], run_id=1)
        await handler.on_llm_end(_llm_result(), run_id=1)

        call = next(e for e in trace.events if e["kind"] == "llm")
        assert call["provider"] == "groq"
        assert call["model"] == "openai/gpt-oss-20b"
        assert call["tokens"] == 42

    async def test_records_what_the_ladder_tried_first(self, handler, trace):
        """The valuable half. A call served by the third provider looks
        identical to one served by the first unless the failures are kept."""
        attempts = [
            {"endpoint": "groq:openai/gpt-oss-20b", "outcome": "rate_limited",
             "detail": "429"},
            {"endpoint": "ollama:gemma4:e4b", "outcome": "success"},
        ]
        await handler.on_chat_model_start({}, [], run_id=1)
        await handler.on_llm_end(_llm_result(attempts=attempts), run_id=1)

        call = next(e for e in trace.events if e["kind"] == "llm")
        # Only the failures: the success is the call itself.
        assert [a["outcome"] for a in call["attempts"]] == ["rate_limited"]

    async def test_a_call_that_failed_over_counts_as_a_fallback(self, handler, trace):
        attempts = [
            {"endpoint": "groq:openai/gpt-oss-20b", "outcome": "rate_limited"},
            {"endpoint": "mistral:mistral-small-2603", "outcome": "success"},
        ]
        await handler.on_chat_model_start({}, [], run_id=1)
        await handler.on_llm_end(_llm_result(attempts=attempts), run_id=1)

        assert trace.totals()["llm_fallbacks"] == 1

    async def test_a_high_ladder_step_alone_is_not_a_fallback(self, handler, trace):
        """`ladder_step` says which rung served the call, and its numbering
        depends on what was pinned.

        Every request this app makes pins a tier rather than a model, so there
        is no step-1 or step-2 rung and all of them report step 3. Reading
        `step > 1` as a fallback marked every single call as one, and reported
        "half of all calls are falling back" when nothing had failed at all.
        """
        result = _llm_result(attempts=[])
        result.generations[0][0].message.response_metadata["llm_router"]["ladder_step"] = 3
        await handler.on_chat_model_start({}, [], run_id=1)
        await handler.on_llm_end(result, run_id=1)

        assert trace.totals()["llm_fallbacks"] == 0

    async def test_a_call_is_counted_once_not_twice(self, handler, trace):
        """The inner provider call is the same call, not a second one.

        `callbacks` is in LLMRoute's `_ALWAYS_PER_CALL`, so the run config's
        handlers are forwarded down to the concrete chat model, which reports
        the call again with no routing metadata. Counting both doubled
        `llm_calls` and `llm_tokens` in every archived run and filed the
        duplicates under "unknown" - which read as a provider attribution gap
        rather than the double count it was.
        """
        bare = LLMResult(generations=[[ChatGeneration(message=AIMessage(content="x"))]])

        await handler.on_chat_model_start({}, [], run_id=1)
        await handler.on_llm_end(_llm_result(tokens=42), run_id=1)   # the router
        await handler.on_chat_model_start({}, [], run_id=2)
        await handler.on_llm_end(bare, run_id=2)                     # the inner model

        totals = trace.totals()
        assert totals["llm_calls"] == 1
        assert totals["llm_tokens"] == 42
        assert totals["llm_calls_by_provider"] == {"groq": 1}
        assert "unknown" not in totals["llm_calls_by_provider"]

    async def test_the_inner_call_is_reported_rather_than_dropped(self, handler, trace):
        """Kept so the two figures can be reconciled: a large gap between them
        means something is calling a model outside LLMRoute."""
        bare = LLMResult(generations=[[ChatGeneration(message=AIMessage(content="x"))]])
        await handler.on_chat_model_start({}, [], run_id=1)
        await handler.on_llm_end(bare, run_id=1)

        assert trace.totals()["llm_inner_calls"] == 1

    async def test_a_reply_with_no_routing_metadata_never_raises(self, handler, trace):
        """A model called outside LLMRoute still must not break tracing."""
        bare = LLMResult(generations=[[ChatGeneration(message=AIMessage(content="x"))]])
        await handler.on_llm_end(bare, run_id=99)   # no matching start, either


class TestSearchEvents:
    def test_records_each_provider_attempt(self, trace):
        """SearchRoute fires per provider, so a query that fell through two
        before a third answered shows all three - which the call site cannot
        see, because by then the fallback has already happened."""
        record = search_hook(trace)
        record("failure", {"provider": "tavily", "error": "429", "kind": "rate_limit"})
        record("success", {"provider": "exa", "latency_ms": 812.4})

        attempts = [e for e in trace.events if e["kind"] == "search_attempt"]
        assert [a["provider"] for a in attempts] == ["tavily", "exa"]
        assert attempts[0]["outcome"] == "failure"

    def test_a_broken_hook_never_breaks_the_search(self, trace):
        """These fire inside SearchRoute's own execution path."""
        record = search_hook(trace)
        record("success", {"provider": object()})  # not str-able into a field cleanly
        # No exception is the assertion.


class TestSafety:
    async def test_a_broken_handler_never_breaks_the_run(self, handler):
        """Every callback fires inside the graph's execution path. A tracer that
        raises there is worse than no tracer at all."""
        await handler.on_chain_start(None, None, run_id=1, name="x")
        await handler.on_chain_end(None, run_id=99)          # unknown run_id
        await handler.on_llm_end("not an LLMResult", run_id=1)  # type: ignore[arg-type]
        await handler.on_llm_error(RuntimeError("x"), run_id=1)

    def test_events_are_capped(self, trace):
        """A pathological run must not produce an archive nobody can open."""
        for _ in range(MAX_EVENTS + 50):
            trace.add("llm", provider="groq")

        assert len(trace.events) == MAX_EVENTS
        assert trace.totals()["events_dropped"] == 50


class TestLangfuse:
    def test_absent_keys_mean_local_tracing_only(self, trace, monkeypatch):
        """Langfuse is optional; the local record does not depend on it."""
        from procurement_agent.config import get_settings

        settings = get_settings()
        monkeypatch.setattr(settings, "langfuse_public_key", "")
        monkeypatch.setattr(settings, "langfuse_secret_key", "")

        handlers = build_callbacks(trace, "t-1", settings)
        assert len(handlers) == 1
        assert isinstance(handlers[0], TraceCallbackHandler)

    def test_keys_without_the_package_degrade_rather_than_fail(
        self, trace, monkeypatch
    ):
        """Setting keys but forgetting the extra must not stop the agent."""
        from procurement_agent.config import get_settings

        settings = get_settings()
        monkeypatch.setattr(settings, "langfuse_public_key", "pk")
        monkeypatch.setattr(settings, "langfuse_secret_key", "sk")
        monkeypatch.setitem(__import__("sys").modules, "langfuse.langchain", None)

        handlers = build_callbacks(trace, "t-1", settings)
        assert any(isinstance(h, TraceCallbackHandler) for h in handlers)


def test_the_allowlist_matches_the_graph():
    """A node renamed in the graph would silently stop being recorded."""
    from procurement_agent.graph.build import build_graph

    assert set(build_graph().nodes) <= TRACED_NODES
