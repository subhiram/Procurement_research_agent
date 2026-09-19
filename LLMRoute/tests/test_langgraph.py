"""The point of the whole exercise: LLMRouter must drop into LangGraph unchanged.

These run against fake providers, so they prove the plumbing - tool binding,
tool-call round trips, streaming, fallback inside an agent turn - without
spending a real free tier.
"""

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from langchain_core.tools import tool

from llm_router import LLMRouter, RateLimited
from llm_router.providers import ProviderResponse, get_provider

pytestmark = pytest.mark.usefixtures("fake_providers")

pytest.importorskip("langgraph")
from langgraph.prebuilt import create_react_agent  # noqa: E402


@tool
def get_weather(city: str) -> str:
    """Look up the weather in a city."""
    return f"It is 18C and clear in {city}."


@pytest.fixture
def router(registry, ledger, sessions):
    return LLMRouter(
        strategy="free_first", model="qwen-27b",
        registry=registry, ledger=ledger, session_state=sessions,
    )


def with_tool_call(fake, endpoint_key, *, name="get_weather", args=None):
    """Make one endpoint answer with a tool call the first time it is asked."""
    state = {"called": False}
    original = get_provider(endpoint_key.split(":")[0]).invoke

    def invoke(endpoint, messages, **kwargs):
        if endpoint.key == endpoint_key and not state["called"]:
            state["called"] = True
            fake.calls.append(endpoint.key)
            message = AIMessage(
                content="",
                tool_calls=[{
                    "name": name, "args": args or {"city": "Lisbon"}, "id": "call_1",
                }],
                usage_metadata={"input_tokens": 8, "output_tokens": 4, "total_tokens": 12},
            )
            return ProviderResponse(message=message, endpoint=endpoint, tokens=12)
        return original(endpoint, messages, **kwargs)

    get_provider(endpoint_key.split(":")[0]).invoke = invoke
    return state


# --------------------------------------------------------------------------- #

def test_router_is_accepted_as_a_langgraph_model(router):
    agent = create_react_agent(model=router, tools=[get_weather])
    result = agent.invoke({"messages": [("user", "hello")]})
    assert isinstance(result["messages"][-1], AIMessage)
    assert result["messages"][-1].response_metadata["llm_router"]["provider"] == "groq"


def test_a_full_tool_call_round_trip(router, fake_providers):
    """The agent must be able to call a tool and come back for a final answer."""
    with_tool_call(fake_providers, "groq:qwen/qwen3.8-27b")
    agent = create_react_agent(model=router, tools=[get_weather])
    result = agent.invoke({"messages": [("user", "weather in Lisbon?")]})

    kinds = [type(m).__name__ for m in result["messages"]]
    assert "ToolMessage" in kinds
    tool_message = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert "18C" in tool_message.content
    assert isinstance(result["messages"][-1], AIMessage)


def test_tools_reach_whichever_provider_serves_the_call(router, fake_providers):
    agent = create_react_agent(model=router, tools=[get_weather])
    agent.invoke({"messages": [("user", "hi")]})
    assert get_provider("groq").last_kwargs["tools"] == [get_weather]


def test_fallback_happens_inside_an_agent_turn(registry, ledger, sessions, fake_providers):
    """A 429 mid-agent must be invisible to the graph."""
    # Groq hosts qwen-27b under two model ids; fail both so the ladder must
    # cross providers to a same-tier peer, rather than just switching model id.
    fake_providers.fail["groq:qwen/qwen3.8-27b"] = RateLimited("groq", "qwen", retry_after=30.0)
    fake_providers.fail["groq:qwen/qwen3.6-27b"] = RateLimited("groq", "qwen", retry_after=30.0)
    router = LLMRouter(model="qwen-27b", registry=registry, ledger=ledger,
                       session_state=sessions)
    agent = create_react_agent(model=router, tools=[get_weather])
    result = agent.invoke({"messages": [("user", "hi")]})
    routing = result["messages"][-1].response_metadata["llm_router"]
    assert routing["provider"] == "mistral"
    assert routing["tier_downgraded"] is False


def test_sticky_holds_one_endpoint_across_agent_steps(registry, ledger, sessions, fake_providers):
    """A research run should not change model between steps of the same graph."""
    with_tool_call(fake_providers, "groq:qwen/qwen3.8-27b")
    router = LLMRouter(strategy="sticky", session_id="research-run-42", model="qwen-27b",
                       registry=registry, ledger=ledger, session_state=sessions)
    agent = create_react_agent(model=router, tools=[get_weather])
    result = agent.invoke({"messages": [("user", "weather in Lisbon?")]})

    providers = {
        m.response_metadata["llm_router"]["provider"]
        for m in result["messages"]
        if isinstance(m, AIMessage) and "llm_router" in m.response_metadata
    }
    assert providers == {"groq"}


def test_agent_streaming(router):
    agent = create_react_agent(model=router, tools=[get_weather])
    chunks = [
        chunk for chunk, _ in agent.stream(
            {"messages": [("user", "hi")]}, stream_mode="messages"
        )
        if isinstance(chunk, AIMessageChunk)
    ]
    assert "".join(str(c.content) for c in chunks).strip()


def test_async_agent(router):
    import asyncio

    agent = create_react_agent(model=router, tools=[get_weather])
    result = asyncio.run(agent.ainvoke({"messages": [("user", "hi")]}))
    assert isinstance(result["messages"][-1], AIMessage)


def test_with_structured_output(router, fake_providers):
    """The schema reaches the serving endpoint and comes back as an instance."""
    from pydantic import BaseModel

    class Answer(BaseModel):
        """A short answer."""

        text: str

    fake_providers.structured = {"text": "hello"}
    structured = router.with_structured_output(Answer)
    assert structured.invoke("say hello") == Answer(text="hello")


def test_structured_output_uses_the_endpoints_own_method(
    registry, ledger, sessions, fake_providers
):
    """The method is a property of the endpoint, not of the provider.

    gpt-oss answers in prose under function calling and Groq 400s it, so its
    decoding is constrained by schema instead - while other Groq models use tool
    calling. One setting for the provider, let alone for the whole ladder, is
    wrong for at least one of them.
    """
    from pydantic import BaseModel

    class Answer(BaseModel):
        text: str

    router = LLMRouter(
        strategy="free_first", model="gpt-oss-20b",
        registry=registry, ledger=ledger, session_state=sessions,
    )
    fake_providers.structured = {"text": "hello"}
    router.with_structured_output(Answer).invoke("say hello")

    endpoint_key, method = fake_providers.structured_methods[-1]
    assert endpoint_key == "groq:openai/gpt-oss-20b"
    assert method == "json_schema"


def test_a_structured_call_routes_away_from_a_model_that_cannot_serve_it(
    registry, ledger, sessions, fake_providers
):
    """allam-2-7b supports neither json_schema nor tool calling.

    Asking for it with a schema must land somewhere that can actually answer,
    rather than spending the attempt to collect a 400.
    """
    from pydantic import BaseModel

    class Answer(BaseModel):
        text: str

    router = LLMRouter(
        strategy="free_first", model="allam-2-7b",
        registry=registry, ledger=ledger, session_state=sessions,
    )
    fake_providers.structured = {"text": "hello"}

    assert router.with_structured_output(Answer).invoke("hi") == Answer(text="hello")
    assert "groq:allam-2-7b" not in fake_providers.calls


def test_with_structured_output_refuses_an_explicit_method(router):
    """Accepting method= would reapply one provider's answer to every provider."""
    from pydantic import BaseModel

    class Answer(BaseModel):
        text: str

    with pytest.raises(ValueError, match="per endpoint"):
        router.with_structured_output(Answer, method="function_calling")


def test_structured_output_include_raw_keeps_the_message(router, fake_providers):
    from pydantic import BaseModel

    class Answer(BaseModel):
        text: str

    fake_providers.structured = {"text": "hello"}
    result = router.with_structured_output(Answer, include_raw=True).invoke("hi")

    assert result["parsed"] == Answer(text="hello")
    assert isinstance(result["raw"], AIMessage)
    assert result["parsing_error"] is None
    # The routing metadata still rides on the raw message.
    assert result["raw"].response_metadata["llm_router"]["provider"] == "groq"


def test_a_schema_violation_falls_through_to_the_next_candidate(
    router, fake_providers
):
    """A provider that cannot hold the schema must not end the call.

    This is the behaviour that makes the ladder worth having for structured
    output: Groq returning prose instead of a conforming object should cost one
    candidate, not the whole request.
    """
    from pydantic import BaseModel

    class Answer(BaseModel):
        text: str

    fake_providers.structured = {"text": "hello"}
    fake_providers.fail["groq:qwen/qwen3.8-27b"] = lambda: ValueError("not json")

    result = router.with_structured_output(Answer).invoke("say hello")
    assert result == Answer(text="hello")
    assert fake_providers.calls[0] == "groq:qwen/qwen3.8-27b"
    assert len(fake_providers.calls) > 1


def test_works_with_langchain_v1_create_agent(router):
    """LangGraph v1 moved the prebuilt agent to langchain.agents.create_agent.

    Skipped when the umbrella `langchain` package is not installed; the router
    is a plain BaseChatModel either way, so both entry points accept it.
    """
    agents = pytest.importorskip("langchain.agents")

    agent = agents.create_agent(model=router, tools=[get_weather])
    result = agent.invoke({"messages": [("user", "hello")]})
    assert isinstance(result["messages"][-1], AIMessage)
