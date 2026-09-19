"""A scriptable stand-in for a real provider, so router behaviour is testable
without spending anyone's free tier."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator

from langchain_core.messages import AIMessage, AIMessageChunk

from llm_router.providers import (
    STRUCTURED_OUTPUT_KWARG,
    BaseProvider,
    ProviderResponse,
    register_provider,
    reset_providers,
)
from llm_router.registry import Endpoint

#: Every provider models.yaml can actually route to. `ollama` is included: it is
#: keyless, so unlike the hosted providers it is reachable in a test environment
#: with no credentials set at all.
ROUTED_PROVIDERS = (
    "groq", "mistral", "google_ai_studio", "nvidia_nim", "openrouter", "ollama",
)


@dataclass
class Script:
    """What the fake should do, keyed by endpoint key."""

    #: endpoint key -> exception to raise (callable returning one, or an instance)
    fail: dict[str, Any] = field(default_factory=dict)
    #: endpoint key -> how many more times to keep failing (None = forever)
    fail_times: dict[str, int] = field(default_factory=dict)
    #: endpoint key -> rate-limit header snapshot to report on success
    headers: dict[str, dict[str, int]] = field(default_factory=dict)
    tokens: int | None = 42
    calls: list[str] = field(default_factory=list)
    #: Field values used to build the schema instance when a call asks for
    #: structured output. Keyed by nothing: one shape serves every endpoint,
    #: since these tests care about which endpoint answered, not what it said.
    structured: dict[str, Any] = field(default_factory=dict)
    #: (endpoint key, method) for every structured call, so a test can assert
    #: the per-provider method actually reached the wrapper.
    structured_methods: list[tuple[str, str]] = field(default_factory=list)

    def clear(self) -> None:
        self.fail.clear()
        self.fail_times.clear()
        self.headers.clear()
        self.calls.clear()
        self.structured.clear()
        self.structured_methods.clear()


class FakeProvider(BaseProvider):
    script = Script()

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    def _next_error(self, endpoint: Endpoint) -> BaseException | None:
        key = endpoint.key
        error = self.script.fail.get(key)
        if error is None:
            return None
        remaining = self.script.fail_times.get(key)
        if remaining is not None:
            if remaining <= 0:
                return None
            self.script.fail_times[key] = remaining - 1
        return error() if callable(error) else error

    def _respond(self, endpoint: Endpoint, schema: Any = None) -> ProviderResponse:
        self.script.calls.append(endpoint.key)
        error = self._next_error(endpoint)
        if error is not None:
            raise error
        message = AIMessage(
            content=f"hello from {endpoint.provider}/{endpoint.model_id}",
            usage_metadata=(
                {"input_tokens": 10, "output_tokens": self.script.tokens - 10,
                 "total_tokens": self.script.tokens}
                if self.script.tokens else None
            ),
        )
        parsed = None
        if schema is not None:
            # Record the method the endpoint declared, so a test can prove the
            # per-provider choice reached the wrapper rather than one method
            # being applied to the whole ladder.
            self.script.structured_methods.append(
                (endpoint.key, endpoint.structured_output_method)
            )
            parsed = schema(**self.script.structured)
        return ProviderResponse(
            message=message, endpoint=endpoint, tokens=self.script.tokens,
            rate_limit=self.script.headers.get(endpoint.key, {}),
            parsed=parsed,
        )

    def invoke(self, endpoint, messages, **kwargs) -> ProviderResponse:
        self.last_kwargs = kwargs
        return self._respond(endpoint, kwargs.get(STRUCTURED_OUTPUT_KWARG))

    async def ainvoke(self, endpoint, messages, **kwargs) -> ProviderResponse:
        self.last_kwargs = kwargs
        return self._respond(endpoint, kwargs.get(STRUCTURED_OUTPUT_KWARG))

    def stream(self, endpoint, messages, **kwargs) -> Iterator[AIMessageChunk]:
        self.last_kwargs = kwargs
        response = self._respond(endpoint)   # raises before any chunk escapes

        def generate():
            for word in str(response.message.content).split():
                yield AIMessageChunk(content=word + " ")
            yield AIMessageChunk(
                content="", usage_metadata={
                    "input_tokens": 10, "output_tokens": 32, "total_tokens": 42
                },
            )

        return generate()


def install_fakes() -> Script:
    """Swap every routable provider for the fake. Returns the shared script."""
    FakeProvider.script = Script()
    for name in ROUTED_PROVIDERS:
        register_provider(name, lambda n=name: FakeProvider(n))
    return FakeProvider.script


def uninstall_fakes() -> None:
    """Put the real provider classes back.

    Driven off ROUTED_PROVIDERS rather than a second hardcoded list, because the
    two silently drifted apart once: adding a provider to ROUTED_PROVIDERS but
    not here left its fake installed for every later test in the session.
    """
    from llm_router.providers import _PROVIDER_CLASSES

    for name in ROUTED_PROVIDERS:
        cls = _REAL_CLASSES.get(name)
        if cls is not None:
            _PROVIDER_CLASSES[name] = cls
    reset_providers()


def _real_classes() -> dict[str, type]:
    """Snapshot the genuine classes before any fake is registered."""
    from llm_router.providers import _PROVIDER_CLASSES

    return {name: _PROVIDER_CLASSES[name] for name in ROUTED_PROVIDERS}


#: Captured at import, which is before any fixture installs a fake.
_REAL_CLASSES: dict[str, type] = _real_classes()
