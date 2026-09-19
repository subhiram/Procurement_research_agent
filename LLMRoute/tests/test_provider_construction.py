"""Checks the real provider wrappers construct, and that Groq's header capture
actually works - it relies on injecting an httpx client, which is the one piece
of machinery that cannot be confirmed by reading the code."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from llm_router.providers import _last_headers, get_provider, reset_providers
from llm_router.registry import Endpoint

CHAT_COMPLETION = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 1,
    "model": "openai/gpt-oss-120b",
    "choices": [{
        "index": 0,
        "message": {"role": "assistant", "content": "ok"},
        "finish_reason": "stop",
    }],
    "usage": {"prompt_tokens": 11, "completion_tokens": 2, "total_tokens": 13},
}

RATE_LIMIT_HEADERS = {
    "x-ratelimit-limit-requests": "30",
    "x-ratelimit-remaining-requests": "26",
    "x-ratelimit-limit-tokens": "8000",
    "x-ratelimit-remaining-tokens": "7891",
    "x-ratelimit-reset-requests": "2m59.56s",
}


def sse_chunks() -> bytes:
    """The same answer in OpenAI/Groq server-sent-event form."""
    def chunk(delta, finish=None):
        payload = {
            "id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 1,
            "model": "openai/gpt-oss-120b",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        return f"data: {json.dumps(payload)}\n\n".encode()

    return (
        chunk({"role": "assistant", "content": ""})
        + chunk({"content": "ok"})
        + chunk({}, "stop")
        + b"data: [DONE]\n\n"
    )


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        request = json.loads(self.rfile.read(length) or b"{}")

        if request.get("stream"):
            body, content_type = sse_chunks(), "text/event-stream"
        else:
            body, content_type = json.dumps(CHAT_COMPLETION).encode(), "application/json"

        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for name, value in RATE_LIMIT_HEADERS.items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def fake_groq_server():
    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()


def test_groq_captures_rate_limit_headers(fake_groq_server, monkeypatch):
    """langchain_groq drops response headers, so we inject an httpx client with
    an event hook. If that ever stops working, the ledger silently loses Groq's
    authoritative counters - hence this test."""
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    reset_providers()
    provider = get_provider("groq")
    endpoint = Endpoint(
        provider="groq", model_id="openai/gpt-oss-120b",
        logical_model="gpt-oss-120b", tier="S", api_key_env=("GROQ_API_KEY",),
    )
    _last_headers.set(None)

    response = provider.invoke(
        endpoint, [("human", "hi")], base_url=fake_groq_server
    )

    assert response.message.content == "ok"
    assert response.tokens == 13
    assert response.rate_limit == {
        "limit_requests": 30, "remaining_requests": 26,
        "limit_tokens": 8000, "remaining_tokens": 7891,
    }
    reset_providers()


def test_captured_headers_flow_into_the_ledger(fake_groq_server, monkeypatch, clock):
    """The whole point: Groq's own numbers override our local count."""
    from llm_router.ledger import UsageLedger

    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    reset_providers()
    provider = get_provider("groq")
    endpoint = Endpoint(
        provider="groq", model_id="openai/gpt-oss-120b",
        logical_model="gpt-oss-120b", tier="S", api_key_env=("GROQ_API_KEY",),
        limits=(),
    )
    ledger = UsageLedger(time_fn=clock)
    response = provider.invoke(endpoint, [("human", "hi")], base_url=fake_groq_server)
    ledger.sync_from_headers(endpoint, **response.rate_limit)

    snapshot = ledger.snapshot()[endpoint.ledger_key]["counters"]
    assert snapshot["requests/minute"] == {"used": 4, "limit": 30, "remaining": 26}
    assert snapshot["tokens/minute"]["used"] == 109
    reset_providers()


@pytest.mark.parametrize(
    "provider_name,model_id,env",
    [
        ("groq", "openai/gpt-oss-120b", "GROQ_API_KEY"),
        ("mistral", "mistral-small-latest", "MISTRAL_API_KEY"),
        ("google_ai_studio", "gemini-2.5-flash", "GOOGLE_API_KEY"),
        ("nvidia_nim", "nvidia/nemotron-3-super-120b-a12b", "NVIDIA_API_KEY"),
        ("openrouter", "z-ai/glm-5.2:free", "OPENROUTER_API_KEY"),
    ],
)
def test_every_wrapper_constructs_its_chat_model(provider_name, model_id, env, monkeypatch):
    """Catches signature drift in the provider packages without a network call."""
    monkeypatch.setenv(env, "test-key")
    reset_providers()
    endpoint = Endpoint(
        provider=provider_name, model_id=model_id, logical_model="m", tier="A",
        api_key_env=(env,),
    )
    model = get_provider(provider_name).chat_model(endpoint, temperature=0.0)
    assert model is not None
    # cached, so a second call does not rebuild it
    assert get_provider(provider_name).chat_model(endpoint, temperature=0.0) is model
    reset_providers()


def test_ollama_constructs_without_any_api_key(monkeypatch):
    """The keyless path: no api_key_env, so the auth check must not fire.

    This is what lets the ladder still serve a request on a machine with no
    credentials configured at all, so it is worth asserting separately from the
    hosted providers rather than folding into the table above.
    """
    for name in ("GROQ_API_KEY", "MISTRAL_API_KEY", "OLLAMA_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama.test:11434")
    reset_providers()
    endpoint = Endpoint(
        provider="ollama", model_id="gemma4:e4b", logical_model="gemma-e4b",
        tier="B", api_key_env=(),
    )
    model = get_provider("ollama").chat_model(endpoint)
    assert model is not None
    assert "ollama.test" in str(model.base_url)
    reset_providers()


def test_streaming_also_syncs_rate_limit_headers(fake_groq_server, monkeypatch, clock):
    """Agents stream constantly; the ledger must not go blind during it."""
    from llm_router.ledger import UsageLedger

    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    reset_providers()
    provider = get_provider("groq")
    endpoint = Endpoint(
        provider="groq", model_id="openai/gpt-oss-120b",
        logical_model="gpt-oss-120b", tier="S", api_key_env=("GROQ_API_KEY",),
    )
    ledger = UsageLedger(time_fn=clock)

    list(provider.stream(endpoint, [("human", "hi")], base_url=fake_groq_server))
    headers = provider.pop_rate_limit()
    assert headers["remaining_requests"] == 26

    ledger.sync_from_headers(endpoint, **headers)
    assert ledger.snapshot()[endpoint.ledger_key]["counters"]["requests/minute"]["used"] == 4
    reset_providers()


def test_headers_are_consumed_only_once(fake_groq_server, monkeypatch):
    """A second call must not be charged with the first call's counters."""
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    reset_providers()
    provider = get_provider("groq")
    endpoint = Endpoint(
        provider="groq", model_id="openai/gpt-oss-120b",
        logical_model="gpt-oss-120b", tier="S", api_key_env=("GROQ_API_KEY",),
    )
    provider.invoke(endpoint, [("human", "hi")], base_url=fake_groq_server)
    assert provider.pop_rate_limit() == {}
    reset_providers()


@pytest.mark.parametrize(
    "provider_name,env",
    [("openrouter", "OPENROUTER_API_KEY"), ("nvidia_nim", "NVIDIA_API_KEY")],
)
def test_openai_compatible_wrappers_call_and_capture_headers(
    provider_name, env, fake_groq_server, monkeypatch
):
    """OpenRouter and NVIDIA NIM are ChatOpenAI pointed elsewhere.

    The fake server above speaks plain OpenAI chat-completions, which is exactly
    what both of these endpoints are, so it stands in for either one. This
    checks the whole path end to end: the wrapper constructs, the call round
    trips, usage is read back, and the injected httpx client's event hook still
    catches the rate-limit headers through langchain_openai rather than
    langchain_groq.
    """
    monkeypatch.setenv(env, "test-key")
    reset_providers()
    provider = get_provider(provider_name)
    endpoint = Endpoint(
        provider=provider_name, model_id="m", logical_model="m", tier="A",
        api_key_env=(env,),
    )
    _last_headers.set(None)

    response = provider.invoke(endpoint, [("human", "hi")], base_url=fake_groq_server)

    assert response.message.content == "ok"
    assert response.tokens == 13
    assert response.rate_limit["remaining_requests"] == 26
    reset_providers()
