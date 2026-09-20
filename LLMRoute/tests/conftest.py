import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_router.registry import load_registry  # noqa: E402


class FakeClock:
    """Controllable time source, so window tests do not sleep."""

    def __init__(self, start: float = 1_000_000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> float:
        self.now += seconds
        return self.now


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture(scope="session")
def registry():
    return load_registry()


@pytest.fixture(autouse=True)
def ollama_reachable(request, monkeypatch):
    """Assume the local Ollama daemon is running, as it is on a dev machine.

    Every test that relies on Ollama's keyless backstop being in the ladder
    depends on this. The registry now also gates Ollama on a live TCP probe of
    OLLAMA_BASE_URL, so without this fixture the whole suite would depend on a
    real daemon running wherever it executes - including CI, which has none.
    Tests that care about the unreachable case (or the probe itself) opt out
    with @pytest.mark.real_ollama_probe.
    """
    if request.node.get_closest_marker("real_ollama_probe"):
        return

    from llm_router import registry

    monkeypatch.setattr(registry, "_ollama_daemon_reachable", lambda: True)


@pytest.fixture
def all_keys(monkeypatch):
    """Pretend every routable provider has credentials.

    Ollama is absent because it needs none: its limits.yaml block declares no
    `api_key_env`, so it is already routable without this fixture.
    """
    for name in (
        "GROQ_API_KEY", "MISTRAL_API_KEY", "GOOGLE_API_KEY",
        "NVIDIA_API_KEY", "OPENROUTER_API_KEY",
    ):
        monkeypatch.setenv(name, "test-key")
    return True


@pytest.fixture
def fake_providers(all_keys):
    """Replace every real provider with a scriptable fake."""
    from fake_provider import install_fakes, uninstall_fakes

    script = install_fakes()
    try:
        yield script
    finally:
        script.clear()
        uninstall_fakes()


@pytest.fixture
def ledger(clock):
    from llm_router.ledger import UsageLedger

    return UsageLedger(time_fn=clock)


@pytest.fixture
def sessions():
    from llm_router.policies import SessionState

    return SessionState()
