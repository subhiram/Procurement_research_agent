"""API surface: auth, SSE event contract, and session lifecycle.

Runs against a stubbed graph so it needs no database — what is under test is the
HTTP layer and the event contract a client depends on, not LangGraph.
"""

from __future__ import annotations

import json

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from procurement_agent.api.routes import router
from procurement_agent.graph.state import MaterialSpec, VendorLead

API_KEY = "dev-local-key"
HEADERS = {"X-API-Key": API_KEY}


class FakeInterrupt:
    def __init__(self, value):
        self.value = value


class FakeState:
    def __init__(self, values=None, interrupts=(), created_at="2026-01-01T00:00:00Z"):
        self.values = values or {}
        self.interrupts = list(interrupts)
        self.created_at = created_at


class FakeGraph:
    """Replays a scripted set of stream chunks and states."""

    def __init__(self, chunks=None, state=None, raises=None):
        self.chunks = chunks or []
        self.state = state or FakeState(created_at=None)
        self.raises = raises
        self.updates = []

    async def astream(self, payload, config, stream_mode="updates"):
        self.last_payload = payload
        if self.raises:
            raise self.raises
        for chunk in self.chunks:
            yield chunk

    async def aget_state(self, config):
        return self.state

    async def aupdate_state(self, config, values):
        self.updates.append(values)


def make_app(graph) -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    app.state.graph = graph
    return app


async def client_for(graph) -> AsyncClient:
    return AsyncClient(
        transport=ASGITransport(app=make_app(graph)), base_url="http://test"
    )


def parse_sse(text: str) -> list[tuple[str, dict]]:
    """Pull (event, data) pairs out of an SSE body, ignoring keep-alives.

    SSE uses CRLF line endings, so normalise before splitting on the blank line
    that separates events.
    """
    events = []
    for block in text.replace("\r\n", "\n").split("\n\n"):
        event = data = None
        for line in block.splitlines():
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data = line[5:].strip()
        if event and data:
            events.append((event, json.loads(data)))
    return events


class TestAuth:
    async def test_rejects_missing_key(self):
        async with await client_for(FakeGraph()) as c:
            assert (await c.get("/health")).status_code == 401

    async def test_rejects_wrong_key(self):
        async with await client_for(FakeGraph()) as c:
            r = await c.get("/health", headers={"X-API-Key": "nope"})
            assert r.status_code == 401

    async def test_accepts_correct_key(self):
        async with await client_for(FakeGraph()) as c:
            assert (await c.get("/health", headers=HEADERS)).status_code == 200


class TestSessionLifecycle:
    async def test_create_returns_a_thread_id(self):
        async with await client_for(FakeGraph()) as c:
            r = await c.post("/sessions", headers=HEADERS)
            assert r.status_code == 201
            assert r.json()["thread_id"]

    async def test_unknown_session_is_404(self):
        async with await client_for(FakeGraph(state=FakeState(created_at=None))) as c:
            r = await c.get("/sessions/missing", headers=HEADERS)
            assert r.status_code == 404

    async def test_returns_pending_questions_when_suspended(self):
        questions = [{"field": "form", "question": "Bar or tube?", "why": "different vendors"}]
        graph = FakeGraph(
            state=FakeState(
                values={"status": "clarifying", "material_spec": MaterialSpec(material_name="Custom 465")},
                interrupts=[FakeInterrupt({"questions": questions})],
            )
        )
        async with await client_for(graph) as c:
            body = (await c.get("/sessions/t1", headers=HEADERS)).json()

        assert body["awaiting_clarification"] is True
        assert body["pending_questions"] == questions
        assert body["material_spec"]["material_name"] == "Custom 465"
        assert "ITAR" in body["disclaimer"]

    async def test_resume_rejects_a_session_not_awaiting_clarification(self):
        graph = FakeGraph(state=FakeState(values={"status": "done"}))
        async with await client_for(graph) as c:
            r = await c.post("/sessions/t1/resume", headers=HEADERS, json={"answers": "x"})
            assert r.status_code == 409

    async def test_resume_on_unknown_session_is_404(self):
        graph = FakeGraph(state=FakeState(created_at=None))
        async with await client_for(graph) as c:
            r = await c.post("/sessions/t1/resume", headers=HEADERS, json={"answers": "x"})
            assert r.status_code == 404


class TestStreaming:
    async def test_emits_node_progress_then_final(self):
        lead = VendorLead(
            company_name="Acme",
            website="https://acme.example",
            source_url="https://acme.example/p",
            email="sales@acme.example",
        )
        graph = FakeGraph(
            chunks=[{"intake_parser": {"status": "clarifying"}}, {"vendor_summary": {"status": "done"}}],
            state=FakeState(values={"status": "done", "vendor_leads": [lead]}),
        )
        async with await client_for(graph) as c:
            r = await c.post("/sessions/t1/messages", headers=HEADERS, json={"content": "hi"})

        events = parse_sse(r.text)
        assert [e for e, _ in events] == ["node_end", "node_end", "final"]
        final = events[-1][1]
        assert final["vendor_leads"][0]["email"] == "sales@acme.example"
        assert "ITAR" in final["disclaimer"]

    async def test_interrupt_chunk_does_not_break_the_stream(self):
        """LangGraph reports suspension as `__interrupt__`, whose value is a
        tuple rather than a state update."""
        graph = FakeGraph(
            chunks=[
                {"intake_parser": {"status": "clarifying"}},
                {"__interrupt__": (FakeInterrupt({"questions": []}),)},
            ],
            state=FakeState(
                values={"status": "clarifying"},
                interrupts=[FakeInterrupt({"kind": "clarification", "questions": [{"q": 1}]})],
            ),
        )
        async with await client_for(graph) as c:
            r = await c.post("/sessions/t1/messages", headers=HEADERS, json={"content": "hi"})

        events = parse_sse(r.text)
        assert [e for e, _ in events] == ["node_end", "interrupt"]
        assert events[-1][1]["questions"] == [{"q": 1}]

    async def test_failure_is_reported_as_an_error_event(self):
        """The stream must always terminate in a state the client can act on."""
        graph = FakeGraph(raises=RuntimeError("provider exploded"))
        async with await client_for(graph) as c:
            r = await c.post("/sessions/t1/messages", headers=HEADERS, json={"content": "hi"})

        events = parse_sse(r.text)
        assert events[-1][0] == "error"
        assert "provider exploded" in events[-1][1]["detail"]


class TestEmailDrafting:
    async def test_rejects_a_session_with_no_leads(self):
        graph = FakeGraph(state=FakeState(values={"vendor_leads": []}))
        async with await client_for(graph) as c:
            r = await c.post("/sessions/t1/emails", headers=HEADERS)
            assert r.status_code == 409


class TestArchiveEndpoints:
    """The archive over HTTP.

    The CLI already reaches it; without these an API-only client cannot get at a
    past run at all. Uses the real archive against a temp directory, because the
    thing worth testing is that a saved run round-trips through the endpoint.
    """

    @pytest.fixture
    def saved(self, tmp_path, monkeypatch):
        from procurement_agent.archive import save_run
        from procurement_agent.config import get_settings

        settings = get_settings()
        monkeypatch.setattr(settings, "runs_dir", tmp_path / "runs")
        state = {
            "raw_input": "Custom 465 Dia 2 inch - 200 KG",
            "material_spec": MaterialSpec(material_name="Custom 465", grade="UNS S46500"),
            "vendor_leads": [
                VendorLead(
                    company_name="Precision Alloys Ltd",
                    website="https://precisionalloys.co.uk",
                    source_url="https://precisionalloys.co.uk/custom-465",
                )
            ],
            "search_credits_remaining": 20,
        }
        trace = {"totals": {"llm_calls": 3}, "events": [{"kind": "node_end", "node": "x"}]}
        path = save_run(state, "thread-1", settings, trace)
        return path.stem

    async def test_listing_requires_the_api_key(self):
        async with await client_for(FakeGraph()) as client:
            assert (await client.get("/runs")).status_code == 401

    async def test_fetching_one_requires_the_api_key(self):
        async with await client_for(FakeGraph()) as client:
            assert (await client.get("/runs/anything")).status_code == 401

    async def test_lists_saved_runs(self, saved):
        async with await client_for(FakeGraph()) as client:
            body = (await client.get("/runs", headers=HEADERS)).json()

        assert body["count"] == 1
        assert body["runs"][0]["run_id"] == saved
        # Index entries only - enough to find the run you want.
        assert "vendor_leads" not in body["runs"][0]

    async def test_filters_by_keyword(self, saved):
        async with await client_for(FakeGraph()) as client:
            hit = (await client.get("/runs?q=custom 465", headers=HEADERS)).json()
            miss = (await client.get("/runs?q=inconel", headers=HEADERS)).json()

        assert hit["count"] == 1
        assert miss["count"] == 0

    async def test_returns_one_run_complete(self, saved):
        """"Shows the run completely" - the whole record, trace included."""
        async with await client_for(FakeGraph()) as client:
            body = (await client.get(f"/runs/{saved}", headers=HEADERS)).json()

        assert body["request"] == "Custom 465 Dia 2 inch - 200 KG"
        assert body["vendor_leads"][0]["company_name"] == "Precision Alloys Ltd"
        assert body["summary"] and body["keywords"]
        assert body["totals"]["llm_calls"] == 3
        assert body["trace"][0]["node"] == "x"
        assert body["disclaimer"]

    async def test_unknown_run_is_a_404(self, tmp_path, monkeypatch):
        from procurement_agent.config import get_settings

        monkeypatch.setattr(get_settings(), "runs_dir", tmp_path / "runs")
        async with await client_for(FakeGraph()) as client:
            assert (await client.get("/runs/nope", headers=HEADERS)).status_code == 404


class TestCORS:
    """Without CORS a browser frontend cannot call this API at all.

    Tested against the real app rather than a bare router, because the
    middleware is registered on the app and a router-only fixture would not
    exercise it.
    """

    @pytest.fixture
    def app_client(self):
        from fastapi.testclient import TestClient

        from procurement_agent.api.main import app

        return TestClient(app)

    def test_preflight_allows_a_configured_origin(self, app_client):
        response = app_client.options(
            "/runs",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "x-api-key",
            },
        )
        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == "http://localhost:5173"

    def test_preflight_refuses_an_unknown_origin(self, app_client):
        response = app_client.options(
            "/runs",
            headers={
                "Origin": "https://evil.example",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "x-api-key",
            },
        )
        assert response.headers.get("access-control-allow-origin") is None

    def test_the_api_key_header_is_permitted(self, app_client):
        """A preflight that rejects X-API-Key blocks every authenticated call,
        which presents as an inexplicable CORS error rather than a 401."""
        response = app_client.options(
            "/runs",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "x-api-key",
            },
        )
        assert "x-api-key" in response.headers["access-control-allow-headers"].lower()

    def test_the_origin_is_never_a_wildcard(self):
        """These endpoints take a credential header and return third-party
        contact data. A wildcard origin is precisely what the same-origin policy
        exists to prevent, so it must not be reachable by configuration."""
        from procurement_agent.config import get_settings

        assert "*" not in get_settings().cors_origin_list
