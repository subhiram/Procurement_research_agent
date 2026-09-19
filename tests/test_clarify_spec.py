"""Clarification, and surviving a replay.

The interesting property here is not what gets asked but what happens when the
asking node runs twice. LangGraph re-executes an interrupted node from the top
when the answer arrives, so anything non-deterministic inside it — a model call,
for instance — can reach a different conclusion the second time and strand the
buyer's answer.
"""

from __future__ import annotations

import pytest

from procurement_agent.graph.nodes import clarify_spec as node_mod
from procurement_agent.graph.nodes.clarify_spec import (
    ask_clarification,
    clarify_spec,
    route_after_clarify,
)
from procurement_agent.graph.state import ClarificationQuestion, MaterialSpec


class _Stub:
    """Returns a scripted payload, recording how often it was asked."""

    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    async def ainvoke(self, messages, *a, **k):
        self.calls += 1
        return self.payload


def _questions(*fields):
    return [
        ClarificationQuestion(field=f, question=f"what {f}?", why="needed")
        for f in fields
    ]


@pytest.fixture
def state():
    return {
        "raw_input": "I need some Hastelloy",
        "material_spec": MaterialSpec(material_name="Hastelloy"),
        "clarification_history": [],
    }


class TestDeciding:
    async def test_records_the_questions_without_asking_yet(
        self, monkeypatch, state, fake_search
    ):
        """The decision is committed to state before anyone is interrupted."""
        monkeypatch.setattr(
            node_mod, "get_model",
            lambda *a, **k: _Stub(node_mod.Questions(questions=_questions("form"))),
        )

        result = await clarify_spec(state)

        assert len(result["pending_questions"]) == 1
        assert result["status"] == "clarifying"

    async def test_a_sufficient_spec_asks_nothing(self, monkeypatch, state, fake_search):
        """Proceeding silently is a success, not a failure."""
        monkeypatch.setattr(
            node_mod, "get_model",
            lambda *a, **k: _Stub(node_mod.Questions(questions=[])),
        )

        result = await clarify_spec(state)

        assert result["pending_questions"] == []
        assert route_after_clarify(result) == "material_research"

    async def test_questions_route_to_the_asking_node(self):
        assert route_after_clarify({"pending_questions": _questions("form")}) == (
            "ask_clarification"
        )


class TestReplaySafety:
    """The bug this split exists for.

    A resume that had failed once was retried; the retry landed on a different
    provider; that provider decided no clarification was needed; and the answer
    was discarded in silence. The run then searched for "Hastelloy" — a family
    of dozens of alloys — instead of the "Hastelloy C-276" just supplied.
    """

    async def test_the_asking_node_never_calls_a_model_to_decide(
        self, monkeypatch, state
    ):
        """It reads the questions from state, so a replay asks the same thing.

        This is what makes re-execution safe: there is no decision left inside
        the interrupting node for a model to make differently.
        """
        decider = _Stub(node_mod.Questions(questions=[]))
        monkeypatch.setattr(node_mod, "get_model", lambda *a, **k: decider)
        monkeypatch.setattr(
            node_mod, "interrupt", lambda payload: "C-276, round bar, 25mm"
        )
        monkeypatch.setattr(
            node_mod, "_apply_answers",
            _apply_returning(MaterialSpec(material_name="Hastelloy C-276", form="bar")),
        )

        state["pending_questions"] = _questions("grade", "form")
        result = await ask_clarification(state)

        # The only model call is the one applying the answers - never one that
        # could decide the questions away.
        assert result["material_spec"].material_name == "Hastelloy C-276"
        assert len(result["clarification_history"]) == 1

    async def test_the_answer_is_recorded_even_when_it_arrives_twice(
        self, monkeypatch, state
    ):
        """Re-executing the asking node must produce the same outcome."""
        monkeypatch.setattr(node_mod, "interrupt", lambda payload: "C-276")
        monkeypatch.setattr(
            node_mod, "_apply_answers",
            _apply_returning(MaterialSpec(material_name="Hastelloy C-276")),
        )
        state["pending_questions"] = _questions("grade")

        first = await ask_clarification(dict(state))
        second = await ask_clarification(dict(state))

        assert first["material_spec"] == second["material_spec"]
        assert len(first["clarification_history"]) == len(second["clarification_history"])

    async def test_asking_with_nothing_pending_is_a_no_op(self, state):
        """Reachable only if state was manipulated directly; must not hang on an
        interrupt nobody will answer."""
        result = await ask_clarification({**state, "pending_questions": []})
        assert result["status"] == "researching"


def _apply_returning(spec):
    async def _apply(*a, **k):
        return spec

    return _apply
