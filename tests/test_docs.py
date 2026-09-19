"""The documentation is checked, not trusted.

Docs drift silently: a test gets renamed, a setting is removed, and the document
keeps confidently describing something that no longer exists. Everything here is
mechanically verifiable, so it is verified.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _test_names_defined() -> set[str]:
    names: set[str] = set()
    for root in ("tests", "LLMRoute/tests", "SearchRoute/tests"):
        for path in (ROOT / root).rglob("test_*.py"):
            names.update(re.findall(r"def (test_[a-z0-9_]+)", path.read_text()))
    return names


def test_every_test_cited_in_the_fallback_docs_exists():
    """fallbacks.md claims each row is held up by a named test.

    A renamed test would leave the claim standing with nothing behind it, which
    is worse than no claim: it reads as coverage that is not there.
    """
    doc = (ROOT / "docs" / "fallbacks.md").read_text()
    cited = set(re.findall(r"`(test_[a-z0-9_]+)`", doc))
    assert cited, "no tests cited - has the document changed shape?"

    missing = sorted(cited - _test_names_defined())
    assert not missing, f"cited in fallbacks.md but no longer defined: {missing}"


def test_every_setting_the_runbook_documents_still_exists():
    """operations.md lists environment variables. A removed setting documented
    as live sends someone configuring something that is silently ignored."""
    from procurement_agent.config import Settings

    doc = (ROOT / "docs" / "operations.md").read_text()
    # Only the settings tables, matched as a row whose first cell is the name.
    # Scanning the whole document instead picked up a Python constant and a SQL
    # keyword, which made the test noisy rather than useful.
    documented = set(re.findall(r"^\| `([A-Z][A-Z0-9_]{3,})` \|", doc, re.M))
    assert documented, "no settings found - have the runbook's tables changed shape?"

    known = {name.upper() for name in Settings.model_fields}
    # Read from the environment by the routers and SDKs rather than through
    # Settings, so they are real but will never appear in model_fields.
    external = {
        "GROQ_API_KEY", "MISTRAL_API_KEY", "NVIDIA_API_KEY", "GOOGLE_API_KEY",
        "GEMINI_API_KEY", "OPENROUTER_API_KEY", "TAVILY_API_KEY", "EXA_API_KEY",
        "FIRECRAWL_API_KEY", "BRAVE_API_KEY", "SEARCHROUTE_CONTACT",
        "XDG_STATE_HOME", "OLLAMA_BASE_URL",
        # Read by docker-compose, not by the application.
        "API_PORT",
    }

    unknown = sorted(documented - known - external)
    assert not unknown, f"operations.md documents settings that do not exist: {unknown}"


@pytest.mark.parametrize(
    "doc",
    ["docs/README.md", "docs/architecture.md", "docs/design-decisions.md",
     "docs/fallbacks.md", "docs/operations.md", "API_CONTRACT.md", "README.md"],
)
def test_internal_links_resolve(doc):
    """A broken link in an index is how documentation starts being ignored."""
    path = ROOT / doc
    text = path.read_text()
    broken = []
    for target in re.findall(r"\]\(([^)#:]+\.md)(?:#[^)]*)?\)", text):
        if not (path.parent / target).resolve().exists():
            broken.append(target)
    assert not broken, f"{doc} links to missing files: {broken}"


def test_the_graph_diagram_lists_the_real_nodes():
    """architecture.md draws the graph by hand from generated output. A node
    added to the graph and not to the diagram makes the diagram a lie."""
    from procurement_agent.graph.build import build_graph

    diagram = (ROOT / "docs" / "architecture.md").read_text()
    nodes = set(build_graph().nodes)
    # draft_outreach_email is documented as deliberately omitted: it is a
    # separate entry point, not part of the research pass.
    missing = sorted(n for n in nodes - {"draft_outreach_email"} if n not in diagram)
    assert not missing, f"nodes missing from the architecture diagram: {missing}"
