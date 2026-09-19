"""The LLM-facing tool surface.

The governing principle: **intent goes to the model, routing stays with the
operator.**

``search()`` takes about fourteen parameters, and the expensive ones —
``depth="content"``, ``strategy="fanout"``, ``max_hydrate`` — are precisely the
ones a model will misjudge, because it cannot see your quota. So none of them
appear in a tool schema. The developer fixes routing when constructing the
``Toolset``; the model supplies a query and a kind of lookup.

The tools are narrow and distinctly named on purpose. Models select on name and
description far more reliably than they pick a value out of a ``mode`` enum, so
``academic_search`` beats ``search(mode="academic")`` in practice.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..types import Capability, Depth, SearchResponse


@dataclass(frozen=True, slots=True)
class ToolDef:
    """One tool: a name, a description the model reads, and a tiny schema."""

    name: str
    description: str
    schema: dict[str, Any]
    handler: str
    """Which Toolset coroutine serves it."""


def _query_schema(with_max: bool, query_hint: str) -> dict[str, Any]:
    props: dict[str, Any] = {"query": {"type": "string", "description": query_hint}}
    required = ["query"]
    if with_max:
        props["max_results"] = {
            "type": "integer",
            "description": "How many results to return. Defaults to 5.",
            "minimum": 1,
            "maximum": 20,
        }
    return {
        "type": "object",
        "properties": props,
        "required": required,
        "additionalProperties": False,
    }


#: Descriptions are written for a model deciding which tool to call, so they say
#: what each is *for* and when not to use it — that is what drives correct
#: selection far more than the schema does.
TOOLS: tuple[ToolDef, ...] = (
    ToolDef(
        name="web_search",
        description=(
            "Search the public web and get back a ranked list of results, each with "
            "a title, URL and short snippet. Use this first for almost any factual "
            "question. It does NOT return full page text — if a result looks worth "
            "reading in full, call read_page with its URL."
        ),
        schema=_query_schema(True, "The search query. Plain natural language works well."),
        handler="web_search",
    ),
    ToolDef(
        name="read_page",
        description=(
            "Fetch one web page and return its full text as clean markdown. Use this "
            "after web_search when a snippet is not enough and you need the actual "
            "content — for quoting, checking details, or reading an argument. Takes "
            "exactly one URL; call it again for another page."
        ),
        schema={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The full URL of the page to read, including https://.",
                }
            },
            "required": ["url"],
            "additionalProperties": False,
        },
        handler="read_page",
    ),
    ToolDef(
        name="academic_search",
        description=(
            "Search scholarly literature — papers, preprints and publication records "
            "from arXiv, PubMed and Crossref. Use this for research questions, when "
            "you need citable sources, or when the user asks what the literature says. "
            "Prefer it over web_search for anything academic."
        ),
        schema=_query_schema(True, "The research topic, paper title, or author."),
        handler="academic_search",
    ),
    ToolDef(
        name="reference_lookup",
        description=(
            "Look something up in an encyclopedia (Wikipedia) and get the article "
            "text. Use this for background on a well-established topic, person, place "
            "or concept — the kind of thing an encyclopedia covers. Not for current "
            "events or niche technical details."
        ),
        schema=_query_schema(False, "The topic, person, place or concept to look up."),
        handler="reference_lookup",
    ),
    ToolDef(
        name="news_search",
        description=(
            "Search recent news articles. Use this when the question is about current "
            "events, or when recency matters more than authority."
        ),
        schema=_query_schema(True, "The news topic or event."),
        handler="news_search",
    ),
)

TOOLS_BY_NAME: dict[str, ToolDef] = {t.name: t for t in TOOLS}

#: How each tool maps onto the router. This is the operator's half of the
#: contract — fixed here, never exposed to the model.
ROUTING: dict[str, dict[str, Any]] = {
    "web_search": {"capability": Capability.SEARCH, "depth": Depth.SNIPPETS},
    "academic_search": {"capability": Capability.ACADEMIC, "depth": Depth.SNIPPETS},
    "reference_lookup": {"capability": Capability.REFERENCE, "depth": Depth.CONTENT},
    "news_search": {"capability": Capability.NEWS, "depth": Depth.SNIPPETS},
}

#: Parameter names that must never appear in a schema handed to a model. Asserted
#: in the tests, because this is the whole safety argument.
FORBIDDEN_PARAMS = frozenset(
    {"depth", "strategy", "providers", "max_hydrate", "reserve_pct", "hydrate_providers"}
)


def render_results(response: SearchResponse, limit: int | None = None) -> str:
    """Compact, token-cheap rendering of search results for a model.

    Never includes page bodies: a search that dumped ten full articles into the
    context would make the tool unusable after two calls. Full text is what
    ``read_page`` is for.
    """
    results = response.results[:limit] if limit else response.results
    if not results:
        return "No results found."

    lines = []
    for i, hit in enumerate(results, start=1):
        lines.append(f"{i}. {hit.title or '(untitled)'}")
        lines.append(f"   {hit.url}")
        if hit.published_date:
            lines.append(f"   published: {hit.published_date.date().isoformat()}")
        snippet = (hit.summary or hit.snippet or "").strip()
        if snippet:
            lines.append(f"   {' '.join(snippet.split())[:400]}")
        lines.append("")

    text = "\n".join(lines).rstrip()
    # Surface degradation rather than letting the model assume it saw everything.
    if response.notes:
        text += "\n\nNote: " + " ".join(response.notes)
    return text


Handler = Callable[..., Any]
