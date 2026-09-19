"""Expose SearchRoute to an LLM as tools.

    from searchroute import SearchRoute
    from searchroute.tools import Toolset

    sr = SearchRoute(profile="rag")          # you choose the routing
    tools = Toolset(sr)

    tools.anthropic()                        # schemas for the Messages API
    result = await tools.dispatch("web_search", {"query": "..."})
    result.text                              # compact, LLM-ready

**No model calls happen here.** This module emits schemas and dispatches to the
client; the agent loop lives in your application. That is the same boundary the
rest of the library holds.
"""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..errors import SearchRouteError
from ..types import Capability, Depth
from .definitions import (
    FORBIDDEN_PARAMS,
    ROUTING,
    TOOLS,
    TOOLS_BY_NAME,
    ToolDef,
    render_results,
)
from .schemas import FORMATS, to_anthropic, to_openai, to_plain

__all__ = [
    "Toolset",
    "ToolResult",
    "TOOLS",
    "ToolDef",
    "FORBIDDEN_PARAMS",
    "to_anthropic",
    "to_openai",
    "to_plain",
]


@dataclass(slots=True)
class ToolResult:
    """What a tool call produced.

    ``text`` is what you hand back to the model; ``data`` is the structured form
    for your own logging or UI. ``is_error`` lets a caller set the Anthropic
    ``is_error`` flag on the tool_result block without string-sniffing.
    """

    text: str
    data: Any = None
    is_error: bool = False

    def __str__(self) -> str:
        return self.text


@dataclass
class Toolset:
    """A fixed, model-safe tool surface bound to one configured client."""

    client: Any
    """A ``SearchRoute`` or ``AsyncSearchRoute``."""
    allow: Sequence[str] | None = None
    """Which tools to expose. ``None`` means all of them."""
    max_results_cap: int = 10
    """Hard ceiling on ``max_results``, whatever the model asks for."""
    default_results: int = 5
    read_page_chars: int | None = None
    """Optional cap on a single page's text. ``None`` (the default) returns the
    page in full — the two-step design exists so this doesn't have to truncate."""

    _tools: tuple[ToolDef, ...] = field(init=False, default=())

    def __post_init__(self) -> None:
        if self.allow is None:
            self._tools = TOOLS
        else:
            unknown = set(self.allow) - set(TOOLS_BY_NAME)
            if unknown:
                raise ValueError(
                    f"unknown tool(s): {', '.join(sorted(unknown))}; "
                    f"available: {', '.join(TOOLS_BY_NAME)}"
                )
            self._tools = tuple(TOOLS_BY_NAME[n] for n in self.allow)

    # ---- schema emission -------------------------------------------------

    @property
    def tools(self) -> tuple[ToolDef, ...]:
        return self._tools

    @property
    def names(self) -> list[str]:
        return [t.name for t in self._tools]

    def anthropic(self) -> list[dict[str, Any]]:
        return [to_anthropic(t) for t in self._tools]

    def openai(self) -> list[dict[str, Any]]:
        return [to_openai(t) for t in self._tools]

    def plain(self) -> list[dict[str, Any]]:
        return [to_plain(t) for t in self._tools]

    def schemas(self, fmt: str = "plain") -> list[dict[str, Any]]:
        try:
            render = FORMATS[fmt]
        except KeyError:
            raise ValueError(
                f"unknown format {fmt!r}; available: {', '.join(FORMATS)}"
            ) from None
        return [render(t) for t in self._tools]

    # ---- dispatch --------------------------------------------------------

    async def dispatch(self, name: str, arguments: dict[str, Any] | None = None) -> ToolResult:
        """Run one tool call.

        Never raises for an expected failure. A tool that throws into an agent
        loop kills the turn; a tool that returns "no search provider is
        configured" lets the model tell the user what's wrong.
        """
        args = dict(arguments or {})
        tool = TOOLS_BY_NAME.get(name)
        if tool is None or tool not in self._tools:
            return ToolResult(
                text=f"Unknown tool {name!r}. Available: {', '.join(self.names)}.",
                is_error=True,
            )

        try:
            handler = getattr(self, f"_run_{tool.handler}")
            return await handler(args)
        except SearchRouteError as exc:
            return ToolResult(
                text=self._explain(exc, tool.name), data={"error": str(exc)}, is_error=True
            )
        except Exception as exc:  # noqa: BLE001 - must not escape into the loop
            return ToolResult(
                text=f"The {name} tool failed: {exc}",
                data={"error": str(exc)},
                is_error=True,
            )

    @staticmethod
    def _explain(exc: SearchRouteError, tool_name: str) -> str:
        """Turn a library error into something a model can act on or report.

        Names the operation that actually failed. Saying "Search failed" when
        ``read_page`` could not run would send the model off to re-search
        instead of reporting the real problem.
        """
        message = str(exc)
        operation = "Reading the page" if tool_name == "read_page" else "Search"

        if "no provider" in message:
            need = (
                "a page-extraction provider (e.g. FIRECRAWL_API_KEY, or install "
                "searchroute[extract] for the keyless local extractor)"
                if tool_name == "read_page"
                else "a search provider (e.g. TAVILY_API_KEY or EXA_API_KEY)"
            )
            return (
                f"{operation} is unavailable: no provider is configured. "
                f"Tell the user their setup needs {need}. Details: {message}"
            )
        return f"{operation} failed: {message}"

    # ---- handlers --------------------------------------------------------

    def _limit(self, args: dict[str, Any]) -> int:
        """Clamp what the model asked for. The cap is enforced here rather than
        in the schema because a schema is a hint; this is a guarantee."""
        try:
            wanted = int(args.get("max_results") or self.default_results)
        except (TypeError, ValueError):
            wanted = self.default_results
        return max(1, min(wanted, self.max_results_cap))

    async def _search(self, args: dict[str, Any], tool_name: str) -> ToolResult:
        query = str(args.get("query") or "").strip()
        if not query:
            return ToolResult(text="No query provided.", is_error=True)

        routing = ROUTING[tool_name]
        limit = self._limit(args)
        response = await self._call(
            self.client.search,
            query,
            capability=routing["capability"],
            depth=routing["depth"],
            max_results=limit,
        )
        return ToolResult(
            text=render_results(response, limit),
            data={
                "query": query,
                "providers_used": response.providers_used,
                "degraded": response.degraded,
                "results": [
                    {"title": r.title, "url": r.url, "snippet": r.snippet}
                    for r in response.results[:limit]
                ],
            },
        )

    async def _run_web_search(self, args: dict[str, Any]) -> ToolResult:
        return await self._search(args, "web_search")

    async def _run_academic_search(self, args: dict[str, Any]) -> ToolResult:
        return await self._search(args, "academic_search")

    async def _run_news_search(self, args: dict[str, Any]) -> ToolResult:
        return await self._search(args, "news_search")

    async def _run_reference_lookup(self, args: dict[str, Any]) -> ToolResult:
        """Wikipedia serves CONTENT natively, so this returns article text in one
        call — no read_page hop needed."""
        query = str(args.get("query") or "").strip()
        if not query:
            return ToolResult(text="No query provided.", is_error=True)

        response = await self._call(
            self.client.search,
            query,
            capability=Capability.REFERENCE,
            depth=Depth.CONTENT,
            max_results=1,
        )
        if not response.results:
            return ToolResult(text=f"No encyclopedia article found for {query!r}.")

        hit = response.results[0]
        if hit.content:
            return ToolResult(
                text=f"# {hit.title}\n{hit.url}\n\n{hit.content}",
                data={"title": hit.title, "url": hit.url, "chars": len(hit.content)},
            )
        return ToolResult(text=render_results(response, 1), data={"url": hit.url})

    async def _run_read_page(self, args: dict[str, Any]) -> ToolResult:
        """One URL per call, deliberately: batching is how a model would blow a
        credit budget in a single turn."""
        url = str(args.get("url") or "").strip()
        if not url:
            return ToolResult(text="No URL provided.", is_error=True)

        docs = await self._call(self.client.extract, [url])
        if not docs:
            return ToolResult(text=f"Could not read {url}.", is_error=True)

        doc = docs[0]
        if not doc.ok or not doc.content:
            return ToolResult(
                text=(
                    f"Could not read {url}: {doc.error or 'no content'}. "
                    "The page may be paywalled, blocked, or JavaScript-rendered. "
                    "Try a different source."
                ),
                data={"url": url, "error": doc.error},
                is_error=True,
            )

        text = doc.content
        if self.read_page_chars and len(text) > self.read_page_chars:
            text = text[: self.read_page_chars] + "\n\n[truncated]"

        header = f"# {doc.title}\n{doc.url}\n\n" if doc.title else f"{doc.url}\n\n"
        return ToolResult(
            text=header + text,
            data={"url": doc.url, "title": doc.title, "chars": len(doc.content)},
        )

    # ---- sync/async bridge ----------------------------------------------

    @staticmethod
    async def _call(fn, *args, **kwargs):
        """Work with either client. ``SearchRoute.search`` is sync, its async
        twin returns a coroutine — awaiting the right one keeps ``dispatch``
        uniform for callers."""
        result = fn(*args, **kwargs)
        if inspect.isawaitable(result):
            return await result
        return result

    # ---- convenience -----------------------------------------------------

    def describe(self) -> str:
        """Human-readable summary, for logs and docs."""
        return "\n".join(
            f"{t.name}({', '.join(t.schema['properties'])}) — {t.description.split('.')[0]}."
            for t in self._tools
        )
