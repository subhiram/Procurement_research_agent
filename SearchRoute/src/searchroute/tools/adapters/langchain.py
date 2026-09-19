"""LangChain adapter.

Written against ``langchain-core`` 1.6.2. LangChain's surface moves quickly; if
``StructuredTool.from_function`` has changed shape, this is the only file that
needs updating.

Requires ``pip install 'searchroute[langchain]'``.
"""

from __future__ import annotations

from typing import Any

from ._schema import model_from_schema


def to_langchain(toolset, **kwargs: Any) -> list[Any]:
    """Convert every tool in a Toolset into a LangChain ``StructuredTool``.

        tools = to_langchain(Toolset(sr))
        agent = create_agent(model, tools)
    """
    try:
        from langchain_core.tools import StructuredTool
    except ImportError as exc:  # pragma: no cover - depends on extras
        raise ImportError(
            "LangChain is required: pip install 'searchroute[langchain]'"
        ) from exc

    built: list[Any] = []
    for tool in toolset.tools:
        built.append(
            StructuredTool.from_function(
                # Async only. Every path underneath is async, and offering a
                # sync entry point would mean spinning an event loop per call
                # inside whatever loop the agent is already running.
                coroutine=_make_coroutine(toolset, tool.name),
                name=tool.name,
                description=tool.description,
                args_schema=model_from_schema(f"{tool.name}_args", tool.schema),
                **kwargs,
            )
        )
    return built


def _make_coroutine(toolset, name: str):
    async def run(**arguments: Any) -> str:
        # Drop unset optionals so the Toolset applies its own defaults rather
        # than seeing an explicit None.
        cleaned = {k: v for k, v in arguments.items() if v is not None}
        result = await toolset.dispatch(name, cleaned)
        # LangChain tools return a string; the error flag is carried in the
        # text, since a raised exception would abort the agent run.
        return result.text

    run.__name__ = name
    return run
