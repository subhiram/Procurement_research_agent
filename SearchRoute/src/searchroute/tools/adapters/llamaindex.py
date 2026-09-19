"""LlamaIndex adapter.

Written against ``llama-index-core`` 0.14.24.

Requires ``pip install 'searchroute[llamaindex]'``.
"""

from __future__ import annotations

from typing import Any

from ._schema import model_from_schema


def to_llamaindex(toolset, **kwargs: Any) -> list[Any]:
    """Convert every tool in a Toolset into a LlamaIndex ``FunctionTool``.

        tools = to_llamaindex(Toolset(sr))
        agent = FunctionAgent(tools=tools, llm=llm)
    """
    try:
        from llama_index.core.tools import FunctionTool
    except ImportError as exc:  # pragma: no cover - depends on extras
        raise ImportError(
            "LlamaIndex is required: pip install 'searchroute[llamaindex]'"
        ) from exc

    built: list[Any] = []
    for tool in toolset.tools:
        built.append(
            FunctionTool.from_defaults(
                async_fn=_make_coroutine(toolset, tool.name),
                name=tool.name,
                description=tool.description,
                fn_schema=model_from_schema(f"{tool.name}_args", tool.schema),
                **kwargs,
            )
        )
    return built


def _make_coroutine(toolset, name: str):
    async def run(**arguments: Any) -> str:
        cleaned = {k: v for k, v in arguments.items() if v is not None}
        result = await toolset.dispatch(name, cleaned)
        return result.text

    run.__name__ = name
    return run
