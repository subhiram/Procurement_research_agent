"""Framework adapters.

Every import here is lazy and behind an optional extra. Importing
``searchroute.tools`` must keep working on a bare directory copy with only
``httpx`` installed, so nothing in this package is imported at module load.
"""

from __future__ import annotations

__all__ = ["to_langchain", "to_llamaindex"]


def to_langchain(toolset, **kwargs):
    """Convert a Toolset into LangChain ``StructuredTool`` objects."""
    from .langchain import to_langchain as _impl

    return _impl(toolset, **kwargs)


def to_llamaindex(toolset, **kwargs):
    """Convert a Toolset into LlamaIndex ``FunctionTool`` objects."""
    from .llamaindex import to_llamaindex as _impl

    return _impl(toolset, **kwargs)
