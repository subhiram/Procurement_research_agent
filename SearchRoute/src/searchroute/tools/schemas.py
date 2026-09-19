"""Tool schemas in the shapes different clients expect.

Pure dicts, no dependencies. The Anthropic and OpenAI tool-calling formats carry
the same information under different keys — ``input_schema`` vs a nested
``function.parameters`` — so both are generated from one definition rather than
maintained separately and allowed to drift.
"""

from __future__ import annotations

from typing import Any

from .definitions import ToolDef


def to_anthropic(tool: ToolDef) -> dict[str, Any]:
    """Anthropic Messages API tool shape."""
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.schema,
    }


def to_openai(tool: ToolDef) -> dict[str, Any]:
    """OpenAI chat-completions function shape."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.schema,
        },
    }


def to_plain(tool: ToolDef) -> dict[str, Any]:
    """Vendor-neutral: name, description, JSON Schema. For your own loop, or a
    client whose format isn't covered here."""
    return {
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.schema,
    }


FORMATS = {
    "anthropic": to_anthropic,
    "openai": to_openai,
    "plain": to_plain,
}
