"""Turn a tool's JSON Schema into a pydantic model.

Both LangChain and LlamaIndex want a pydantic model for arguments, while our
tool definitions are plain JSON Schema — which is what the Anthropic and OpenAI
formats need, and what keeps the core dependency-free. This converts one to the
other at adapter time, so JSON Schema stays the single source of truth and the
two frameworks cannot drift apart from it.
"""

from __future__ import annotations

from typing import Any

_JSON_TO_PY: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def model_from_schema(name: str, schema: dict[str, Any]):
    """Build a pydantic model matching a (flat) JSON Schema object.

    Our tool schemas are deliberately flat — a query and maybe a result count —
    so this handles that shape rather than implementing a general JSON Schema
    compiler nobody needs.
    """
    from pydantic import Field, create_model

    properties = schema.get("properties", {})
    required = set(schema.get("required", []))

    fields: dict[str, Any] = {}
    for field_name, spec in properties.items():
        py_type = _JSON_TO_PY.get(spec.get("type", "string"), str)
        description = spec.get("description", "")
        if field_name in required:
            fields[field_name] = (py_type, Field(..., description=description))
        else:
            # Optional in the schema means optional here; the Toolset applies
            # its own default and cap regardless of what the model sends.
            fields[field_name] = (py_type | None, Field(None, description=description))

    return create_model(name, **fields)
