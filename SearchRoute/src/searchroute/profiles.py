"""Named presets.

Depth, strategy and budgets travel together — a research client wants full
content, wide recall and a generous hydration budget, while a quick-lookup tool
wants none of that. Bundling them means an app picks one word instead of wiring
five knobs, and every value stays overridable per call.
"""

from __future__ import annotations

from typing import Any

from .errors import ConfigError
from .types import Depth

PROFILES: dict[str, dict[str, Any]] = {
    "quicklook": {
        # Cheapest useful call: one provider, snippets, no extraction spend.
        "depth": Depth.SNIPPETS,
        "strategy": "priority",
        "max_results": 5,
        "max_hydrate": 0,
    },
    "cheap": {
        # Stretch the free tiers as far as they go.
        "depth": Depth.SNIPPETS,
        "strategy": "quota_aware",
        "max_results": 10,
        "max_hydrate": 0,
    },
    "fast": {
        "depth": Depth.SNIPPETS,
        "strategy": "race",
        "max_results": 10,
        "max_hydrate": 0,
    },
    "rag": {
        # Full text for a retrieval pipeline, but from a single provider so the
        # credit cost per query stays predictable.
        "depth": Depth.CONTENT,
        "strategy": "quota_aware",
        "max_results": 10,
        "max_hydrate": 8,
    },
    "research": {
        # Widest recall: several providers fused, most results hydrated. This is
        # the expensive one — a single query can cost a dozen credits.
        "depth": Depth.CONTENT,
        "strategy": "fanout",
        "max_results": 25,
        "max_hydrate": 15,
    },
}


def get(name: str) -> dict[str, Any]:
    try:
        return dict(PROFILES[name])
    except KeyError:
        raise ConfigError(
            f"unknown profile {name!r}; available: {', '.join(sorted(PROFILES))}"
        ) from None


def names() -> list[str]:
    return sorted(PROFILES)
