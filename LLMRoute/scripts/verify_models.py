#!/usr/bin/env python3
"""Diff config/models.yaml against each provider's live model catalogue.

Model ids drift constantly on free tiers: models get renamed, versioned,
promoted out of preview, or retired. A stale id in models.yaml shows up at
runtime as a 404, which the router survives (it parks that endpoint and moves
on) but which quietly costs you an endpoint you thought you had.

Run this whenever you add a model, and periodically after that:

    python scripts/verify_models.py              # check what is configured
    python scripts/verify_models.py --suggest    # also list unused chat models

Only providers whose API key is set are checked; the rest are reported as
skipped. Exits non-zero if any configured endpoint is missing upstream, so it
can sit in CI.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx  # noqa: E402

from llm_router.registry import Endpoint, load_registry  # noqa: E402

TIMEOUT = 30.0

# Where each provider lists its models, and how to read the ids back out.
CATALOGUES: dict[str, dict[str, Any]] = {
    "groq": {
        "url": "https://api.groq.com/openai/v1/models",
        "auth": "bearer",
        "path": ("data", "id"),
    },
    "ollama": {
        # Local daemon, no auth. /api/tags rather than /models: Ollama is not
        # OpenAI-compatible on this endpoint and reports pulled models, not a
        # catalogue, so a "missing" model here means `ollama pull` not a
        # deprecation.
        "url": "http://localhost:11434/api/tags",
        "auth": "none",
        "path": ("models", "name"),
    },
    "mistral": {
        "url": "https://api.mistral.ai/v1/models",
        "auth": "bearer",
        "path": ("data", "id"),
    },
    "google_ai_studio": {
        "url": "https://generativelanguage.googleapis.com/v1beta/models",
        "auth": "query",
        "path": ("models", "name"),
        # Google returns "models/gemini-2.5-flash"
        "strip_prefix": "models/",
        # and lists embedding / TTS models alongside chat ones
        "requires": "generateContent",
    },
    "nvidia_nim": {
        "url": "https://integrate.api.nvidia.com/v1/models",
        "auth": "bearer",
        "path": ("data", "id"),
    },
    "openrouter": {
        # Lists the whole catalogue, paid and free; the `:free` variants appear
        # as their own ids, which is exactly what models.yaml configures, so no
        # filtering is needed to check them. `--suggest` will however list every
        # paid model as "available but not configured" - grep for ":free".
        "url": "https://openrouter.ai/api/v1/models",
        "auth": "bearer",
        "path": ("data", "id"),
    },
}

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"


def fetch(provider: str, api_key: str | None) -> set[str]:
    """Ask a provider what models it actually has."""
    spec = CATALOGUES[provider]
    headers, params = {}, {}
    url = spec["url"]
    if spec["auth"] == "bearer":
        headers["Authorization"] = f"Bearer {api_key}"
    elif spec["auth"] == "query":
        params["key"] = api_key
    if provider == "ollama":
        # The daemon may not be on the default port.
        base = os.environ.get("OLLAMA_BASE_URL")
        if base:
            url = f"{base.rstrip('/')}/api/tags"

    with httpx.Client(timeout=TIMEOUT) as client:
        response = client.get(url, headers=headers, params=params)
        response.raise_for_status()
        payload = response.json()

    container, id_field = spec["path"]
    prefix = spec.get("strip_prefix", "")
    requires = spec.get("requires")

    ids: set[str] = set()
    for entry in payload.get(container) or []:
        if not isinstance(entry, dict):
            continue
        if requires:
            methods = entry.get("supportedGenerationMethods") or entry.get(
                "supportedActions"
            ) or []
            if requires not in methods:
                continue
        model_id = str(entry.get(id_field, ""))
        if prefix and model_id.startswith(prefix):
            model_id = model_id[len(prefix):]
        if model_id:
            ids.add(model_id)
    return ids


def api_key_for(endpoints: Iterable[Endpoint]) -> str | None:
    for endpoint in endpoints:
        if endpoint.api_key:
            return endpoint.api_key
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suggest", action="store_true",
        help="also list chat models the provider offers that are not configured",
    )
    parser.add_argument("--provider", help="check only this provider")
    args = parser.parse_args()

    registry = load_registry()
    by_provider: dict[str, list[Endpoint]] = {}
    for endpoint in registry.all_endpoints():
        by_provider.setdefault(endpoint.provider, []).append(endpoint)

    problems = 0
    checked = 0
    for provider, endpoints in sorted(by_provider.items()):
        if args.provider and provider != args.provider:
            continue
        print(f"\n{provider}")
        if provider not in CATALOGUES:
            print(f"  {DIM}no catalogue endpoint known; cannot verify{RESET}")
            continue

        key = api_key_for(endpoints)
        # A keyless provider (Ollama) declares no api_key_env at all, so having
        # no key is the normal state rather than a missing configuration.
        if not key and CATALOGUES[provider]["auth"] != "none":
            names = " or ".join(endpoints[0].api_key_env) or "an API key"
            print(f"  {DIM}skipped - set {names} to verify{RESET}")
            continue

        try:
            available = fetch(provider, key)
        except httpx.HTTPStatusError as exc:
            print(f"  {RED}catalogue request failed: HTTP {exc.response.status_code}{RESET}")
            problems += 1
            continue
        except httpx.HTTPError as exc:
            print(f"  {RED}catalogue request failed: {exc}{RESET}")
            problems += 1
            continue

        checked += 1
        configured = set()
        for endpoint in sorted(endpoints, key=lambda e: e.model_id):
            configured.add(endpoint.model_id)
            if endpoint.model_id in available:
                print(f"  {GREEN}ok{RESET}      {endpoint.model_id}  "
                      f"{DIM}({endpoint.logical_model}, tier {endpoint.tier}){RESET}")
            else:
                problems += 1
                near = sorted(
                    m for m in available
                    if _looks_like(endpoint.model_id, m)
                )
                hint = f"  did you mean: {', '.join(near[:3])}" if near else ""
                print(f"  {RED}MISSING{RESET} {endpoint.model_id}  "
                      f"{DIM}({endpoint.logical_model}){RESET}{YELLOW}{hint}{RESET}")

        if args.suggest:
            unused = sorted(available - configured)
            if unused:
                print(f"  {DIM}available but not configured ({len(unused)}):{RESET}")
                for model_id in unused:
                    flag = " [non-chat - do not add]" if _is_non_chat(registry, model_id) else ""
                    print(f"    {DIM}{model_id}{flag}{RESET}")

    print()
    if problems:
        print(f"{RED}{problems} problem(s).{RESET} "
              f"Update llm_router/config/models.yaml (and limits.yaml) to match.")
        return 1
    if not checked:
        print(f"{YELLOW}Nothing was verified - no API keys were set.{RESET} "
              f"Model ids in models.yaml remain unconfirmed.")
        return 2
    print(f"{GREEN}All configured model ids exist upstream "
          f"({checked} provider(s) checked).{RESET}")
    return 0


def _looks_like(wanted: str, candidate: str) -> bool:
    """Cheap fuzzy match, just to make a rename obvious."""
    wanted_parts = set(_tokens(wanted))
    candidate_parts = set(_tokens(candidate))
    if not wanted_parts:
        return False
    overlap = len(wanted_parts & candidate_parts) / len(wanted_parts)
    return overlap >= 0.5


def _tokens(model_id: str) -> list[str]:
    out, current = [], ""
    for char in model_id.lower():
        if char.isalnum():
            current += char
        elif current:
            out.append(current)
            current = ""
    if current:
        out.append(current)
    return out


def _is_non_chat(registry: Any, model_id: str) -> bool:
    return registry._is_non_chat(model_id)


if __name__ == "__main__":
    raise SystemExit(main())
