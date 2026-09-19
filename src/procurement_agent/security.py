"""API authentication.

A single shared key, which is proportionate for a small internal tool. It is
isolated here so replacing it with per-user tokens later touches one module.
"""

from __future__ import annotations

import secrets

from fastapi import Header, HTTPException, status

from procurement_agent.config import get_settings


async def require_api_key(x_api_key: str = Header(default="")) -> None:
    """Reject requests without the shared key."""
    expected = get_settings().api_key
    # Constant-time comparison: a shared secret compared with `==` leaks its
    # prefix to a patient caller.
    if not expected or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing or invalid X-API-Key",
        )
