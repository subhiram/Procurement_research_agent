"""Shared helpers for provider implementations."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

#: SERP APIs return dates in whatever the underlying page said, so we accept a
#: handful of common shapes rather than failing the whole result on a date.
_DATE_FORMATS = (
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%b %d, %Y",
    "%d %b %Y",
    "%B %d, %Y",
    "%d %B %Y",
)

#: Google-style relative dates: "3 days ago", "2 hours ago".
_RELATIVE = re.compile(r"(\d+)\s+(second|minute|hour|day|week|month|year)s?\s+ago", re.I)

_RELATIVE_SECONDS = {
    "second": 1,
    "minute": 60,
    "hour": 3600,
    "day": 86400,
    "week": 604800,
    "month": 2592000,
    "year": 31536000,
}


def parse_date(value: Any, *, now: datetime | None = None) -> datetime | None:
    """Best-effort date parsing.

    Always returns either an aware datetime or ``None`` — never raises. A result
    with an unparseable date is still a perfectly good result, so a date that
    doesn't fit any known shape is simply dropped.
    """
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if not value or not isinstance(value, str):
        return None

    text = value.strip()
    if not text:
        return None

    # ISO-8601, the common case.
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        pass

    match = _RELATIVE.search(text)
    if match:
        from datetime import timedelta

        amount, unit = int(match.group(1)), match.group(2).lower()
        base = now or datetime.now(timezone.utc)
        return base - timedelta(seconds=amount * _RELATIVE_SECONDS[unit])

    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def clamp_score(value: Any) -> float | None:
    """Normalize a provider's relevance score into 0-1, or drop it.

    Scores are only ever used for ordering within one provider's results, never
    compared across providers — that's what rank fusion is for.
    """
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if score != score:  # NaN
        return None
    return max(0.0, min(1.0, score))
