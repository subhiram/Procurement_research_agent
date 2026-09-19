"""Exception hierarchy.

Providers raise ``ProviderError`` subclasses; the router turns those into
``ErrorKind`` via ``Provider.classify_error`` and decides whether to retry the
same provider, skip to the next, or disable it for the process.
"""

from __future__ import annotations

from .types import Attempt, ErrorKind


class SearchRouteError(Exception):
    """Base for everything this package raises."""


class ConfigError(SearchRouteError):
    """Bad or missing configuration — a typo'd provider name, an absent key."""


class CapabilityNotSupported(SearchRouteError):
    """Asked a provider for something it does not declare. A guard, not a
    control-flow path: the router filters on capability before dispatching."""


class ProviderError(SearchRouteError):
    """Base for a failure attributable to one provider."""

    kind: ErrorKind = ErrorKind.FATAL

    def __init__(self, provider: str, message: str, *, status: int | None = None):
        self.provider = provider
        self.status = status
        super().__init__(f"[{provider}] {message}")


class AuthError(ProviderError):
    """Bad or missing credentials. Surfaced loudly — a wrong key should not be
    silently masked by a fallback."""

    kind = ErrorKind.AUTH


class QuotaExceeded(ProviderError):
    """The free tier is drained. The provider is skipped until its window resets."""

    kind = ErrorKind.QUOTA


class RateLimited(ProviderError):
    """Too fast, not out of credit. Backoff-and-retry, then move on."""

    kind = ErrorKind.RATE_LIMIT

    def __init__(
        self,
        provider: str,
        message: str,
        *,
        status: int | None = None,
        retry_after: float | None = None,
    ):
        self.retry_after = retry_after
        super().__init__(provider, message, status=status)


class TransientError(ProviderError):
    """Timeout, connection reset, 5xx. Worth retrying."""

    kind = ErrorKind.TRANSIENT


class NoProviderAvailable(SearchRouteError):
    """Every candidate was exhausted. Carries the full attempt trail so the
    caller can see *why* each one was ruled out."""

    def __init__(self, message: str, attempts: list[Attempt] | None = None):
        self.attempts = attempts or []
        if self.attempts:
            detail = "; ".join(
                f"{a.provider}: {a.error or 'ok'}" for a in self.attempts
            )
            message = f"{message} (attempts: {detail})"
        super().__init__(message)
