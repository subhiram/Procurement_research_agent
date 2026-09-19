"""The Provider contract.

Providers are self-describing: they declare their capabilities, the depths they
can serve in a single call, their cost model and their free-tier quota. The
router reads those declarations rather than hard-coding vendor knowledge, which
is what lets a user register their own provider and have it routed, budgeted and
failed-over exactly like a built-in.
"""

from __future__ import annotations

import abc
from typing import Any

import httpx

from ..errors import (
    AuthError,
    CapabilityNotSupported,
    ProviderError,
    QuotaExceeded,
    RateLimited,
    TransientError,
)
from ..quota import QuotaPolicy
from ..types import (
    Answer,
    Capability,
    Depth,
    Document,
    ErrorKind,
    SearchQuery,
    SearchResponse,
)


class Provider(abc.ABC):  # noqa: B024 - see below
    """Base class for every search/extract backend.

    Deliberately has no ``@abstractmethod``. Which operations a provider
    implements is declared by ``capabilities``, not enforced by the class: a
    SERP API implements ``search`` only, an extraction service implements
    ``extract`` only, and forcing either to stub out the rest would just add
    dead code. ``ABC`` is here to say "don't instantiate this directly".
    """

    name: str = ""
    capabilities: frozenset[Capability] = frozenset({Capability.SEARCH})
    native_depths: frozenset[Depth] = frozenset({Depth.LINKS, Depth.SNIPPETS})
    """Depths this provider can serve in ONE call. Anything deeper than the max
    of this set has to be filled in by the hydration stage."""

    requires_key: bool = True
    keyed_capabilities: frozenset[Capability] = frozenset()
    """Capabilities that need a key even when the provider is otherwise keyless.

    Jina is the motivating case: ``r.jina.ai`` reads pages without credentials,
    but ``s.jina.ai`` search returns 401. Without this distinction the router
    would offer Jina for search, take a guaranteed auth failure, and trip the
    breaker — disabling the keyless extraction that *does* work.
    """
    quota: QuotaPolicy | None = None
    quality_hint: float = 0.5
    """0-1 search quality, used by the ``quality`` strategy."""
    extract_quality: float = 0.0
    """0-1 extraction quality, tracked separately: the best searcher is rarely
    the best extractor."""
    default_priority: int = 100
    """Lower sorts earlier in the default chain."""
    last_resort: bool = False
    """Forced to the end of every ordering, whatever the strategy says.

    For unofficial or unreliable backends (DuckDuckGo scraping). Without this
    flag a quota-aware strategy would rank them *first* for being free, which is
    exactly backwards: they're the safety net, not the preferred path.
    """
    opt_in_only: bool = False
    """Excluded from environment auto-discovery; must be named explicitly.

    For providers that can charge real money (Brave, which retired its free tier
    and now meters every query past a small credit). Having an API key in the
    environment is not consent to spend from it — that should be a deliberate
    choice, not a side effect of a variable being set.
    """

    min_interval: float = 0.0
    """Minimum seconds between calls to this provider.

    The quota ledger counts *credits*; this counts *rate*, which is a different
    constraint and the one the free public APIs actually enforce. arXiv asks for
    one request every 3 seconds — fanning out against it at full speed gets you
    blocked, and no amount of remaining quota changes that.
    """

    base_url: str = ""
    timeout: float = 20.0

    def __init__(
        self,
        api_key: str | None = None,
        *,
        timeout: float | None = None,
        client: httpx.AsyncClient | None = None,
        **options: Any,
    ):
        self.api_key = api_key
        self.options = options
        if timeout is not None:
            self.timeout = timeout
        self._client = client
        self._owns_client = client is None

    # ---- lifecycle -------------------------------------------------------

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    # ---- configuration ---------------------------------------------------

    @property
    def configured(self) -> bool:
        """Whether this provider is usable. Keyless providers are always ready;
        keyed ones need their key present."""
        return bool(self.api_key) or not self.requires_key

    def supports(self, capability: Capability) -> bool:
        """Whether this provider implements the capability at all."""
        return capability in self.capabilities

    def can_serve(self, capability: Capability) -> bool:
        """Whether it implements the capability *and* has what that needs.

        This is what the router filters on. ``supports`` alone would let a
        keyless Jina be picked for search, which always 401s.
        """
        if capability not in self.capabilities:
            return False
        if capability in self.keyed_capabilities and not self.api_key:
            return False
        return self.configured

    @property
    def max_native_depth(self) -> Depth:
        return max(self.native_depths) if self.native_depths else Depth.LINKS

    def serves_natively(self, depth: Depth) -> bool:
        return depth <= self.max_native_depth and depth in self.native_depths

    # ---- operations ------------------------------------------------------

    async def search(self, query: SearchQuery) -> SearchResponse:
        raise CapabilityNotSupported(f"{self.name} does not support search")

    async def extract(self, urls: list[str], **kwargs: Any) -> list[Document]:
        raise CapabilityNotSupported(f"{self.name} does not support extract")

    async def answer(self, query: SearchQuery) -> Answer:
        raise CapabilityNotSupported(f"{self.name} does not support answer")

    # ---- cost ------------------------------------------------------------

    def cost_of(self, query: SearchQuery) -> int:
        """Credits this query will consume, in the provider's own units.

        Takes the whole query — including its depth — because asking for full
        page contents is not the same spend as asking for links.
        """
        return 1

    def extract_cost_of(self, urls: list[str]) -> int:
        return len(urls)

    def observed_cost(self, response: httpx.Response, fallback: int) -> int:
        """Prefer the provider's own reported usage over our estimate. Providers
        that expose a usage header override this."""
        return fallback

    # ---- errors ----------------------------------------------------------

    def classify_error(self, exc: BaseException) -> ErrorKind:
        """Map an exception to a routing decision.

        The default handles the shapes every HTTP API shares; providers override
        only where they deviate (e.g. an API that returns 200 with an error body).
        """
        if isinstance(exc, ProviderError):
            return exc.kind
        if isinstance(exc, httpx.HTTPStatusError):
            return self._classify_status(exc.response.status_code)
        if isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
            return ErrorKind.TRANSIENT
        return ErrorKind.FATAL

    @staticmethod
    def _classify_status(status: int) -> ErrorKind:
        if status in (401, 403):
            return ErrorKind.AUTH
        if status == 402:
            return ErrorKind.QUOTA
        if status == 429:
            return ErrorKind.RATE_LIMIT
        if status >= 500:
            return ErrorKind.TRANSIENT
        return ErrorKind.FATAL

    def raise_for_status(self, response: httpx.Response) -> None:
        """Turn an HTTP response into the right typed error.

        402 vs 429 is the distinction that matters most: one means the free tier
        is gone until the window resets, the other means slow down.
        """
        if response.is_success:
            return
        status = response.status_code
        body = response.text[:300]
        if status in (401, 403):
            raise AuthError(self.name, f"authentication failed: {body}", status=status)
        if status == 402:
            raise QuotaExceeded(self.name, f"quota exhausted: {body}", status=status)
        if status == 429:
            retry_after = response.headers.get("retry-after")
            raise RateLimited(
                self.name,
                f"rate limited: {body}",
                status=status,
                retry_after=float(retry_after) if retry_after and retry_after.isdigit() else None,
            )
        if status >= 500:
            raise TransientError(self.name, f"server error {status}: {body}", status=status)
        raise ProviderError(self.name, f"http {status}: {body}", status=status)

    # ---- helpers ---------------------------------------------------------

    async def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        """One HTTP round trip with this package's error semantics applied."""
        try:
            response = await self.client.request(method, url, timeout=self.timeout, **kwargs)
        except httpx.TimeoutException as exc:
            raise TransientError(self.name, f"timeout after {self.timeout}s") from exc
        except httpx.TransportError as exc:
            raise TransientError(self.name, f"transport error: {exc}") from exc
        self.raise_for_status(response)
        return response

    def __repr__(self) -> str:
        return f"<Provider {self.name} configured={self.configured}>"
