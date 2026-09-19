"""Keyless local extraction — fetch the page ourselves and clean it.

This is the tail of the extract chain, and the reason ``depth="content"`` works
with no API keys at all. It is a genuine fallback, not an equal: it fetches the
raw HTML and runs a readability extractor over it, so anything that renders its
content client-side comes back empty. We report that as a failure rather than
returning a shell of navigation chrome — a research agent citing an empty page
is worse than one that knows the page is missing.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from ..types import Capability, Document
from .base import Provider

#: Identify honestly rather than impersonating a browser.
#:
#: This is not just etiquette, it measurably works better. Wikipedia's policy
#: rejects generic browser UAs from scripts: a spoofed Chrome string gets a 403
#: on every article, while this descriptive one gets a 200. Sites that want to
#: be read by tools let identified tools read them.
#:
#: Operators running this at any volume should append contact details, which
#: several sites' bot policies ask for:
#:     HTTPExtractProvider(user_agent="MyApp/1.0 (+https://example.com; me@example.com)")
_USER_AGENT = "SearchRoute/0.1.0 (+https://github.com/subhiram/searchroute)"

#: Anything not in here is not something a readability extractor can help with.
_HTML_TYPES = ("text/html", "application/xhtml+xml", "text/plain")


class HTTPExtractProvider(Provider):
    name = "http_extract"
    capabilities = frozenset({Capability.EXTRACT})
    requires_key = False
    quota = None
    quality_hint = 0.0
    extract_quality = 0.35
    """Well below the hosted extractors: no JS rendering, no proxy rotation."""
    default_priority = 800
    timeout = 15.0

    @property
    def user_agent(self) -> str:
        """Overridable so operators can add contact details."""
        return self.options.get("user_agent") or _USER_AGENT

    def extract_cost_of(self, urls: list[str]) -> int:
        return 0

    async def extract(self, urls: list[str], **kwargs: Any) -> list[Document]:
        """Fetch and clean each URL concurrently, tolerating individual failures."""
        try:
            import trafilatura  # noqa: F401
        except ImportError:
            return [
                Document(
                    url=url,
                    ok=False,
                    provider=self.name,
                    error="trafilatura not installed: pip install 'searchroute[extract]'",
                )
                for url in urls
            ]

        semaphore = asyncio.Semaphore(int(kwargs.get("concurrency", 5)))

        async def one(url: str) -> Document:
            async with semaphore:
                return await self._fetch_and_clean(url)

        return list(await asyncio.gather(*(one(url) for url in urls)))

    async def _fetch_and_clean(self, url: str) -> Document:
        try:
            response = await self.client.get(
                url,
                timeout=self.timeout,
                follow_redirects=True,
                headers={"User-Agent": self.user_agent, "Accept": "text/html,*/*"},
            )
        except httpx.HTTPError as exc:
            return Document(url=url, ok=False, provider=self.name, error=f"fetch failed: {exc}")

        if response.status_code >= 400:
            return Document(
                url=url, ok=False, provider=self.name, error=f"http {response.status_code}"
            )

        content_type = response.headers.get("content-type", "").split(";")[0].strip()
        if content_type and not content_type.startswith(_HTML_TYPES):
            # A PDF or an image is not something we can clean. Say so plainly
            # instead of handing back mojibake.
            return Document(
                url=url,
                ok=False,
                provider=self.name,
                error=f"unsupported content type: {content_type}",
            )

        # trafilatura is synchronous and CPU-bound; keep it off the event loop.
        extracted = await asyncio.to_thread(self._clean, response.text, url)
        if not extracted:
            return Document(
                url=url,
                ok=False,
                provider=self.name,
                error="no extractable content (likely JavaScript-rendered)",
            )

        return Document(
            url=str(response.url),
            title=extracted.get("title") or "",
            content=extracted.get("text"),
            ok=True,
            provider=self.name,
        )

    @staticmethod
    def _clean(html: str, url: str) -> dict[str, Any] | None:
        import trafilatura

        text = trafilatura.extract(
            html,
            url=url,
            output_format="markdown",
            include_links=True,
            include_tables=True,
            favor_precision=True,
        )
        if not text or not text.strip():
            return None

        title = ""
        try:
            metadata = trafilatura.extract_metadata(html)
            if metadata is not None:
                title = metadata.title or ""
        except Exception:
            # Metadata is a nicety; never lose the body over it.
            pass
        return {"text": text, "title": title}
