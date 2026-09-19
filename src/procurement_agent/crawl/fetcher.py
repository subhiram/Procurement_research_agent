"""Local page fetching with crawl4ai — the free tier of the fetch ladder.

Every page this retrieves is a Tavily or Firecrawl credit not spent, which is
what makes following contact pages affordable at all: on a 1,000-credit/month
free tier, paying to fetch three extra pages per vendor was never going to work.

Two behaviours of crawl4ai are load-bearing here and were both confirmed
against real vendor sites:

- **`result.success` cannot be trusted.** A bot-blocked page came back with
  `success=True`, `status_code=418` and 92 characters of markdown. Success is
  therefore judged on *content length*, reusing the same `MIN_USEFUL_CONTENT`
  threshold the search router already uses to decide a result is too thin.
- **A non-200 status does not mean failure.** Two sites returned 301 with
  complete, usable content. Status is recorded but never gates the result.

Not every site can be crawled. `rockwellind.com` blocks the headless browser
exactly as it blocks plain httpx, while Tavily retrieves it fine — which is the
whole reason the paid tier stays in place behind this one.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from procurement_agent.config import Settings, get_settings

log = logging.getLogger(__name__)

#: Matches the search router's threshold for "too thin to be worth anything".
MIN_USEFUL_CONTENT = 400


@dataclass
class CrawledPage:
    """One fetched page, plus the links needed to find its contact page."""

    url: str
    content: str = ""
    links: list[dict] = field(default_factory=list)
    status_code: int | None = None
    failed: bool = False
    error: str | None = None

    @property
    def is_useful(self) -> bool:
        return not self.failed and len(self.content) >= MIN_USEFUL_CONTENT


class Crawl4AIFetcher:
    """Shared browser-backed fetcher.

    Browser startup dominates the cost of a single fetch, so one crawler is
    started lazily and reused for the life of the process. `aclose()` shuts it
    down; the FastAPI lifespan and the CLI both call it.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._crawler = None
        self._start_lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(self._settings.crawl_concurrency)
        self._failed_to_start = False

    def is_enabled(self) -> bool:
        return self._settings.enable_crawl4ai and not self._failed_to_start

    async def _get_crawler(self):
        """Start the browser once, on first use."""
        if self._crawler is not None:
            return self._crawler

        async with self._start_lock:
            if self._crawler is not None:
                return self._crawler
            from crawl4ai import AsyncWebCrawler, BrowserConfig

            crawler = AsyncWebCrawler(
                config=BrowserConfig(
                    headless=True,
                    verbose=False,
                    user_agent=self._settings.crawl_user_agent,
                )
            )
            await crawler.start()
            self._crawler = crawler
            log.info("crawl4ai: browser started")
            return self._crawler

    def _run_config(self):
        from crawl4ai import CacheMode, CrawlerRunConfig

        return CrawlerRunConfig(
            # crawl4ai prints a progress banner per page to the console by
            # default, which drowns the agent's own logs on a 12-vendor run.
            verbose=False,
            log_console=False,
            page_timeout=self._settings.crawl_timeout * 1000,
            # These are other companies' websites: identify ourselves, obey
            # their robots.txt, and keep concurrency low.
            check_robots_txt=self._settings.crawl_respect_robots,
            cache_mode=CacheMode.BYPASS,
            exclude_external_links=True,
            exclude_all_images=True,
        )

    async def fetch(self, url: str) -> CrawledPage:
        """Fetch one page. Never raises — a failure is a `CrawledPage`."""
        if not self.is_enabled():
            return CrawledPage(url=url, failed=True, error="crawl4ai disabled")

        try:
            crawler = await self._get_crawler()
        except Exception as exc:  # noqa: BLE001 - fall back to the paid tier
            # A browser that will not start should disable this tier for the
            # rest of the process rather than failing every page slowly.
            self._failed_to_start = True
            log.warning("crawl4ai: browser failed to start, disabling: %s", exc)
            return CrawledPage(url=url, failed=True, error=str(exc))

        try:
            async with self._semaphore:
                result = await crawler.arun(url, config=self._run_config())
        except Exception as exc:  # noqa: BLE001 - one bad page is not fatal
            log.info("crawl4ai: %s failed (%s)", url, exc)
            return CrawledPage(url=url, failed=True, error=str(exc))

        content = _markdown_of(result)
        page = CrawledPage(
            url=url,
            content=content,
            links=list((getattr(result, "links", None) or {}).get("internal", [])),
            status_code=getattr(result, "status_code", None),
        )

        # Judged on content, not on `success` or status: a 418 block reported
        # success with 92 characters, and two good pages reported 301.
        if not page.is_useful:
            page.failed = True
            page.error = f"content too thin ({len(content)} chars)"
            log.info(
                "crawl4ai: %s returned %d chars (status %s) - falling back",
                url,
                len(content),
                page.status_code,
            )

        return page

    async def fetch_many(self, urls: list[str]) -> list[CrawledPage]:
        """Fetch several pages, bounded by the configured concurrency."""
        if not urls:
            return []
        return list(await asyncio.gather(*(self.fetch(u) for u in urls)))

    async def aclose(self) -> None:
        if self._crawler is not None:
            try:
                await self._crawler.close()
            except Exception as exc:  # noqa: BLE001 - shutdown is best-effort
                log.debug("crawl4ai: error closing browser: %s", exc)
            finally:
                self._crawler = None


def _markdown_of(result: object) -> str:
    """crawl4ai returns markdown as either a string or a wrapper object."""
    markdown = getattr(result, "markdown", None)
    if markdown is None:
        return ""
    raw = getattr(markdown, "raw_markdown", None)
    return raw if isinstance(raw, str) else str(markdown)


_fetcher: Crawl4AIFetcher | None = None


def get_fetcher(settings: Settings | None = None) -> Crawl4AIFetcher:
    """Process-wide fetcher, so the browser is started at most once."""
    global _fetcher
    if _fetcher is None:
        _fetcher = Crawl4AIFetcher(settings)
    return _fetcher


async def close_fetcher() -> None:
    global _fetcher
    if _fetcher is not None:
        await _fetcher.aclose()
        _fetcher = None


async def _main() -> int:
    """`python -m procurement_agent.crawl.fetcher <url>` — manual check."""
    import sys

    logging.basicConfig(level="INFO", format="%(levelname)-7s %(name)s %(message)s")
    if len(sys.argv) < 2:
        print("usage: python -m procurement_agent.crawl.fetcher <url>")
        return 2

    url = sys.argv[1]
    fetcher = get_fetcher()
    try:
        page = await fetcher.fetch(url)
    finally:
        await close_fetcher()

    print(f"url:     {page.url}")
    print(f"status:  {page.status_code}")
    print(f"useful:  {page.is_useful} ({len(page.content)} chars)")
    if page.error:
        print(f"error:   {page.error}")
    if page.links:
        from procurement_agent.crawl.contact_links import find_contact_links

        print(f"contact: {find_contact_links(page.links, url)}")
    print("-" * 60)
    print(page.content[:1500])
    return 0 if page.is_useful else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
