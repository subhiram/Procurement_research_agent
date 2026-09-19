"""The hydration stage: filling a depth gap with a second-hop extract.

When the router lands on a provider that cannot serve the requested depth in one
call — Google CSE can only ever return snippets — we do not fail and we do not
silently return less than was asked for. We compose: search on one provider,
extract on another, and label every result with how its content was obtained.

Three rules shape this code:

* **Partial failure is normal.** A paywalled URL keeps its snippet and is marked
  FAILED; the rest still come back full. Hydration never raises for one URL.
* **It is budgeted.** Extraction spends real credits, so ``max_hydrate`` caps how
  many URLs get fetched and the rest are honestly marked SKIPPED.
* **Nothing is invented.** If SUMMARY was requested and no provider generates
  summaries, the result degrades and says so rather than being synthesized.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from ..types import ContentStatus, Depth, SearchResponse, SearchResult

logger = logging.getLogger("searchroute.hydrate")


@dataclass(slots=True)
class HydrationPlan:
    """What hydration will do, decided before any call is made."""

    needed: bool
    to_fetch: list[SearchResult]
    to_skip: list[SearchResult]
    reason: str = ""


def plan(
    response: SearchResponse, requested: Depth, max_hydrate: int
) -> HydrationPlan:
    """Decide which results need a content fetch.

    Only CONTENT is hydratable. SUMMARY deliberately is not: a summary is
    generated text, and generating it locally would mean shipping an LLM
    dependency and putting words in a source's mouth. If no provider supplied
    one, the honest outcome is a degraded response.
    """
    if requested < Depth.CONTENT:
        return HydrationPlan(needed=False, to_fetch=[], to_skip=[], reason="depth below content")

    missing = [r for r in response.results if not r.content]
    if not missing:
        return HydrationPlan(needed=False, to_fetch=[], to_skip=[], reason="already native")

    if max_hydrate <= 0:
        return HydrationPlan(
            needed=False, to_fetch=[], to_skip=missing, reason="max_hydrate is 0"
        )

    return HydrationPlan(
        needed=True,
        to_fetch=missing[:max_hydrate],
        to_skip=missing[max_hydrate:],
        reason="depth gap",
    )


async def hydrate(
    response: SearchResponse,
    requested: Depth,
    *,
    extractor,
    max_hydrate: int = 10,
    concurrency: int = 5,
    batch_size: int = 10,
    only: list[str] | None = None,
) -> SearchResponse:
    """Fill in ``content`` for results that lack it, then label the outcome.

    ``extractor`` is any coroutine ``(urls, only=...) -> list[Document]`` — in
    practice ``Engine.extract``, so hydration inherits the same provider
    fallback, quota accounting and circuit breakers as everything else.
    """
    hydration = plan(response, requested, max_hydrate)

    for result in hydration.to_skip:
        result.content_status = ContentStatus.SKIPPED

    if not hydration.needed:
        return _finalize(response, requested)

    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def fetch(batch: list[SearchResult]) -> None:
        async with semaphore:
            urls = [r.url for r in batch]
            try:
                docs = await extractor(urls, only=only)
            except Exception as exc:  # noqa: BLE001
                # The whole extract chain failed. That degrades the response;
                # it does not lose the search results we already have.
                #
                # Surface it loudly: "content was requested but no extractor is
                # available" is indistinguishable from "every page was blocked"
                # if the reason is only buried in each result's raw payload.
                message = str(exc)
                for result in batch:
                    result.content_status = ContentStatus.FAILED
                    result.raw.setdefault("_searchroute", {})["hydration_error"] = message
                note = f"content could not be fetched: {message}"
                if note not in response.notes:
                    response.notes.append(note)
                    if "no provider configured for extract" in message:
                        response.notes.append(
                            "no EXTRACT-capable provider is configured — pinning "
                            "providers=[...] restricts extraction too. Add an extractor "
                            "(firecrawl, tavily, exa, jina, http_extract) to the client, "
                            "or pass hydrate_providers=[...]."
                        )
                    logger.warning("hydration failed for %d url(s): %s", len(batch), message)
                return

            by_url = {doc.url: doc for doc in docs}
            for result in batch:
                doc = by_url.get(result.url)
                if doc is not None and doc.ok and doc.content:
                    result.content = doc.content
                    result.content_provider = doc.provider
                    result.content_status = ContentStatus.HYDRATED
                    if not result.title and doc.title:
                        result.title = doc.title
                else:
                    result.content_status = ContentStatus.FAILED
                    if doc is not None and doc.error:
                        result.raw.setdefault("_searchroute", {})["hydration_error"] = doc.error

    # Batch generously and bound parallelism with the semaphore instead.
    #
    # Batch size is a *billing* decision, not a concurrency one: providers price
    # extraction per call or per N URLs (Tavily bills per 5), so splitting 3 URLs
    # into 3 single-URL calls would triple the cost for no benefit. One call with
    # all three is both cheaper and faster.
    batch_size = max(1, batch_size)
    batches = [
        hydration.to_fetch[i : i + batch_size]
        for i in range(0, len(hydration.to_fetch), batch_size)
    ]
    await asyncio.gather(*(fetch(b) for b in batches))

    return _finalize(response, requested)


def _finalize(response: SearchResponse, requested: Depth) -> SearchResponse:
    """Set the response-level depth verdict from the per-result statuses."""
    response.requested_depth = requested

    if requested >= Depth.CONTENT:
        got_content = [r for r in response.results if r.content]
        response.degraded = len(got_content) < len(response.results)
        response.depth = Depth.CONTENT if got_content else Depth.SNIPPETS
    elif requested is Depth.SUMMARY:
        got_summary = [r for r in response.results if r.summary]
        response.degraded = len(got_summary) < len(response.results)
        response.depth = Depth.SUMMARY if got_summary else Depth.SNIPPETS
    else:
        response.depth = requested
        response.degraded = False

    # Anything still unlabelled at content depth was never attempted.
    if requested >= Depth.CONTENT:
        for result in response.results:
            if result.content and result.content_status is ContentStatus.NOT_REQUESTED:
                result.content_status = ContentStatus.NATIVE
    return response
