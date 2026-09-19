"""PubMed — biomedical literature via NCBI E-utilities.

Keyless and unmetered, ``ACADEMIC`` only.

**A search costs two HTTP calls**, not one: ``esearch`` returns PMIDs, then
``esummary`` turns those into metadata. Both count against NCBI's rate limit, so
``min_interval`` is set for the pair rather than per call.

NCBI asks for at most 3 requests/second without an API key (10 with one), plus a
``tool`` and ``email`` on every request. Both are sent.

API: https://www.ncbi.nlm.nih.gov/books/NBK25501/
"""

from __future__ import annotations

from typing import Any

from ..types import (
    Capability,
    Depth,
    SearchQuery,
    SearchResponse,
    SearchResult,
    Usage,
)
from ._util import parse_date
from .base import Provider


class PubMedProvider(Provider):
    name = "pubmed"
    capabilities = frozenset({Capability.ACADEMIC})
    native_depths = frozenset({Depth.LINKS, Depth.SNIPPETS})
    requires_key = False
    quota = None
    quality_hint = 0.85
    default_priority = 14
    min_interval = 0.7
    """NCBI allows ~3 req/sec keyless. A search is two calls, so pacing the
    *search* at 0.7s keeps the pair comfortably inside that."""
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    timeout = 30.0

    def __init__(self, *args, contact: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.contact = contact

    def cost_of(self, query: SearchQuery) -> int:
        return 0

    def _common(self) -> dict[str, Any]:
        params: dict[str, Any] = {"tool": "searchroute", "retmode": "json"}
        if self.contact:
            params["email"] = self.contact
        if self.api_key:
            params["api_key"] = self.api_key
        return params

    async def search(self, query: SearchQuery) -> SearchResponse:
        # Step 1: query -> PMIDs
        search_params = {
            **self._common(),
            "db": "pubmed",
            "term": query.query,
            "retmax": min(query.max_results, 100),
            "sort": "relevance",
        }
        search_params.update(query.params_for(self.name))
        response = await self._request(
            "GET", f"{self.base_url}/esearch.fcgi", params=search_params
        )
        ids = response.json().get("esearchresult", {}).get("idlist", []) or []
        if not ids:
            return SearchResponse(
                query=query.query,
                results=[],
                depth=Depth.SNIPPETS,
                requested_depth=query.depth,
                providers_used=[self.name],
                cost=Usage(),
            )

        # Step 2: PMIDs -> metadata
        summary = await self._request(
            "GET",
            f"{self.base_url}/esummary.fcgi",
            params={**self._common(), "db": "pubmed", "id": ",".join(ids)},
        )
        payload = summary.json().get("result", {}) or {}

        results = []
        # `payload["uids"]` preserves relevance order; iterating the dict would not.
        for rank, pmid in enumerate(payload.get("uids", ids)):
            item = payload.get(pmid)
            if not isinstance(item, dict):
                continue
            authors = [a.get("name", "") for a in item.get("authors", []) or []]
            journal = item.get("fulljournalname") or item.get("source") or ""
            byline = ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "")
            results.append(
                SearchResult(
                    url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    title=item.get("title", "").rstrip("."),
                    # PubMed's esummary has no abstract; the byline plus journal
                    # is the honest snippet. Fetching abstracts would need a
                    # third call per search.
                    snippet=" — ".join(x for x in (byline, journal) if x) or None,
                    published_date=parse_date(item.get("pubdate")),
                    provider=self.name,
                    rank=rank,
                    raw={**item, "pmid": pmid, "authors_flat": authors},
                )
            )

        return SearchResponse(
            query=query.query,
            results=results,
            depth=Depth.SNIPPETS,
            requested_depth=query.depth,
            providers_used=[self.name],
            cost=Usage(),
        )
