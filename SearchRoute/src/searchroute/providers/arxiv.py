"""arXiv — preprints in CS, physics, maths and related fields.

Keyless, unmetered, authoritative for its domain. Declares ``ACADEMIC`` only, so
it never appears in a general search.

Two things make it unusual among the providers here:

* **It returns Atom XML, not JSON.** Parsed with stdlib ``xml.etree`` — no new
  dependency, and no vendor SDK.
* **It asks for one request every 3 seconds.** That is a *rate* limit, which the
  quota ledger cannot express, so it is enforced by ``min_interval`` and the
  engine's rate gate. Being free and unmetered does not make bursts acceptable.

API: https://info.arxiv.org/help/api/user-manual.html
"""

from __future__ import annotations

from typing import Any
from xml.etree import ElementTree

from ..errors import ProviderError
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

_ATOM = "{http://www.w3.org/2005/Atom}"
_ARXIV = "{http://arxiv.org/schemas/atom}"


class ArxivProvider(Provider):
    name = "arxiv"
    capabilities = frozenset({Capability.ACADEMIC})
    native_depths = frozenset({Depth.LINKS, Depth.SNIPPETS})
    """The abstract is the snippet. It is author-written, not a generated
    summary, so this deliberately does not claim SUMMARY depth."""
    requires_key = False
    quota = None
    quality_hint = 0.9
    default_priority = 12
    min_interval = 3.0
    """arXiv's stated limit: one request every 3 seconds."""
    base_url = "https://export.arxiv.org/api/query"
    """HTTPS, not the http:// URL the arXiv docs still show — that 301s, and we
    don't follow redirects by default."""
    timeout = 30.0

    def cost_of(self, query: SearchQuery) -> int:
        return 0

    def __init__(self, *args, contact: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.contact = contact

    def _headers(self) -> dict[str, str]:
        agent = "SearchRoute/0.1.0 (+https://github.com/subhiram/searchroute)"
        if self.contact:
            agent = f"SearchRoute/0.1.0 (+https://github.com/subhiram/searchroute; {self.contact})"
        return {"User-Agent": agent}

    async def search(self, query: SearchQuery) -> SearchResponse:
        params: dict[str, Any] = {
            "search_query": f"all:{query.query}",
            "max_results": min(query.max_results, 100),
            "sortBy": "relevance",
        }
        params.update(query.params_for(self.name))

        response = await self._request(
            "GET", self.base_url, params=params, headers=self._headers()
        )

        try:
            root = ElementTree.fromstring(response.text)
        except ElementTree.ParseError as exc:
            raise ProviderError(self.name, f"malformed Atom response: {exc}") from exc

        results = []
        for rank, entry in enumerate(root.findall(f"{_ATOM}entry")):
            parsed = _parse_entry(entry)
            if not parsed["url"]:
                continue
            results.append(
                SearchResult(
                    url=parsed["url"],
                    title=parsed["title"],
                    snippet=parsed["summary"],
                    published_date=parse_date(parsed["published"]),
                    provider=self.name,
                    rank=rank,
                    # Domain metadata lives in raw rather than becoming
                    # first-class fields nine other providers would leave empty.
                    raw=parsed,
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


def _text(node, path: str) -> str:
    found = node.find(path)
    return (found.text or "").strip() if found is not None else ""


def _parse_entry(entry) -> dict[str, Any]:
    """One Atom <entry> into a plain dict."""
    pdf_url = ""
    for link in entry.findall(f"{_ATOM}link"):
        if link.get("title") == "pdf":
            pdf_url = link.get("href", "")
            break

    return {
        "url": _text(entry, f"{_ATOM}id"),
        "title": " ".join(_text(entry, f"{_ATOM}title").split()),
        "summary": " ".join(_text(entry, f"{_ATOM}summary").split()),
        "published": _text(entry, f"{_ATOM}published"),
        "updated": _text(entry, f"{_ATOM}updated"),
        "authors": [
            _text(a, f"{_ATOM}name") for a in entry.findall(f"{_ATOM}author")
        ],
        "categories": [
            c.get("term", "") for c in entry.findall(f"{_ATOM}category")
        ],
        "primary_category": (
            entry.find(f"{_ARXIV}primary_category").get("term", "")
            if entry.find(f"{_ARXIV}primary_category") is not None
            else ""
        ),
        "doi": _text(entry, f"{_ARXIV}doi"),
        "comment": _text(entry, f"{_ARXIV}comment"),
        "journal_ref": _text(entry, f"{_ARXIV}journal_ref"),
        "pdf_url": pdf_url,
    }
