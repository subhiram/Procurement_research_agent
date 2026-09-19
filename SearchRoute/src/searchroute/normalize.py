"""URL canonicalization, dedup, and rank fusion.

Two providers returning "the same" page will disagree about tracking params,
trailing slashes, ``www.`` and scheme. Canonicalizing before dedup is what stops
a fan-out search from handing a research agent the same article three times.
"""

from __future__ import annotations

from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from .types import SearchResult

TRACKING_PREFIXES = ("utm_", "pk_", "mc_", "hsa_", "_hs")
TRACKING_PARAMS = frozenset(
    {
        "fbclid", "gclid", "dclid", "msclkid", "yclid", "igshid", "mkt_tok",
        "ref", "referrer", "source", "spm", "cmpid", "campaign_id",
        "__hstc", "__hssc", "__hsfp", "vero_id", "trk", "trkCampaign",
    }
)


def canonicalize(url: str) -> str:
    """A stable identity for a URL, used as the dedup key.

    Deliberately conservative: we strip only things that are known-cosmetic.
    Query params that select content (``?id=``, ``?page=``) are load-bearing and
    are kept, because dropping them would merge genuinely different pages.
    """
    if not url:
        return ""
    raw = url.strip()
    if "://" not in raw:
        raw = f"https://{raw}"

    parts = urlsplit(raw)
    scheme = "https" if parts.scheme in ("http", "https", "") else parts.scheme

    host = parts.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    # Drop default ports.
    if host.endswith(":443") or host.endswith(":80"):
        host = host.rsplit(":", 1)[0]

    path = parts.path or "/"
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")
    if not path:
        path = "/"

    kept = [
        (k, v)
        for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if not (k.lower() in TRACKING_PARAMS or k.lower().startswith(TRACKING_PREFIXES))
    ]
    kept.sort()  # param order is not meaningful for identity

    # Fragments are never content-bearing for our purposes.
    return urlunsplit((scheme, host, path, urlencode(kept), ""))


def dedupe(results: list[SearchResult]) -> list[SearchResult]:
    """Keep the first occurrence of each canonical URL, but merge in richer
    fields from later duplicates.

    Order is preserved: whichever provider ranked it first wins the position,
    while a later duplicate that happens to carry content still contributes it.
    """
    seen: dict[str, SearchResult] = {}
    ordered: list[SearchResult] = []
    for result in results:
        key = canonicalize(result.url)
        if not key:
            continue
        existing = seen.get(key)
        if existing is None:
            seen[key] = result
            ordered.append(result)
            continue
        # Prefer the richer of the two for each field.
        if not existing.content and result.content:
            existing.content = result.content
            existing.content_status = result.content_status
            existing.content_provider = result.content_provider
        if not existing.summary and result.summary:
            existing.summary = result.summary
        if not existing.snippet and result.snippet:
            existing.snippet = result.snippet
        if not existing.title and result.title:
            existing.title = result.title
        if existing.published_date is None and result.published_date is not None:
            existing.published_date = result.published_date
    return ordered


def reciprocal_rank_fusion(
    result_lists: list[list[SearchResult]], k: int = 60
) -> list[SearchResult]:
    """Merge several providers' rankings into one.

    RRF is the right default here because provider scores are not comparable —
    Exa's neural relevance and Google's rank mean different things, so we fuse on
    rank position rather than pretending the scores share a scale. A URL that
    several providers rank highly rises above one that a single provider loved.
    """
    scores: dict[str, float] = {}
    best: dict[str, SearchResult] = {}
    contributors: dict[str, set[str]] = {}

    for results in result_lists:
        for rank, result in enumerate(results, start=1):
            key = canonicalize(result.url)
            if not key:
                continue
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            contributors.setdefault(key, set()).add(result.provider)
            current = best.get(key)
            if current is None:
                best[key] = result
            else:
                # Merge richer fields so fusion never loses content.
                if not current.content and result.content:
                    current.content = result.content
                    current.content_status = result.content_status
                    current.content_provider = result.content_provider
                if not current.summary and result.summary:
                    current.summary = result.summary
                if not current.snippet and result.snippet:
                    current.snippet = result.snippet

    fused = sorted(best.items(), key=lambda kv: scores[kv[0]], reverse=True)
    out: list[SearchResult] = []
    for position, (key, result) in enumerate(fused):
        result.rank = position
        result.score = scores[key]
        if len(contributors[key]) > 1:
            result.raw.setdefault("_searchroute", {})["found_by"] = sorted(contributors[key])
        out.append(result)
    return out


def apply_domain_filters(
    results: list[SearchResult],
    include: list[str] | None = None,
    exclude: list[str] | None = None,
) -> list[SearchResult]:
    """Enforce domain filters client-side.

    Necessary because provider support is uneven — Exa and Tavily take domain
    filters natively, DuckDuckGo and most SERP APIs do not. Applying them here
    too means the caller gets the same semantics whichever provider served the
    query.
    """
    if not include and not exclude:
        return results

    def host_of(url: str) -> str:
        return urlsplit(canonicalize(url)).netloc

    def matches(host: str, domain: str) -> bool:
        domain = domain.lower().lstrip(".")
        if domain.startswith("www."):
            domain = domain[4:]
        return host == domain or host.endswith("." + domain)

    out = []
    for result in results:
        host = host_of(result.url)
        if include and not any(matches(host, d) for d in include):
            continue
        if exclude and any(matches(host, d) for d in exclude):
            continue
        out.append(result)
    return out
