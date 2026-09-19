# SearchRoute — Usage Guide

Everything you can do with the library, what each option actually does, and what it costs.

- [1. Setup](#1-setup)
- [2. The mental model](#2-the-mental-model)
- [3. Per-function provider control](#3-per-function-provider-control)
- [4. What you get back](#4-what-you-get-back)
- [5. Depth: how rich each result is](#5-depth-how-rich-each-result-is)
- [6. Strategies: how a provider is chosen](#6-strategies-how-a-provider-is-chosen)
- [7. Profiles](#7-profiles)
- [8. Quota and cost](#8-quota-and-cost)
- [9. Errors and failure handling](#9-errors-and-failure-handling)
- [10. Extract and answer](#10-extract-and-answer)
- [11. Custom providers](#11-custom-providers)
- [12. Observability](#12-observability)
- [13. Recipes](#13-recipes)
- [14. Gotchas](#14-gotchas)
- [15. API reference](#15-api-reference)

---

## 1. Setup

```bash
pip install searchroute            # core: httpx only
pip install 'searchroute[all]'     # + keyless DuckDuckGo search and local extraction
```

Set keys for whichever providers you have. SearchRoute enables exactly those.

```bash
export EXA_API_KEY=...
export TAVILY_API_KEY=...
export FIRECRAWL_API_KEY=...
export GOOGLE_API_KEY=...  GOOGLE_CSE_ID=...
export SERPAPI_API_KEY=... SERPER_API_KEY=... SEARCHAPI_API_KEY=...
export JINA_API_KEY=...            # only needed for Jina *search*; extraction is keyless
export SEARXNG_URL=http://localhost:8080
```

```python
from searchroute import SearchRoute

sr = SearchRoute()                 # auto-discovers from the environment
for hit in sr.search("what is RAG?"):
    print(hit.title, hit.url)
```

**It works with no keys at all**, degraded: DuckDuckGo searches, Jina and local trafilatura
extract. Good for development; don't build a product on it (see [Gotchas](#14-gotchas)).

Check what's live:

```python
>>> from searchroute import discover_providers
>>> discover_providers()
['exa', 'tavily', 'google_cse', 'jina', 'http_extract', 'duckduckgo']
```

---

## 2. The mental model

Two independent axes. Everything else is detail.

| Axis | Question | Controlled by |
|---|---|---|
| **Which provider** | Who answers this query? | `providers=`, `strategy=`, quota, health |
| **How rich** | What comes back per result? | `depth=` |

The same query can return bare links or full page markdown without changing anything about
provider selection — and vice versa.

```python
sr.search("q")                      # snippets, default provider order
sr.search("q", depth="content")     # same routing, full page markdown
sr.search("q", providers=["exa"])   # same depth, pinned provider
```

---

## 3. Per-function provider control

This is the most common real requirement: different parts of your app want different
providers.

### The recommended pattern: one client, pin per call

Build **one** client holding every provider your app will ever use, then pin per call.

```python
from searchroute import SearchRoute

# module-level, built once
search = SearchRoute(providers=["tavily", "exa", "duckduckgo"])

def research(query: str):
    """Tavily first; fall back to Exa if Tavily fails."""
    return search.search(query, providers=["tavily", "exa"], depth="content")

def cheap_lookup(query: str):
    """Explicitly DuckDuckGo, nothing else."""
    return search.search(query, providers=["duckduckgo"], max_results=5)
```

**Order is exactly what you wrote.** `providers=["tavily", "exa"]` tries Tavily first and
only touches Exa if Tavily fails or returns nothing. It does not reorder behind your back.

Sharing one client matters — it shares the quota ledger, the circuit breakers, and the HTTP
connection pool across your whole app. Two clients means two sets of breakers, so a provider
that just failed in `research()` would be retried from scratch in `cheap_lookup()`.

> A provider must be in the client's `providers=[...]` list to be pinnable later.
> Pinning one that isn't raises `NoProviderAvailable: requested provider(s) not configured: firecrawl`.

### ⚠️ Pinning restricts extraction too

`providers=[...]` limits the pool for **extraction as well as search**. A client built only
from search-only providers cannot serve `depth="content"` — every result comes back `FAILED`:

```python
sr = SearchRoute(providers=["duckduckgo"])          # search-only
r  = sr.search("q", depth="content")
r.degraded                                          # True
[h.content_status for h in r]                       # all FAILED
r.notes
# ['content could not be fetched: no provider configured for extract',
#  'no EXTRACT-capable provider is configured — pinning providers=[...] restricts
#   extraction too. Add an extractor (firecrawl, tavily, exa, jina, http_extract)
#   to the client, or pass hydrate_providers=[...].']
```

The fix is to include an extractor. The keyless ones cost nothing:

```python
sr = SearchRoute(providers=["duckduckgo", "jina", "http_extract"])
```

This is explicit on purpose — SearchRoute won't quietly send your URLs to a third-party
extractor you didn't list. **Always check `response.notes`** when a response comes back
degraded; it tells you what to change.

### What "fallback" means here

`providers=["tavily", "exa"]` moves to Exa when Tavily:

- raises any error (auth, quota, rate limit, timeout, 5xx), **or**
- succeeds but returns **zero results**

A success with zero hits isn't an answer, so the chain keeps walking. Every attempt is
recorded on `response.attempts` either way.

### Separate clients, when you actually want isolation

Use separate clients when the *configuration* differs, not just the provider list — e.g. a
background job that should never touch your scarce Google CSE daily quota:

```python
interactive = SearchRoute(providers=["exa", "google_cse"], strategy="quality")
batch       = SearchRoute(providers=["exa"], strategy="quota_aware", reserve_pct=0.30)
```

They still share the on-disk quota ledger by default (same file path), so spend is counted
once across both. Breakers are per-client.

### Excluding instead of pinning

```python
sr.search("q", exclude=["duckduckgo"])   # everything else, in normal order
```

`exclude=` is better than pinning when you want "the usual routing, minus one".

---

## 4. What you get back

Real output from `sr.search("what is a circuit breaker pattern", max_results=3, depth="content")`:

```
SearchResponse
  query            = 'what is a circuit breaker pattern'
  depth            = Depth.CONTENT          # what we actually achieved
  requested_depth  = Depth.CONTENT          # what you asked for
  degraded         = True                   # something fell short
  providers_used   = ['duckduckgo']
  cost             = {}  total=0
  results          = [SearchResult, SearchResult, SearchResult]
  attempts         = [Attempt(duckduckgo, ok=True, 2538ms, n=3, cost=0)]
```

One result:

```
SearchResult
  url               = 'https://en.wikipedia.org/wiki/Circuit_breaker_design_pattern'
  title             = 'Circuit breaker design pattern - Wikipedia'
  snippet           = 'Circuit breaker design pattern.'
  summary           = None                        # provider-generated only
  content           = <19331 chars of markdown>
  content_status    = ContentStatus.HYDRATED      # how content was obtained
  content_provider  = 'jina'                      # who fetched it
  provider          = 'duckduckgo'                # who found it
  score             = None
  published_date    = None
  rank              = 0
  raw               = {'title': ..., 'href': ..., 'body': ...}   # untouched payload
```

Note `provider` and `content_provider` differ: DuckDuckGo found the page, Jina fetched its
text. That composition is the point.

### Convenience access

```python
r = sr.search("q")

len(r)                  # number of results
for hit in r: ...       # iterates results directly
r.urls                  # ['https://...', ...]
r.results[0].text       # richest available: content or summary or snippet
r.cost.total            # total credits spent
r.cost.by_provider      # {'exa': 50}
```

`.text` is the one to feed a model when you don't care how rich it is.

### `raw` is never lost

Anything the provider returned that SearchRoute doesn't model is on `result.raw`. You are
never blocked by the abstraction.

```python
hit.raw.get("favicon")            # Serper-specific field, still there
```

---

## 5. Depth: how rich each result is

```python
sr.search("q", depth="links")      # title + url
sr.search("q", depth="snippets")   # + provider snippet          (default)
sr.search("q", depth="summary")    # + provider-generated summary
sr.search("q", depth="content")    # + full cleaned page markdown
```

Accepts `"content"`, `Depth.CONTENT`, or `3` — you never have to import the enum.

| Depth | Cost | Notes |
|---|---|---|
| `links` | 1 call | Rarely what you want. |
| `snippets` | 1 call | Default. Every provider supports it. |
| `summary` | 1 call, pricier params | Only Exa generates these. Never synthesized locally. |
| `content` | 1 search + K extracts | The expensive one. Budget it. |

### Hydration: how `content` works when a provider can't

Google CSE and DuckDuckGo only ever return snippets. Ask them for `content` and SearchRoute
**composes** instead of failing — it searches on one provider and extracts on another:

```
search(google_cse) → 10 urls
      │ depth gap detected
hydrate: firecrawl → tavily → jina → http_extract   (budgeted, concurrent)
      │
results[].content filled, each labelled
```

Every result tells you how it got its content:

```python
for hit in sr.search("q", depth="content"):
    if hit.content_status == "native":     # search provider returned it inline, no extra call
        ...
    elif hit.content_status == "hydrated": # a second extract call fetched it
        ...
    elif hit.content_status == "failed":   # paywall, 403, JS-rendered — snippet still usable
        ...
    elif hit.content_status == "skipped":  # outside the max_hydrate budget
        ...
```

`failed` means *we could not fetch this page*. That is very different from *this page has no
content* — and if you're citing sources, you need the distinction.

### Budgeting hydration

`max_hydrate` caps how many pages get fetched. It is a **credit control**, not a formatting
one: hydrating 50 URLs against Firecrawl's 1,000/month tier burns 5% of your month on a
single query.

```python
sr.search("q", depth="content", max_hydrate=3)                    # fetch at most 3
sr.search("q", depth="content", max_hydrate=0)                    # never hydrate
sr.search("q", depth="content", hydrate_providers=["jina"])       # pin the extractor
```

Results beyond the budget come back `SKIPPED` with their snippets intact.

### Why `summary` never falls back

If no provider generated a summary, the response degrades to `snippets` and sets
`degraded=True`. The library has no model and will not invent text and attribute it to a
source. If you want summaries, call your own LLM on `hit.text`.

---

## 6. Strategies: how a provider is chosen

```python
SearchRoute(strategy="quota_aware")           # client default
sr.search("q", strategy="fanout", n=3)        # per-call override
```

A per-call override doesn't disturb the client default, and shares the same breakers and
quota ledger.

### `priority` — default

Try providers in the order they were configured. Predictable: the same query hits the same
provider every time, so cost and behaviour are easy to reason about.

```python
SearchRoute(providers=["tavily", "exa", "duckduckgo"])   # exactly this order
```

**Use when:** you know what you want. This is the right default for most apps.
**Cost:** 1 call (plus one per failed fallback).

### `quota_aware`

Ranks by how much headroom each provider has left in its current window. This is what makes
several small free tiers behave like one larger one.

Ordering logic, in order of precedence:

1. **Unmetered providers first** (SearXNG, Jina) — spending nothing is strictly better.
2. **Then by *fraction* of allowance remaining** — fraction, not absolute, so a 100/day tier
   and a 1,000/month tier compete fairly.
3. **One-time grants last** (Serper's 2,500, SearchApi's trial). A monthly tier refills
   whether you used it or not; a one-time grant never comes back. It's the reserve tank.

**Use when:** you're running on free tiers and want them to last the month.
**Cost:** 1 call.

### `quality`

Highest `quality_hint` first, spend be damned. For extract calls it uses `extract_quality`
instead — the best searcher is rarely the best extractor.

Serving your requested depth natively is itself a quality signal, so a provider that returns
content inline is preferred over one that would need a lossy second-hop extraction.

**Use when:** answer quality matters more than credits.
**Cost:** 1 call, often the most expensive provider.

### `latency`

Fastest observed provider first, by exponentially-weighted moving average of real response
times. Unmeasured providers are tried optimistically so they can establish a baseline.

**Use when:** an interactive UI is waiting.
**Cost:** 1 call.

### `round_robin`

Rotates the head of the list on each call, so several tiers drain evenly instead of burning
one to zero before touching the next.

**Use when:** you have many small tiers and a steady query volume.
**Cost:** 1 call.

### `race(n=2)`

Fires the top *n* concurrently and takes the first success. Losers are cancelled.

```python
sr.search("q", strategy="race", n=2)
```

**Use when:** latency is critical and you'll pay for it.
**Cost:** up to *n* calls per query. You pay for the losers.

### `fanout(n=3)`

Queries *n* providers in parallel and **fuses** all their results.

```python
r = sr.search("q", strategy="fanout", n=3)
r.providers_used            # ['exa', 'tavily', 'google_cse']
r.results[0].raw["_searchroute"]["found_by"]   # ['exa', 'tavily'] — who agreed
```

Merging uses **Reciprocal Rank Fusion**, not score averaging, because provider scores aren't
comparable — Exa's neural relevance and Google's rank mean different things. RRF fuses on
rank *position*, so a page several providers rank highly beats one a single provider loved.
Duplicates are collapsed by canonical URL first (`?utm_source=`, `www.`, trailing slashes,
and fragments are all normalized away).

**Use when:** recall matters most — research, RAG corpus building.
**Cost:** *n* calls per query. The most expensive strategy.

### Custom strategy

Any callable `(candidates, ctx) -> ordered candidates`:

```python
def prefer_academic(candidates, ctx):
    return sorted(candidates, key=lambda p: "academic" not in p.capabilities)

sr.search("q", strategy=prefer_academic)
```

`ctx` gives you `ctx.query`, `ctx.capability`, `ctx.quota` (call
`ctx.quota.fraction_remaining(name)`), `ctx.latency`, and `ctx.counter`.

### Summary

| Strategy | Calls/query | Optimizes for |
|---|---|---|
| `priority` | 1 | Predictability |
| `quota_aware` | 1 | Free-tier longevity |
| `quality` | 1 | Result quality |
| `latency` | 1 | Speed |
| `round_robin` | 1 | Even drain |
| `race(n)` | up to n | Speed, at cost |
| `fanout(n)` | n | Recall, at cost |

---

## 7. Profiles

Named bundles of depth + strategy + budgets, so you pick one word instead of five knobs.

```python
SearchRoute(profile="quicklook")
SearchRoute(profile="research")
```

| Profile | Depth | Strategy | Results | Hydrate | For |
|---|---|---|---|---|---|
| `quicklook` | snippets | priority | 5 | 0 | A quick tool call |
| `cheap` | snippets | quota_aware | 10 | 0 | High volume, tight budget |
| `fast` | snippets | race | 10 | 0 | Interactive UI |
| `rag` | content | quota_aware | 10 | 8 | Retrieval pipeline |
| `research` | content | fanout | 25 | 15 | Deep research |

Everything stays overridable — explicit kwargs beat the profile, and per-call args beat both:

```python
sr = SearchRoute(profile="research", max_results=10)   # research, but 10 not 25
sr.search("q", depth="snippets")                       # this one call is cheap
```

---

## 8. Quota and cost

SearchRoute keeps a local ledger of what each provider spent in its current reset window, so
`quota_aware` can prefer whoever has room instead of discovering exhaustion by getting a 402.

```python
>>> sr.status()
{'providers': ['exa', 'tavily', 'duckduckgo'],
 'strategy': 'quota_aware',
 'quota': {'exa':    {'limit': 10000, 'unit': 'credits', 'remaining': 9450,
                      'resets_at': '2026-10-01T00:00:00+00:00', 'disabled': False},
           'tavily': {'limit': 1000, 'unit': 'credits', 'remaining': 812, ...}},
 'breakers': {'duckduckgo': {'state': 'closed', 'failures': 0, 'latency_ms': 3039.6}}}
```

Per-query spend is on the response:

```python
r = sr.search("q", depth="content")
r.cost.by_provider     # {'exa': 150}
r.cost.total           # 150
```

### Where the ledger lives

A lock-protected JSON file at `$XDG_STATE_HOME/searchroute/quota.json`, so several worker
processes on one machine share a single count.

```python
SearchRoute(quota_store="memory")                 # in-process, resets on restart (tests)
SearchRoute(quota_store="~/.myapp/quota.json")    # custom path
```

### It's advisory, not authoritative

The ledger can drift — another process, another machine, a call it never saw. A real quota
error from the provider always wins and immediately syncs the local counter to drained. Don't
treat `remaining` as billing truth; it's a routing hint.

### `reserve_pct`

Holds a sliver of every tier back (default 5%) so a background job can't drain the last
credit an interactive call needed.

```python
SearchRoute(reserve_pct=0.30)   # keep 30% in reserve
SearchRoute(reserve_pct=0.0)    # spend to the last credit
```

### Reset windows

Daily tiers (Google CSE) reset at UTC midnight. Monthly tiers reset on the 1st by default; if
you signed up mid-month, tell it so the window is right:

```python
from searchroute import QuotaPolicy, Period, Anchor
# Tavily billing anchored to the 15th
QuotaPolicy(limit=1000, period=Period.MONTHLY, anchor=Anchor.SIGNUP_DAY, anchor_day=15)
```

---

## 9. Errors and failure handling

The library distinguishes failure kinds because they need different recovery:

| Error | Meaning | What the router does |
|---|---|---|
| `AuthError` | Bad/missing key | **Trips the breaker immediately.** A wrong key won't fix itself. |
| `QuotaExceeded` | Tier drained | Marks it drained until the window resets. Skips it. |
| `RateLimited` | Too fast | Backs off with jitter, retries once, then cools off 60s. |
| `TransientError` | Timeout, 5xx | Retries with backoff, then moves on. |
| `ProviderError` | Everything else | Moves on. |

Auth failures are loud on purpose — a typo'd key silently masked by a fallback is how you
discover months later that you've been running on DuckDuckGo.

### When everything fails

```python
from searchroute import NoProviderAvailable

try:
    r = sr.search("q")
except NoProviderAvailable as exc:
    for a in exc.attempts:
        print(a.provider, a.error_kind, a.error)
```

The message also explains *why* each provider was ruled out, so it's debuggable:

```
no provider available for this call — exa: quota exhausted (remaining=0);
tavily: circuit open after repeated failures; jina: search needs an API key
(other capabilities work without one)
```

### Degraded success is not failure

A search that returns results but couldn't fetch all the content sets `degraded=True` and
still returns. Check it rather than assuming:

```python
r = sr.search("q", depth="content")
if r.degraded:
    log.warning("only %d/%d hydrated", sum(1 for h in r if h.content), len(r))
```

### Circuit breakers

After 3 consecutive infrastructure failures a provider is skipped entirely for 60s, then one
probe is allowed through. Quota exhaustion does **not** trip a breaker — that's the ledger's
job, and it resets on a schedule.

```python
sr.status()["breakers"]   # {'exa': {'state': 'open', 'failures': 3, ...}}
```

---

## 10. Extract and answer

### `extract` — fetch and clean pages

```python
docs = sr.extract(["https://example.com/a", "https://example.com/b"])
for d in docs:
    if d.ok:
        print(d.url, d.title, len(d.content))
    else:
        print(d.url, "failed:", d.error)
```

Falls back **per URL**, not per batch — a paywalled page doesn't cost the others their
content. Order always matches the URLs you passed, and no URL is silently dropped.

The extract chain is ranked independently by `extract_quality`:
Firecrawl (0.95, renders JS) → Tavily (0.8) / Jina (0.8) → Exa (0.75) → local trafilatura (0.35).

```python
sr.extract(urls, providers=["jina"])       # pin the extractor
```

### `answer` — a provider-generated direct answer

```python
a = sr.answer("who acquired Deepmind and when?")
print(a.answer)      # str, or None
print(a.results)     # the citations it was grounded in
```

Only Exa and Tavily generate answers. **If none is configured, `answer` is `None` and you get
plain search results instead** — the library will not write one. That's deliberate: it has no
model. Always check for `None`.

---

## 11. Custom providers

Your internal search endpoint becomes a first-class provider — routed, quota-tracked, and
failed-over exactly like a built-in.

```python
from searchroute import Provider, Capability, Depth, SearchResponse, SearchResult, register

@register
class CompanyDocs(Provider):
    name = "company_docs"
    capabilities = frozenset({Capability.SEARCH})
    native_depths = frozenset({Depth.LINKS, Depth.SNIPPETS, Depth.CONTENT})
    requires_key = False
    quota = None                 # unmetered
    quality_hint = 0.9           # we trust our own index
    default_priority = 1         # try it first

    async def search(self, query):
        rows = await my_backend.search(query.query, limit=query.max_results)
        return SearchResponse(
            query=query.query,
            results=[
                SearchResult(
                    url=row["url"], title=row["title"], snippet=row["excerpt"],
                    content=row["body"] if query.depth >= Depth.CONTENT else None,
                    provider=self.name, rank=i, raw=row,
                )
                for i, row in enumerate(rows)
            ],
            providers_used=[self.name],
        )

sr = SearchRoute(providers=["company_docs", "exa"])
```

Or pass an instance without registering:

```python
sr = SearchRoute(custom_providers=[CompanyDocs()])
```

Declare `native_depths` honestly. If you say you serve `CONTENT` and return `None`, results
are marked `FAILED` rather than hydrated from elsewhere.

---

## 12. Observability

Hooks fire on every attempt. A hook that raises can never break the search it observes.

```python
def log_attempts(event, payload):
    # events: "success", "failure", "extract_failure"
    print(event, payload)

sr = SearchRoute(hooks=[log_attempts])
```

Per-call audit trail:

```python
for a in r.attempts:
    print(a.provider, a.ok, f"{a.latency_ms:.0f}ms", a.n_results, a.cost, a.error_kind)
```

---

## 13. Recipes

**Different providers per function** — see [§3](#3-per-function-provider-control).

**Research pipeline feeding an LLM**

```python
sr = SearchRoute(profile="research")

def gather(question: str) -> list[dict]:
    r = sr.search(question)
    return [
        {"url": h.url, "title": h.title, "text": h.text}
        for h in r if h.content              # only fully-fetched pages
    ]
```

**Domain-scoped search**

```python
sr.search("transformer scaling laws", include_domains=["arxiv.org"])
sr.search("news", exclude_domains=["pinterest.com", "quora.com"])
```

Applied client-side too, so semantics are identical whichever provider served the query.

**Academic / news routing**

```python
from searchroute import Capability
sr.search("attention is all you need", capability=Capability.ACADEMIC)
sr.search("openai funding", capability=Capability.NEWS)
```

**Date filtering**

```python
from datetime import datetime, timezone
sr.search("ai policy", start_date=datetime(2026, 1, 1, tzinfo=timezone.utc))
```

**Provider-specific parameters** — full escape hatch, keyed by provider name:

```python
sr.search("q", extra={"exa": {"type": "neural"}, "tavily": {"topic": "finance"}})
```

**Cheap first, expensive only if needed**

```python
r = sr.search(q, providers=["google_cse"], depth="snippets")
if len(r) < 3:
    r = sr.search(q, providers=["exa"], depth="content")
```

**Async**

```python
from searchroute import AsyncSearchRoute

async with AsyncSearchRoute(profile="rag") as sr:
    r = await sr.search("q")
    docs = await sr.extract(r.urls[:5])
```

`AsyncSearchRoute` is the real implementation. The sync `SearchRoute` is a facade running it
on a background loop, so it also works inside notebooks and web handlers where
`asyncio.run()` would fail.

---

## 14. Gotchas

**`n=` only applies with an explicit `strategy=`.**
`sr.search(q, n=3)` alone does nothing. Write `sr.search(q, strategy="fanout", n=3)`.

**Pin only what the client holds.** A per-call `providers=["firecrawl"]` fails unless the
client was built with Firecrawl. Build the client with the union of everything you'll pin.

**`depth="content"` is never implicit.** It spends real credits and pulls whole pages. The
default stays `snippets`.

**`max_hydrate` is about credits, not formatting.** Raising it multiplies extraction spend.

**DuckDuckGo is a safety net, not a tier.** `ddgs` is unofficial, trips bot detection well
under 30 requests/minute from one IP, and describes itself as educational-use-only. It's
forced to the back of every ordering. Don't plan capacity around it.

**Brave needs explicit opt-in.** It killed its free tier and now bills past a small credit,
so `BRAVE_API_KEY` alone won't enable it — you must name it in `providers=[...]`. A key in
your environment isn't consent to spend from it.

**Jina's two endpoints differ.** `r.jina.ai` (reading) is keyless; `s.jina.ai` (search)
returns 401 without a key. Keyless Jina is offered for *extraction only*. Add `JINA_API_KEY`
to use it for search too.

**Local extraction can't render JavaScript.** `http_extract` fetches HTML and runs a
readability pass. A client-rendered page comes back `FAILED`, not as empty content — an agent
citing an empty page is worse than one told the page failed. Use Firecrawl for JS-heavy sites.

**Set a contact User-Agent if you extract at volume.** The default identifies itself honestly
(which measurably works better — Wikipedia returns 403 to a spoofed Chrome UA and 200 to a
descriptive one), but several bot policies want contact details:

```python
SearchRoute(provider_options={
    "http_extract": {"user_agent": "MyApp/1.0 (+https://example.com; me@example.com)"}
})
```

**`answer()` can legitimately return `None`.** Check it.

**No LLM anywhere.** SearchRoute never calls a model. No query planning, no summarization, no
semantic reranking, no chunking. It returns faithful cleaned markdown; orchestration is yours.

---

## 15. API reference

### Constructor

```python
SearchRoute(
    providers=None,          # list[str] — explicit order. None = auto-discover
    profile=None,            # "quicklook" | "cheap" | "fast" | "rag" | "research"
    strategy=None,           # name, instance, or callable
    depth=None,              # "links" | "snippets" | "summary" | "content"
    max_results=10,
    max_hydrate=10,
    exclude=None,            # list[str]
    api_keys=None,           # {"exa": "..."} — overrides the environment
    provider_options=None,   # {"google_cse": {"cx": "..."}}
    quota_store=None,        # path | "memory" | StateStore instance
    reserve_pct=0.05,
    timeout=20.0,
    max_retries=1,
    hooks=None,              # [callable(event, payload)]
    custom_providers=None,   # [Provider instance]
    http_client=None,        # share an httpx.AsyncClient
)
```

### `search`

```python
sr.search(
    query,
    depth=None, max_results=None,
    providers=None, exclude=None,
    strategy=None, n=None,
    capability=Capability.SEARCH,
    max_hydrate=None, hydrate_providers=None,
    include_domains=None, exclude_domains=None,
    start_date=None, end_date=None,
    lang=None, region=None,
    extra=None,
) -> SearchResponse
```

### `extract` / `answer`

```python
sr.extract(urls, providers=None, exclude=None) -> list[Document]
sr.answer(query, providers=None, **kwargs)     -> Answer
```

### Introspection

```python
sr.providers      # list[str] of configured provider names
sr.settings       # resolved Settings
sr.status()       # {"providers", "strategy", "quota", "breakers"}
sr.close()        # or use as a context manager
```

### Types

```python
from searchroute import (
    SearchRoute, AsyncSearchRoute,
    SearchResponse, SearchResult, Document, Answer, Attempt, Usage,
    Capability, Depth, ContentStatus,
    QuotaPolicy, Period, Unit, Anchor,
    Provider, register, available_providers, discover_providers,
    SearchRouteError, ConfigError, ProviderError, AuthError,
    QuotaExceeded, RateLimited, TransientError, NoProviderAvailable,
)
```
