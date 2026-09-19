# SearchRoute — guide for AI agents

You are looking at a Python package that has been copied into this project. This
file tells you what it does and how to call it correctly. Read the **Gotchas**
section before writing code; those are the mistakes that produce silently wrong
results.

## What it is

SearchRoute is a **router in front of many web-search providers**. It normalizes
Tavily, Exa, Firecrawl, Google, DuckDuckGo, arXiv, Wikipedia and others behind one
interface, tracks each provider's free-tier quota, and automatically falls back to
the next provider when one is drained, rate-limited or down.

You give it a query. It gives you back results, and page text when you ask for it.

## What it does NOT do

**It makes no LLM calls.** No query planning, no summarization, no semantic
reranking, no chunking. It returns faithful cleaned markdown and stops.

Do not expect `search()` to reason. *You* are the reasoning step — the library
fetches, you decide. If you need a summary of a page, call `read_page`/`extract`
and summarize it yourself.

## Import it

```python
from searchroute import SearchRoute
```

Only hard dependency is `httpx`. Two optional extras improve the keyless path:
`ddgs` (DuckDuckGo search) and `trafilatura` (local page extraction). Both degrade
with a clear message if missing — nothing crashes.

## 60-second usage

```python
from searchroute import SearchRoute

sr = SearchRoute()                        # auto-discovers API keys from the env
r = sr.search("what is retrieval augmented generation")

for hit in r:
    print(hit.title, hit.url, hit.snippet)
```

Works with **zero API keys** (DuckDuckGo + local extraction), just less reliably.

## The two axes

Almost every question resolves to one of these. They are independent.

| Axis | Question | Parameter |
|---|---|---|
| Which provider | Who answers this? | `providers=`, `strategy=` |
| How rich | What comes back per result? | `depth=` |

```python
sr.search("q")                                # snippets (default)
sr.search("q", depth="content")               # + full page markdown
sr.search("q", providers=["tavily", "exa"])   # try Tavily, fall back to Exa
```

## Decision table

| The user wants | Write this |
|---|---|
| A quick factual lookup | `sr.search(q, max_results=5)` |
| Full text of the results | `sr.search(q, depth="content")` |
| Text of one known URL | `sr.extract([url])` |
| A specific provider only | `sr.search(q, providers=["exa"])` |
| Ordered fallback | `sr.search(q, providers=["tavily", "exa"])` |
| Maximum recall (research) | `sr.search(q, strategy="fanout", n=3)` |
| Cheapest possible call | `SearchRoute(profile="quicklook")` |
| Academic papers | `sr.search(q, capability=Capability.ACADEMIC)` |
| Encyclopedia background | `sr.search(q, capability=Capability.REFERENCE)` |
| Developer discussion | `sr.search(q, capability=Capability.DISCUSSION)` |
| Recent news | `sr.search(q, capability=Capability.NEWS)` |
| A direct answer | `sr.answer(q)` — **may return `None`** |

`Capability` imports from `searchroute`.

The specialized sources (arXiv, PubMed, Crossref, Wikipedia, Hacker News) are
**only** reachable through their capability. They deliberately never appear in a
general `search()`, so a question about pizza cannot return an arXiv preprint.

## Signatures

```python
sr.search(
    query: str,
    *,
    depth: str = "snippets",       # "links" | "snippets" | "summary" | "content"
    max_results: int = 10,
    providers: list[str] | None = None,    # order IS the fallback order
    exclude: list[str] | None = None,
    strategy: str | None = None,   # priority|quota_aware|quality|latency|round_robin|race|fanout
    n: int | None = None,          # only with strategy="race"/"fanout"
    capability: Capability = Capability.SEARCH,
    max_hydrate: int | None = None,        # cap pages fetched at depth="content"
    include_domains: list[str] | None = None,
    exclude_domains: list[str] | None = None,
    start_date: datetime | None = None,
    lang: str | None = None,
    region: str | None = None,
) -> SearchResponse

sr.extract(urls: list[str]) -> list[Document]
sr.answer(query: str) -> Answer
sr.status() -> dict     # current state: quota left, circuit breakers
sr.stats() -> dict      # history: calls, success rate, latency per provider
sr.close()              # or use as a context manager
```

`AsyncSearchRoute` has the same surface with `await`.

## Return types

```python
SearchResponse
    results: list[SearchResult]
    answer: str | None
    depth: Depth              # what was achieved
    requested_depth: Depth    # what was asked for
    degraded: bool            # True if anything fell short
    notes: list[str]          # plain-language reasons — READ THESE when degraded
    providers_used: list[str]
    attempts: list[Attempt]   # per-provider audit trail
    cost: Usage               # .total, .by_provider

    len(r), iter(r), r.urls   # convenience

SearchResult
    url: str
    title: str
    snippet: str | None
    content: str | None           # only at depth="content"
    summary: str | None           # provider-generated only, often None
    content_status: ContentStatus # native | hydrated | failed | skipped | not_requested
    provider: str                 # who found it
    content_provider: str | None  # who fetched the text (often different)
    published_date: datetime | None
    score: float | None
    rank: int
    raw: dict                     # untouched provider payload
    .text                         # richest available: content or summary or snippet

Document                          # from extract()
    url, title, content, ok, provider, error
```

Use `hit.text` when you just want something to reason over.

## Gotchas — these cause broken code

**1. `n=` does nothing without `strategy=`.**
```python
sr.search(q, n=3)                          # WRONG — silently ignored
sr.search(q, strategy="fanout", n=3)       # right
```

**2. `providers=[...]` also restricts extraction.**
A pinned list of search-only providers makes `depth="content"` return nothing.
```python
sr = SearchRoute(providers=["duckduckgo"])
r = sr.search(q, depth="content")          # WRONG — every result FAILED
print(r.notes)                             # tells you exactly this

sr = SearchRoute(providers=["duckduckgo", "jina", "http_extract"])   # right
```

**3. Provider list order IS the fallback order.**
`providers=["tavily", "exa"]` means try Tavily first. It is not reordered.

**4. `answer()` can legitimately return `None`.**
Only some providers generate answers. Always check.
```python
a = sr.answer(q)
if a.answer is None:
    ...  # fall back to a.results
```

**5. `content_status` distinguishes two very different things.**
`failed` = we could not fetch the page. `not_requested` = you didn't ask for it.
Never present a `failed` result as if the page had no content.

**6. `depth="content"` costs real credits.** It is never implicit. Default is
`snippets`. Use `max_hydrate=N` to cap how many pages get fetched.

**7. Check `response.degraded` and `response.notes`.** A degraded response still
returns results; the notes say what went wrong and how to fix it.

## Cost model

Calls per query, per strategy. Do not reach for `fanout` on a trivial lookup.

| Strategy | Calls | Use for |
|---|---|---|
| `priority` (default) | 1 | Predictable, most cases |
| `quota_aware` | 1 | Making free tiers last |
| `quality` | 1 | Best result quality |
| `latency` | 1 | Speed |
| `round_robin` | 1 | Even drain across tiers |
| `race(n)` | up to n | Speed, paying for losers |
| `fanout(n)` | n | Maximum recall |

`depth="content"` adds up to `max_hydrate` extra extraction calls on top.

## Diagnosing "no results"

```python
from searchroute import discover_providers
discover_providers()      # which providers are actually configured
sr.status()               # quota left, circuit breaker state
sr.stats()                # what has been failing, and why
```

If a search raises `NoProviderAvailable`, its message names each provider and why
it was ruled out (no key / quota exhausted / circuit open).

## Environment variables

```
TAVILY_API_KEY  EXA_API_KEY  FIRECRAWL_API_KEY  SERPER_API_KEY  SERPAPI_API_KEY
GOOGLE_API_KEY + GOOGLE_CSE_ID     JINA_API_KEY     SEARXNG_URL
SEARCHROUTE_CONTACT      # email; polite-pool access to Crossref/PubMed/Wikipedia
```

`BRAVE_API_KEY` alone does **not** enable Brave — it bills per query, so it must
be named explicitly in `providers=[...]`.

## Registering these as LLM tools

If your job is to give a model search ability, don't hand it `search()` — its
routing and cost parameters are yours to set, not the model's.

```python
from searchroute import SearchRoute
from searchroute.tools import Toolset

sr = SearchRoute(profile="rag")            # you fix the routing
tools = Toolset(sr, max_results_cap=5)     # and the cost ceiling

tools.anthropic()      # [{"name", "description", "input_schema"}]
tools.openai()         # [{"type": "function", "function": {...}}]

result = await tools.dispatch("web_search", {"query": "..."})
result.text            # compact, ready to hand back to the model
```

Five tools: `web_search`, `read_page`, `academic_search`, `reference_lookup`,
`news_search`. The model searches, then calls `read_page` on what's worth reading
in full — that keeps context small without truncating anything.

Adapters also exist: `searchroute.tools.adapters.to_langchain(tools)`,
`to_llamaindex(tools)`, and an MCP server via
`python -m searchroute.tools.mcp_server`.

`dispatch()` never raises for an expected failure — it returns readable error text
so your agent loop keeps running.

## Full documentation

`USAGE.md` in the source repository, if it was copied alongside this package.
