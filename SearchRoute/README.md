# SearchRoute

One search interface for AI agents, many providers underneath.

Agents need web search, but every provider is a different SDK, a different response shape,
and a different free tier that eventually runs out. SearchRoute sits in between: it
normalizes the providers, tracks each one's free-tier quota locally, and falls back down
the chain when one is drained, rate-limited or down.

```python
from searchroute import SearchRoute

sr = SearchRoute()                        # auto-discovers API keys from the environment
for hit in sr.search("what is retrieval augmented generation?"):
    print(hit.title, hit.url)
```

📖 **[Full usage guide →](USAGE.md)** — every option, strategy, and use case explained.
🤖 **[Agent guide →](src/searchroute/AGENTS.md)** — lives inside the package, so it travels with a directory copy.

## Install

```bash
pip install searchroute                  # core
pip install 'searchroute[all]'           # + keyless DuckDuckGo and local extraction
```

Set a key for whichever providers you have; SearchRoute enables exactly those.

```bash
export EXA_API_KEY=...
export TAVILY_API_KEY=...
```

## Two things it routes on

**Which provider** — by strategy, quota and health.
**How rich each result is** — by `Depth`.

These are independent. The same query can come back as bare links or as full page markdown
without changing anything about provider selection.

| Depth | You get | Cost |
|---|---|---|
| `links` | title + url | 1 call |
| `snippets` *(default)* | + the provider's snippet | 1 call |
| `summary` | + a provider-generated summary | 1 call, pricier params |
| `content` | + full cleaned page markdown | 1 search + K extracts |

```python
sr.search("q")                                  # snippets
sr.search("q", depth="content")                 # full markdown per result
sr.search("q", depth="content", max_hydrate=3)  # ...but only fetch 3 pages
```

### Hydration

Not every provider can return page content. Google CSE and DuckDuckGo only ever give you
snippets. When you ask for `content` and the selected provider can't serve it, SearchRoute
**composes** rather than failing — it searches on one provider and extracts on another:

```
search(google_cse) → 10 urls
      │ depth gap
hydrate: firecrawl → tavily → jina → local trafilatura
      │
results[].content filled in, each labelled with how
```

Every result carries a `content_status`, so you always know what you're holding:

```python
r = sr.search("q", depth="content")
for hit in r:
    match hit.content_status:
        case "native":   ...  # the search provider returned it inline
        case "hydrated": ...  # a second extract call fetched it
        case "failed":   ...  # paywalled, 403, timeout — snippet still available
        case "skipped":  ...  # outside the max_hydrate budget

r.degraded    # True if anything fell short of what you asked for
r.cost        # exactly what this query spent, per provider
```

That distinction matters: `failed` means *we couldn't fetch this page*, which is a very
different thing from *this page has no content*.

## Strategies

```python
sr = SearchRoute(strategy="quota_aware")
sr.search("q", strategy="fanout", n=3)     # per-call override
```

| Strategy | Behavior |
|---|---|
| `priority` *(default)* | Your explicit order, sequential fallback. Predictable. |
| `quota_aware` | Whoever has the most headroom left. Makes several free tiers last. |
| `quality` | Best provider for the job, spend be damned. |
| `latency` | Fastest observed provider first. |
| `round_robin` | Rotate, so tiers drain evenly. |
| `race(n=2)` | Fire n at once, take the first success. Costs n calls. |
| `fanout(n=3)` | Query n and fuse the rankings. Best recall, costs n calls. |

Fan-out merges with Reciprocal Rank Fusion rather than comparing provider scores directly —
Exa's neural relevance and Google's rank don't share a scale, so we fuse on rank position.

## Profiles

```python
SearchRoute(profile="quicklook")   # snippets, 5 results, no extraction
SearchRoute(profile="rag")         # content, quota-aware, hydrate 8
SearchRoute(profile="research")    # content, fanout(3), 25 results, hydrate 15
```

## Providers and free tiers

Verified September 2026. These move — check before relying on them.

| Provider | Free tier | Renews | Card? | Max depth | Env |
|---|---|---|---|---|---|
| **SearXNG** | unlimited self-hosted | — | no | `snippets` | `SEARXNG_URL` |
| **Exa** | $20 signup + $10/mo | monthly | no | `content` | `EXA_API_KEY` |
| **Tavily** | 1,000 credits | monthly | no | `content` | `TAVILY_API_KEY` |
| **Firecrawl** | 1,000 credits | monthly | no | `content` | `FIRECRAWL_API_KEY` |
| **Google CSE** | 100 queries/day | **daily** | no | `snippets` | `GOOGLE_API_KEY` + `GOOGLE_CSE_ID` |
| **SerpAPI** | ~100 searches | monthly | no | `snippets` | `SERPAPI_API_KEY` |
| **Serper** | 2,500 queries | **one-time** | no | `snippets` | `SERPER_API_KEY` |
| **SearchApi** | small trial | **one-time** | no | `snippets` | `SEARCHAPI_API_KEY` |
| **Jina** | extract free; search keyed | — | no | `content` | `JINA_API_KEY` *(search only)* |
| **http_extract** | local, unlimited | — | no | extract only | — |
| **DuckDuckGo** | unmetered, rate-limited | — | no | `snippets` | — |
| **Brave** | none — card required | — | **yes** | `snippets` | `BRAVE_API_KEY` *(opt-in)* |

Providers are listed in default chain order. Free and unmetered first, then the most
generous recurring tiers, with one-time grants held back as reserve and the keyless
fallbacks last.

Three honest caveats:

- **DuckDuckGo is a safety net, not a tier.** The `ddgs` library is unofficial, trips bot
  detection well under 30 requests/minute from one IP, and describes itself as
  educational-use-only. It is forced to the back of every ordering and exists so a call can
  still return something when everything else is drained. Don't plan capacity around it.
- **Brave killed its free tier.** It now requires a card and meters past a $5 credit. It is
  implemented but **excluded from auto-discovery**: setting `BRAVE_API_KEY` is not enough,
  you must name it in `providers=[...]`. A key in your environment isn't consent to spend.
- **Jina's two endpoints differ.** `r.jina.ai` (reading a page) works with no key;
  `s.jina.ai` (search) returns 401 without one. So keyless Jina is offered for extraction
  only — which is still valuable, since it renders pages server-side and beats the local
  extractor. Add `JINA_API_KEY` to also use it for search.

### One-time vs renewing

The router treats these differently on purpose. A monthly tier refills whether you used it
or not, so spending it is free; a one-time grant like Serper's 2,500 never comes back. The
`quota_aware` strategy therefore sorts one-time grants **last** and keeps them as reserve.

## Specialized sources

Five more providers, all keyless and unmetered — and all **invisible to a general
search**. Somebody asking about pizza should never get an arXiv preprint, so these
are reached by asking for a *kind* of search instead:

```python
from searchroute import Capability

sr.search("attention is all you need", capability=Capability.ACADEMIC)    # arXiv, PubMed, Crossref
sr.search("photosynthesis",            capability=Capability.REFERENCE)   # Wikipedia
sr.search("rust async runtimes",       capability=Capability.DISCUSSION)  # Hacker News
```

| Source | Capability | Notes |
|---|---|---|
| **arXiv** | `ACADEMIC` | Preprints. Returns Atom XML; rate-limited to 1 request / 3s. |
| **PubMed** | `ACADEMIC` | Biomedical. Two HTTP calls per search (ids, then metadata). |
| **Crossref** | `ACADEMIC` | DOI metadata across publishers. |
| **Wikipedia** | `REFERENCE` | Serves full article text natively — no extraction hop. |
| **Hacker News** | `DISCUSSION` | Practitioner opinion, with points and comment counts. |

Because they're unmetered, `quota_aware` tries them *before* spending on Exa, and
a fan-out across all three academic sources costs nothing:

```python
sr.search(q, capability=Capability.ACADEMIC, strategy="fanout", n=3)   # cost: 0
```

Domain metadata (DOI, authors, journal, citation counts) is on `result.raw`.

Several of these run "polite pools" for identified callers. Set
`SEARCHROUTE_CONTACT=you@example.com` — it costs nothing and it's what their terms
ask for.

## Use it as LLM tools

Don't hand a model `search()` — its routing and cost parameters are yours to set,
not the model's. `Toolset` exposes a fixed, narrow surface instead:

```python
from searchroute.tools import Toolset

sr = SearchRoute(profile="rag")          # you fix the routing
tools = Toolset(sr, max_results_cap=5)   # and the ceiling

tools.anthropic()   # or .openai()
result = await tools.dispatch("web_search", {"query": "..."})
```

Five tools — `web_search`, `read_page`, `academic_search`, `reference_lookup`,
`news_search`. **`depth`, `strategy`, `providers` and `max_hydrate` never appear
in a schema**, so a model can't spend your month on one call. The model searches,
then reads only what's worth reading in full.

Also available: `searchroute.tools.adapters.to_langchain(tools)` /
`to_llamaindex(tools)`, and an MCP server (`python -m searchroute.tools.mcp_server`)
for Claude Code and Claude Desktop. Worked examples in
[`examples/research/`](examples/research/).

## Local extraction (no keys needed)

`http_extract` is the tail of the extract chain: it fetches the page itself and runs a
readability pass over it. It's why `depth="content"` works with no API keys at all — but
it's a genuine fallback, not an equal. No JavaScript rendering and no proxy rotation, so a
client-rendered page comes back as `FAILED` rather than as a shell of navigation chrome.
An agent citing an empty page is worse than one that knows the page is missing.

It identifies itself honestly rather than impersonating a browser, which is also simply
more effective: Wikipedia returns **403** to a spoofed Chrome User-Agent and **200** to a
descriptive one. If you run this at any volume, add contact details — several sites' bot
policies ask for them:

```python
SearchRoute(provider_options={
    "http_extract": {"user_agent": "MyApp/1.0 (+https://example.com; me@example.com)"}
})
```

## Quota tracking

SearchRoute keeps a local ledger of what each provider has spent in its current reset
window, so `quota_aware` can prefer whoever has room instead of discovering exhaustion by
getting a 402 back.

```python
sr.status()   # remaining credits, reset times, circuit-breaker state per provider
```

The ledger is deliberately advisory — it can drift when another process or machine spends
against the same key. A real quota error from the provider always wins and immediately
syncs the local counter to drained. It defaults to a lock-protected JSON file under
`$XDG_STATE_HOME/searchroute/`, so several worker processes on one machine share one count.

`reserve_pct` (default 5%) holds a sliver of every tier back, so a background job can't
drain the last credit an interactive call was going to need.

## Use it without installing

The package is a plain directory. Copy it and import:

```bash
cp -r src/searchroute /your/project/
pip install httpx                        # the only hard dependency
pip install ddgs trafilatura             # optional: keyless search + extraction
```

`AGENTS.md` travels inside the package, so an AI agent working in that project can
read what SearchRoute does and how to call it without this repository.

## Custom providers

Your own search endpoint is a first-class provider — it gets routed, budgeted and
failed-over exactly like a built-in.

```python
from searchroute import Provider, Capability, Depth, register

@register
class InternalDocs(Provider):
    name = "internal"
    capabilities = frozenset({Capability.SEARCH})
    native_depths = frozenset({Depth.LINKS, Depth.SNIPPETS, Depth.CONTENT})
    requires_key = False

    async def search(self, query): ...
```

## What this is not

SearchRoute has **no LLM dependency and makes no model calls**. It does not plan queries,
decompose questions, summarize, semantically rerank, or chunk content. It returns complete
cleaned markdown and hands your agent faithful material to work with.

`summary` depth is passed through from providers that generate summaries themselves — never
synthesized locally. Research orchestration belongs in your application, where you already
have a model.

## Async

`AsyncSearchRoute` is the real implementation; the sync `SearchRoute` is a facade over it
that runs on a background loop, so it works unchanged in notebooks and web handlers.

```python
async with AsyncSearchRoute(profile="research") as sr:
    r = await sr.search("q")
```

## License

MIT
