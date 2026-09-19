# Procurement Research Agent

Turns a raw-material purchase request into a verified list of worldwide vendors
with contact details and source links.

```
"I am looking to buy Custom 465 with Dia 2 inch - 200 KG"
   -> clarifying questions (only what is genuinely ambiguous)
   -> material research, verified against retrieved sources
   -> research literature mined for named suppliers
   -> vendor search, then each vendor's own contact page (crawled free)
   -> verified contacts, each traceable to the page it came from
   -> the whole run archived to runs/ with a trace of what it did
```

A run takes 2–3 minutes and returns around twelve vendors, usually ten with a
contact. Optionally drafts outreach emails to them.

**Out of scope:** quotation generation, pricing, and defense-side document
creation. This automates research and vendor discovery only.

---

## Quick start

```bash
cp .env.example .env          # add API keys - all of them are optional
docker compose up -d
curl -H "X-API-Key: $API_KEY" localhost:8000/health
```

**It runs with no API keys at all**, on a local Ollama daemon and the keyless
search providers (arXiv, PubMed, Crossref, Wikipedia, DuckDuckGo). Adding keys
widens the routing ladder rather than changing any code.

From a checkout instead:

```bash
uv sync --extra dev
uv run crawl4ai-setup                      # one-off: fetches Chromium
docker compose up -d postgres              # checkpoints live here

uv run python -m procurement_agent.cli --new "Custom 465 Dia 2 inch - 200 KG"
uv run python -m procurement_agent.cli --resume <thread-id> --answer "H900, round bar"
uv run python -m procurement_agent.cli --list-runs
```

The resume works from a fresh process days later — the session is checkpointed
in Postgres, which is the whole point of the conversational front half.

---

## Documentation

| | |
| --- | --- |
| **[docs/](docs/)** | Start here |
| [docs/architecture.md](docs/architecture.md) | The three components, the graph, where state lives |
| [docs/design-decisions.md](docs/design-decisions.md) | Every significant choice, with its evidence |
| [docs/fallbacks.md](docs/fallbacks.md) | What happens when each part fails |
| [docs/operations.md](docs/operations.md) | Configuring, running, diagnosing |
| [API_CONTRACT.md](API_CONTRACT.md) | The HTTP surface, for building a frontend |

---

## How it works, briefly

Provider routing is delegated to two libraries developed alongside this agent:
**[LLMRoute](LLMRoute/)** picks which model serves a call and what happens when
it fails; **[SearchRoute](SearchRoute/)** does the same for search. This project
keeps only what they cannot know — that a designation must appear in a retrieved
source before it can be trusted, that a page selling bar by the inch is not a
supplier for a 150 kg order, that a contact absent from the page it claims to
come from must be dropped.

Three properties are worth stating plainly, because they shape everything else:

**Nothing is invented.** Contacts are found by regex, so they are literal
substrings of the source page, then re-verified against it by an independent
check. Material designations are discarded unless they appear in retrieved
sources — a live run dropped five, including `ASTM B348`, which had previously
made Grade 7 titanium pages match a Grade 5 enquiry.

**Free-tier budgets are the binding constraint, not latency.** Fan-out is
deliberately narrow, model tiers are assigned by measured bake-off, and one run
has a hard credit ceiling so a single wide search cannot consume the month.

**Runs are archived, never cached.** Nothing in the graph reads a past run back,
enforced by a test. The vendor cache this replaced made a silent reuse decision
the operator could not see.

---

## Providers

| | Role |
| --- | --- |
| **Groq** | Leads every tier — fastest and most accurate here |
| **Mistral / NVIDIA NIM / Google AI Studio / OpenRouter** | The ladder, ordered by how generous each free tier is |
| **Ollama** | Keyless local backstop; last in its tier, free and unmetered |
| **crawl4ai** | Page fetching, first tier — free, and the only path that renders JavaScript |
| **Tavily / Exa / Firecrawl** | Search and extraction |
| **arXiv, PubMed, Crossref, Wikipedia, DuckDuckGo** | Keyless, no credits |

Ceilings and where they bite: [docs/operations.md](docs/operations.md#free-tier-ceilings-and-where-they-bite).

---

## Tests

```bash
uv run pytest                                  # the agent
(cd LLMRoute && uv run pytest)                 # model routing
(cd SearchRoute && uv run pytest)              # search routing
uv run ruff check src/ tests/ scripts/

uv run python scripts/fallback_drill.py        # live degradation drill
```

The drill forces real failure conditions — no keys, a bad key, no Ollama, a
starved budget, no browser — because unit tests prove the error mapping and only
a live run proves the wiring.

---

## Export control

Contact details are extracted automatically and should be confirmed before use.
Materials sourced through this tool may be subject to export control (ITAR/EAR);
the agent flags this but does not attempt to enforce it.
