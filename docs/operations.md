# Operations

Running it, configuring it, and diagnosing it.

---

## Running

### Docker (the normal way)

```bash
cp .env.example .env          # add keys; all of them are optional
docker compose up -d
curl -H "X-API-Key: $API_KEY" localhost:8000/health
```

Two containers: `postgres` (checkpoints) and `api`. Ollama stays on the **host**
— compose reaches it via `host.docker.internal`, with an `extra_hosts` entry so
the same file works on Linux. That keeps its GPU/Metal acceleration and your
existing model pulls; a containerised Ollama on macOS has neither.

If port 8000 is taken, set `API_PORT`. A collision otherwise stops the stack
with a daemon-level bind error that explains nothing.

### From a checkout

```bash
uv sync --extra dev
uv run crawl4ai-setup                      # one-off: fetches Chromium
docker compose up -d postgres              # still needed for checkpoints
uv run python -m procurement_agent.cli --new "Custom 465 Dia 2 inch - 200 KG"
```

### Checking it is wired up

```bash
uv run python -m procurement_agent.cli --validate-models    # every task has an endpoint
uv run python LLMRoute/scripts/verify_models.py             # model ids still exist upstream
uv run python -c "from searchroute import discover_providers; print(discover_providers())"
```

`verify_models.py` is worth running after any provider change: a stale model id
costs an endpoint an hour-long cooldown at runtime, mid-request.

### Streamlit UI

```bash
uv sync --extra ui
docker compose up -d postgres
uv run streamlit run src/procurement_agent/ui/app.py
```

A chat interface that opens the graph directly (`src/procurement_agent/ui/app.py`)
— no FastAPI in the loop. Needs exactly what the CLI needs: `.env` loaded and
Postgres reachable, nothing else. A clarification interrupt renders as a
normal, tagged chat bubble, and the same input box is used to answer it — free
text, not a form, matching the shape `API_CONTRACT.md` documents for the HTTP
client. One caveat worth knowing: `open_graph()` closes the crawl4ai browser
and the search client at the end of every call, so this server pays that
startup cost once per chat turn rather than once per process the way the CLI
does — a deliberate simplicity tradeoff, not an oversight.

---

## Configuration

Every setting, what it does, and when to touch it. All are environment
variables; `.env` is read at startup.

### Budgets

| Setting | Default | |
| --- | --- | --- |
| `SEARCH_CREDITS_PER_SESSION` | `25` | Ceiling for **one run**, across every node. Counts metered searches; keyless providers cost nothing. Raise if runs come back `truncated` with vendors still worth finding. |
| `RESULTS_PER_QUERY` | `5` | Hits per query. Raising it costs more extraction than search. |
| `MAX_FANOUT` | `3` | Graph-wide concurrency. Deliberately low: on free tiers, wide parallelism produces simultaneous 429s rather than speed. |
| `RESEARCH_SEARCH_CREDITS` | `4` | The academic pass's share. Usually free in practice — arXiv, PubMed and Crossref cost nothing. |

Per-node allocations live in `graph/context.py` (`ALLOCATIONS`), not in the
environment: they are a property of what each node does, not a deployment
choice.

### Providers

No API keys appear in this app's settings — LLMRoute and SearchRoute read them
from the environment themselves. See `.env.example` for the names.

| Setting | Default | |
| --- | --- | --- |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | The keyless backstop. In Docker this is `http://host.docker.internal:11434`. |
| `LLM_STICKY_SESSION` | `false` | Pin a run to whichever provider served its first call. Off because spreading across providers beats concentrating load on one limit — and this graph's tasks span tiers, so a single pin does not apply cleanly. |
| `LLM_LEDGER_PATH` | `~/.cache/procurement_agent/llm_ledger.json` | **Needs a volume in Docker.** Losing it means every restart believes the daily quota is unspent and rediscovers the truth through 429s. |

### Behaviour

| Setting | Default | |
| --- | --- | --- |
| `ENABLE_RESEARCH_SEARCH` | `true` | The academic pass. Coverage is uneven — a specialty alloy does far better than a structural steel. |
| `ENABLE_SPEC_LOOKUP` | `true` | One reference lookup before the clarifying questions, so they are not built on a misread trade name. This is the only node a person waits on; turn it off if latency there matters more. |
| `ENABLE_CRAWL4AI` | `true` | The free page-fetch tier. Off means metered extraction only, and fewer verified contacts. |
| `CRAWL_MAX_CONTACT_PAGES` | `3` | Contact pages followed per vendor. This is the main quality lever: following them took verified contacts from 6 of 12 to 11 of 12. |
| `CRAWL_CONCURRENCY` | `2` | These are other companies' websites. Keep it low. |
| `CRAWL_RESPECT_ROBOTS` | `true` | Leave it on. |
| `ENABLE_RUN_ARCHIVE` | `true` | Writes `runs/<date>-<thread>.json`. |

### API

| Setting | Default | |
| --- | --- | --- |
| `API_KEY` | `dev-local-key` | **Change it.** |
| `API_PORT` | `8000` | Host port, bound to `127.0.0.1`. |
| `CORS_ORIGINS` | localhost 3000/5173/8080 | Explicit allowlist. Never `*` — these endpoints take a credential and return third-party contact data. |

### Observability

`LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` enable Langfuse; both must be
set. `LANGFUSE_HOST` chooses cloud or a self-hosted instance — that is the only
change needed to move, and `docker-compose.langfuse.yml` exists for when you do.

**The local trace does not depend on any of this.** Every run records one in its
archive file regardless.

---

## Free-tier ceilings, and where they bite

| Provider | Limit | Bites when |
| --- | --- | --- |
| Groq | 30 rpm, 1000 rpd, **8000 tpm** | The tpm figure is the tight one, and `contact_extraction` is token-heavy |
| NVIDIA NIM | 40 rpm, **no daily or token cap** | Rarely — but it is slow under load, hence its 120 s timeout |
| Mistral | ~60 rpm, 20k tpm | Sooner than its published numbers suggest |
| Google AI Studio | **5 rpm, 20 rpd** on Flash | Almost immediately; it is a last resort |
| OpenRouter | 20 rpm, 50 rpd **account-wide across all `:free`** | The tightest daily budget here |
| Ollama | local | Never — it is slow, not limited |
| Tavily / Firecrawl | 1,000 credits/month | A content search costs 2 |
| Exa | 10,000 credits/month | A content search costs **100** |
| arXiv, PubMed, Crossref, Wikipedia, DuckDuckGo | keyless | Never charged; DuckDuckGo is IP rate-limited |

Set `SEARCHROUTE_CONTACT` to an email — Crossref and PubMed give politer rate
limits to callers who identify themselves.

---

## Reading a trace

Every archived run carries what it actually did.

```bash
jq '.totals' runs/<run>.json
jq '.trace[] | select(.kind=="search")' runs/<run>.json
jq '.trace[] | select(.kind=="llm") | {provider, model, attempts}' runs/<run>.json
```

`totals` at a glance:

| Field | Means |
| --- | --- |
| `llm_calls` / `llm_inner_calls` | Should track each other. A large gap means something is calling a model outside LLMRoute. |
| `llm_fallbacks` | Calls where something failed before one succeeded. Counted from recorded attempts — **not** from `ladder_step`, which says which rung served the call and is always 3 for a tier-pinned request. |
| `searches_charged` | Against the run ceiling. |
| `provider_cost_units` | The providers' own units. Not comparable between providers; use it to spot an expensive one, not to budget. |
| `nodes_run` | Node order. `contact_extraction` appears once per vendor. |

---

## Diagnosing

| Symptom | Likely cause |
| --- | --- |
| `truncated: true`, short vendor list | Budget spent. Check `searches_charged` against `SEARCH_CREDITS_PER_SESSION`. |
| Few verified contacts | Crawler not working. Look for `crawl4ai:` lines; in Docker check `shm_size` — Chromium crashes on 64 MB `/dev/shm`. |
| Designations look thin | Working as intended: unsourced ones are discarded. `research.notes` says what went and why. |
| Run fails with an `error` event | Read `detail`. A fatal 4xx is a genuine bad request; anything else should have fallen through, which is a bug worth reporting. |
| `AllCandidatesExhausted` | Every endpoint in the tier is parked. Check `--validate-models` and the ledger. |
| Quota exhausted right after a restart | The LLM ledger is not persisted. In Docker, that is the `agent_state` volume. |
| Browser CORS error from the frontend | Origin not in `CORS_ORIGINS`. It presents as opaque, never as a helpful message. |
| Stream appears to hang then end | The client is splitting SSE frames on `\n\n` without normalising CRLF. See [API_CONTRACT.md](../API_CONTRACT.md#streaming). |

---

## Data and retention

- `runs/` — one JSON file per run, holding **real companies' email addresses and
  phone numbers**. Gitignored and excluded from Docker images. Nothing prunes
  it.
- Postgres `public.checkpoints` — every session's state, including suspended
  ones. `docker compose down -v` destroys these and every past run's state with
  them.
- The two quota ledgers — LLMRoute's at `LLM_LEDGER_PATH`, SearchRoute's at
  `$XDG_STATE_HOME/searchroute/quota.json`.

The Postgres image is still `pgvector/pgvector:pg16`. The `vector` extension is
no longer used — it went with the vendor cache — but changing the image forces a
collation `REINDEX` on the existing volume, so it is left alone deliberately.
