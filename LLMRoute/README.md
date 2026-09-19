# llm_router

One callable that routes an LLM request across free-tier providers, so nothing
else in your codebase imports a provider SDK.

```python
from llm_router import route

response = route(messages, strategy="free_first")
response = route(messages, strategy="sticky", session_id="research-run-42")
response = route(messages, model="qwen-27b")                        # prefer a logical model
response = route(messages, provider="groq", model="gpt-oss-120b")   # pin an exact endpoint
response = route(messages, tier="S")                                # set a quality floor
```

and the same thing as a LangChain model, which is the primary use case:

```python
from llm_router import LLMRouter
from langgraph.prebuilt import create_react_agent    # or langchain.agents.create_agent

router = LLMRouter(strategy="free_first")
agent = create_react_agent(model=router, tools=my_tools)
```

`LLMRouter` is a thin `BaseChatModel` over `route()` - invoke, ainvoke, stream,
batch, `bind_tools` and `with_structured_output` all funnel into the same
routing logic, so the function and the class cannot drift apart.

## Status

v0. Providers in scope: **Groq, Mistral, Google AI Studio, NVIDIA NIM,
OpenRouter**, plus a local **Ollama** daemon. Cloudflare Workers AI and Ollama
Cloud have their quota facts recorded in `config/limits.yaml` but are disabled
until each gets a wrapper.

**Ollama is the one keyless endpoint.** Its `config/limits.yaml` block declares
no `api_key_env`, so it stays in the ladder on a machine with no API keys at
all, and it is free and unmetered. It is also roughly an order of magnitude
slower, so it sits at priority 950 - last within its tier - and absorbs overflow
rather than leading. Point it somewhere else with `OLLAMA_BASE_URL`.

**Cerebras has been removed.** Its free allowance was a one-time grant of
sign-up credit rather than a quota that refills, so routing ordinary traffic
there spent a non-renewable balance and then lost it - the opposite of what
every other provider here is for. The provider block, the wrapper class and the
`langchain-cerebras` dependency are all gone.

Not a pip package yet - copy the `llm_router/` folder into a project and import
it.

## Install

```bash
pip install -r requirements.txt          # or just the providers you use
```

Keys can be real exported env vars, or a `.env` file in your project root -
`import llm_router` loads it automatically (via `python-dotenv`) before
anything reads `os.environ`, so either works:

```bash
# .env
GROQ_API_KEY=...
MISTRAL_API_KEY=...
GOOGLE_API_KEY=...                       # or GEMINI_API_KEY
NVIDIA_API_KEY=...
OPENROUTER_API_KEY=...
```

Providers without a key are skipped automatically - the ladder just routes
around them, so you can start with one key and add more later.

## Verify the model ids first

**Do this once before trusting the config.** Free-tier model ids drift
constantly: models get renamed, versioned, promoted out of preview, or retired.

```bash
python scripts/verify_models.py            # diff config against live catalogues
python scripts/verify_models.py --suggest  # also list models you are not using
python scripts/smoke_test.py               # one real call per endpoint
```

`verify_models.py` exits non-zero when a configured id is missing upstream, so
it can sit in CI. A stale id is survivable at runtime - the router parks that
endpoint for an hour and moves on - but it silently costs you an endpoint you
thought you had.

## How routing works

### The ladder

`resolve_candidates()` builds an ordered list of endpoints worth trying,
cheapest deviation first:

| Step | What it tries | Quality |
|---|---|---|
| 1 | the same model on another provider | unchanged |
| 2 | another model in the same tier, on a provider that hosts the requested model | unchanged |
| 3 | another model in the same tier, anywhere | unchanged |
| 4 | a lower tier, anywhere | **downgraded** |

Only step 4 changes the answer's quality, so candidates reached that way carry
`tier_downgraded=True`. Steps 2 and 3 swap the model but hold the tier and must
never be reported as a downgrade - and equally, a tier-A answer is never passed
off as the tier-S one you asked for.

Every response records how it got there:

```python
response.response_metadata["llm_router"]
# {'provider': 'nvidia_nim', 'model_id': 'gemma-4-31b-it', 'logical_model': 'gemma-31b',
#  'tier': 'A', 'requested_tier': 'A', 'tier_downgraded': False, 'ladder_step': 1,
#  'reason': 'requested model', 'strategy': 'free_first', 'tokens': 128,
#  'latency': 0.83, 'attempts': [...]}
```

`route(..., return_route=True)` returns a `RouteResult` with the same
information as attributes, plus every attempt that was made and why it failed.

### Tiers

Tier is a property of the model family, not of whoever hosts it.

- **S** (flagship): `gpt-oss-120b`, `deepseek-v4-pro`, `kimi-k3`, `glm-5.2`, `minimax-m3`, `nemotron-ultra`
- **A** (mid): `qwen-27b`, `gemma-31b`, `mistral-medium`, `ministral-14b`, `nemotron-super`, `gemini-flash`, `minimax-m2.7`, `inkling`
- **B** (small/fast): `gpt-oss-20b`, `allam-2-7b`, `mistral-small`, `ministral-8b`, `ministral-3b`, `gemma-26b`, `gemini-flash-lite`, `inkling-small`, `laguna-xs`, `lfm-2.5`

`route()` tries at most `max_attempts` endpoints (default 8) before giving up.
That ceiling has to clear the largest tier: it is applied to the ordered
candidate list, so if one tier holds more live endpoints than the cap, a request
pinned to that tier spends the whole budget inside it and never reaches the
downgrade it is entitled to. Tier S is currently six endpoints - raise the
default alongside any tier that grows past it.

Restricted to models each account's own free-tier quota dashboard actually
lists - see the comments at the top of `config/models.yaml` for what was
deliberately left out (agentic meta-models, code-completion models, unversioned
previews) and why.

### Strategies

- **`free_first`** - take the ladder as it comes; the first endpoint with quota
  wins. Optimised for "just get it done", and does not make noise about tier
  downgrades (they are still recorded in the metadata).
- **`sticky`** - keep a `session_id` on the endpoint it has been using, so a long
  run does not silently change model half way through. Falls back down the
  ladder when that endpoint runs dry, and **logs a warning on any tier
  downgrade**, because a silent quality drop is exactly what a
  consistency-sensitive pipeline must not get.

A tier downgrade is deliberately never pinned to a session. Downgrades are
temporary relief while the requested tier is out of quota; pinning one would
hold the rest of the session at the lower quality even after the good endpoint
came back.

Custom strategies:

```python
from llm_router import register_policy

def prefer_groq(candidates, session_id=None, session_state=None):
    return sorted(candidates, key=lambda c: c.endpoint.provider != "groq")

register_policy("prefer_groq", prefer_groq)
route(messages, strategy="prefer_groq")
```

## Quota tracking

Only Groq (response headers) and OpenRouter (a key endpoint) will tell you how
much quota is left. For everyone else the `UsageLedger` is the only source of
truth: it is seeded from `config/limits.yaml`, counts what you spend locally,
and is corrected the moment a real 429 arrives.

Counting is deliberately conservative - every ambiguity resolves towards "we
have used more than we think", because over-counting costs one fallback hop
while under-counting costs a 429 and a wasted round trip.

| Provider | Quota scope | Ledger bucket | Notes |
|---|---|---|---|
| Groq | per model | one per model | `x-ratelimit-*` headers correct the ledger on every call; the non-standard **498** "flex tier capacity exceeded" is treated as a rate limit, not a failure |
| Mistral | per model | one per model | free tier is ~1 request/second, enforced by `min_interval_seconds`. No usage endpoint exists, so the ledger is all you have |
| NVIDIA NIM | **per account** | **one shared bucket** | one flat 40 rpm across every model, spaced 1.5s apart. Shared public infrastructure that its own docs warn can be slow or time out under load, so its wrapper waits 120s rather than 60s before giving up; a timeout is transient, so the ladder steps past it either way |
| OpenRouter | **per account** | **one shared bucket** | the whole `:free` catalogue draws on one 20 rpm / **50 rpd** budget - the tightest daily cap here, which is why every OpenRouter endpoint sits last at priority 60. Adding more `:free` models buys variety, not capacity. A **402** (out of credits) is normalised to a rate limit with a long cooldown rather than a fatal bad request, so it parks OpenRouter instead of failing the call. `x-ratelimit-reset` is an absolute epoch, not a duration, and is converted |
| Google AI Studio | **per project** | **one shared bucket** | extra API keys under one project do not multiply anything, so spending quota on `gemini-flash` eats into `gemma-31b`'s budget. Each model is still checked against its own ceiling, so `gemini-flash-lite` keeps its larger allowance. Free RPM is low enough that reacting to a 429 is already a wasted request, so calls are spaced pre-emptively. 429s are parsed for `QuotaFailure.violations[].quotaId` and `RetryInfo.retryDelay` |

All eight Gemini and Gemma models the account exposes are configured, but under
`per_project` they pool into that one bucket and a single 429 cools all of them
down together - so the extra ids buy model variety, not capacity. Google's own
429 body names the limit `GenerateRequestsPerDayPerProjectPerModel-FreeTier`,
which suggests the counters really are per model; switching `quota_scope` to
`per_model` in `config/limits.yaml` gives each of the eight its own allowance.
It is left strict by default because under-using a free tier is cheaper than
429ing on it.

Inspect it at any time:

```python
from llm_router import default_ledger
default_ledger().snapshot()
# {'groq:openai/gpt-oss-120b': {'counters': {'requests/minute': {'used': 3, 'limit': 30, ...}}, ...}}
```

### Surviving restarts

The ledger is in-memory by default. Daily and weekly caps are the reason you may
not want that - a fresh process with an empty ledger will happily respend a
quota it already used and only find out via 429s:

```python
from llm_router import configure
configure(ledger_path="~/.cache/llm_router/ledger.json")
```

## Config

Two data files, no code.

**`config/models.yaml`** maps a logical model to the real endpoints that serve
it, with its tier:

```yaml
models:
  gemma-31b:
    tier: A
    endpoints:
      - {provider: nvidia_nim, model_id: gemma-4-31b-it, priority: 40}
      - {provider: google_ai_studio, model_id: gemma-4-31b-it, priority: 50}
      - {provider: openrouter, model_id: google/gemma-4-31b-it:free, priority: 60}
```

Lower `priority` is tried first. The shipped priorities run
groq(10) < mistral(30) < nvidia_nim(40) < google(50) < openrouter(60), roughly
by how generous each free tier is and tie-broken by how reliably it answers, so
a request that names no provider spends the abundant quota first and keeps
Google's 5 rpm / 20 rpd and OpenRouter's 50 rpd in reserve.

`non_chat_patterns` at the bottom of the file
refuses classifiers, guard models, embeddings and audio models at load time -
things like Groq's `llama-prompt-guard-2-*` and `*-safeguard-*`, or
OpenRouter's `nvidia/nemotron-3.5-content-safety:free`, are not general chat
models and must never be selectable as one.

**`config/limits.yaml`** holds the quota facts per provider. Limit keys are
generic - `rpm`/`rph`/`rpd`/`rpw`/`rpmo` for requests and `tpm`/`tpd`/... for
tokens, over rolling windows. A `_uncached` suffix is accepted and tracked as
the same counter. An unrecognised key is an error rather than a silent no-op,
because a typo'd limit would otherwise read as "unlimited".

## Adding a provider

1. Add its quota facts under `providers:` in `limits.yaml` and set
   `enabled: true`.
2. Add its endpoints to the relevant models in `models.yaml`.
3. Add a `BaseProvider` subclass in `providers.py` - usually just `_construct()`,
   plus an `interpret()` override if its errors are unusual.
4. `register_provider("name", YourProvider)`.
5. Run `python scripts/verify_models.py --provider name`.

The base class already handles 429/5xx/auth/timeouts structurally rather than by
exception class, which matters because these SDKs reshuffle their exception
types between releases.

## Tests

```bash
python -m pytest              # 147 tests, no network
```

The suite runs against scripted fakes, including a real LangGraph agent doing a
full tool-call round trip, so it covers the plumbing without spending anyone's
free tier. `scripts/smoke_test.py` is the live counterpart.

## Known limitations

- **Model ids are unverified** until you run `verify_models.py` with real keys.
- **Rolling windows approximate calendar resets.** Providers that reset daily
  quota at a fixed wall-clock time are modelled as a rolling 24h window, which
  is conservative (it under-uses rather than over-uses).
- **Token estimates are crude** - characters/4 plus assumed output - and are used
  only to pre-check token quotas, never sent to a provider.
- **Streaming can only fall back before the first chunk.** The provider wrapper
  pulls that chunk eagerly so a rate limit surfaces early; once tokens have been
  handed to the caller, switching model mid-answer would produce nonsense.
- **Google's quota scope is set strict.** Its 429 bodies name limits as
  `...PerProjectPerModel`, which suggests per-model counters rather than one
  project-wide pool. The config pools them anyway (see the comment in
  `limits.yaml`), which may leave free requests unused; flipping one line to
  `per_model` changes it.
- **The ledger is per process.** Several processes sharing one API key will each
  count only their own spend. Groq's headers correct for this; nobody else's do.
