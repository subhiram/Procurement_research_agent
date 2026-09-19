# Fallbacks

What happens when each part fails, and which test holds it up.

These paths only run when something has already gone wrong, which makes them the
least-exercised code here. That is not a theoretical concern: a cap silently
amputated the routing ladder six places short of the free local model, every
test passed, and three live runs never noticed. Everything below is therefore
forced deliberately — with scripted providers in the unit tests, and with a
broken environment in the live drill.

The rule they all share: **a run that loses a source degrades and says so,
rather than failing.** A vendor with no page text still ships with its source
URL, because a real company the buyer can look up beats a gap.

---

## LLM ladder

Failures are normalised into two shapes — `RateLimited` and `ProviderError` —
in `LLMRoute/llm_router/providers.py`, then acted on in `router.py`.

| Condition | What happens | Test |
| --- | --- | --- |
| 429, or Groq's non-standard 498 | `RateLimited`; endpoint parked for its `retry_after`; next candidate serves | `test_rate_limit_falls_through_to_the_next_provider`, `test_groq_498_is_treated_as_a_rate_limit` |
| 401 / 403 | `AuthError`; parked 900 s, so a bad key costs one attempt not every call | `test_auth_failures_are_auth_errors` |
| 404 unknown model | parked 3600 s; a deprecated id does not poison the whole run | `test_an_unknown_model_id_parks_that_endpoint` |
| 5xx, timeout, connection refused | transient; 20 s cooldown; ladder continues | `test_a_transient_5xx_moves_on` |
| 400 `json_validate_failed` | **skippable** — the model could not produce the shape, another can | `test_a_model_failing_to_produce_valid_json_is_skippable` |
| 400 "does not support response format" / "tool calling" | skippable, same reasoning | `test_unsupported_response_format_is_skippable_not_fatal` |
| Any other non-transient 4xx | **fatal** — a malformed request fails once, not fourteen times | `test_an_ordinary_bad_request_is_still_fatal` |
| Model declared incapable of structured output | filtered out of schema-carrying requests before it is tried | `test_a_model_that_cannot_do_structured_output_is_hidden_from_those_calls` |
| Tier exhausted | downgrade S→A→B, flagged as `tier_downgraded` | `test_exhausting_a_tier_downgrades_and_flags_it` |
| **Whole tier rate-limited** | **reaches Ollama** — the cap must clear the tier | `test_the_local_backstop_is_reached_when_every_hosted_endpoint_is_spent` |
| A tier grows past the attempt cap | the suite fails, rather than the tail going unreachable | `test_max_attempts_clears_every_tier` |
| Ollama daemon absent | transient; hosted providers carry the run | `test_ollama_is_routable_with_no_credentials_at_all` |
| Brief 429 inside `max_wait` | waited out rather than downgraded | `test_max_wait_retries_once_quota_frees_up` |
| Everything gone | `AllCandidatesExhausted`, naming what was tried and when to retry | `test_exhaustion_error_explains_itself` |
| Process restart | ledger reloads; quota already spent is not respent | `LLMRoute/tests/test_ledger.py` |

### The one that was broken

`_Walk.acquirable()` truncates candidates at `DEFAULT_MAX_ATTEMPTS` **before**
availability is checked, so parked endpoints consume slots. Tier B grew to 14
endpoints when Ollama was added at position 14; the cap was 8. The free,
unmetered backstop was unreachable in exactly the circumstance it exists for —
every hosted endpoint rate-limited.

It failed silently: no exception, no warning, just `AllCandidatesExhausted`
listing `ollama/gemma4:e4b [tier B] - available` among the candidates it never
tried.

---

## Search ladder

Provider fallback is SearchRoute's; budget and degradation are the agent's
(`search/client.py`).

| Condition | What happens | Test |
| --- | --- | --- |
| A provider errors | SearchRoute tries the next; every attempt is recorded as a `search_attempt` event | `test_records_each_provider_attempt` |
| Every provider fails | `search()` returns `[]`; the run continues degraded, never raises | `test_a_provider_failure_returns_nothing_rather_than_raising` |
| Run budget exhausted | `BudgetExhausted`; node degrades; run marked `truncated` | `test_exhaustion_marks_the_run_truncated` |
| One allocation exhausted | that node stops; the rest of the run still spends | `test_an_allocation_cannot_outspend_its_share` |
| A twelve-way fan-out | shares one allocation, not twelve | `test_a_fan_out_cannot_exceed_its_shared_allocation` |
| Concurrent branches racing the ceiling | the ceiling holds — reserve is atomic | `test_the_ceiling_holds_under_concurrent_branches` |
| Keyless provider answers | costs nothing against the run budget | `test_a_free_provider_costs_nothing` |
| Expensive provider answers | costs one, not its internal 100 | `test_an_expensive_provider_does_not_exhaust_the_run` |
| Chromium will not start | tier disabled for the process; metered extraction takes over | `test_a_browser_that_will_not_start_is_not_fatal` |
| Chromium gets nothing useful | those URLs fall through to `extract()` | `test_what_the_crawler_misses_falls_through_to_extraction` |
| Extraction fails too | lead ships with its source URL and no page text | `test_extraction_failing_leaves_the_lead_intact` |
| Page text already good | neither tier is called | `test_content_that_is_already_good_is_left_alone` |
| Capability gating | `ACADEMIC` reaches only arXiv/PubMed/Crossref; `REFERENCE` only Wikipedia | `test_searches_the_academic_capability`, `test_the_spec_lookup_asks_for_reference_not_the_open_web` |

---

## The live drill

`scripts/fallback_drill.py` forces the same conditions against real providers.
Unit tests prove the error *mapping*; only a live run proves the *wiring*.

```bash
uv run python scripts/fallback_drill.py            # all five
uv run python scripts/fallback_drill.py --only no-keys
uv run python scripts/fallback_drill.py --list
```

| Scenario | Forces | Asserts |
| --- | --- | --- |
| `no-keys` | every provider key blanked | only Ollama and keyless search served; nothing charged |
| `bad-groq-key` | 401 on the tier leader | Groq served nothing; the run still completed |
| `no-ollama` | daemon pointed at a dead port | hosted providers carried it; nothing waited on the daemon |
| `tiny-budget` | 2 credits | completed, marked `truncated`, results still grounded |
| `no-crawler` | `ENABLE_CRAWL4AI=false` | completed via metered extraction, contacts still found |

It asserts on **the archived trace**, not on stdout — a run can print a perfectly
good answer while having quietly reached it the wrong way, and the trace is what
says which provider actually served each call.

### Two things the drill found on its first run

**Blanking a key is not the same as deleting it.** `import llm_router` calls
`load_dotenv()`, which repopulates anything *absent* from the environment
straight out of `.env`. The first `no-keys` run had Groq, Mistral, Tavily and
Exa all serving the run it was supposed to be testing without them. The
scenarios now set keys to `""`, which dotenv will not overwrite and both routers
read as "no credential".

**With no keys there is exactly one web-search provider.** DuckDuckGo is IP
rate-limited, and `ddgs` responds to pressure by rotating onto a backup backend
— which then failed DNS, producing zero vendors. It is paced with a
`min_interval` now. But the honest statement of the no-keys claim is: *the
routing degrades correctly and the run completes; the yield depends on a single
free provider that may be unavailable.* The drill asserts the first and warns
about the second, rather than going red for someone else's outage.

---

## What is not covered

- **No test forces a mid-run Postgres outage.** The checkpointer would raise and
  the run would fail; that is untested.
- **Langfuse being unreachable** is handled (the handler is only built when keys
  are set, and construction failure is caught) but not tested against a real
  dead endpoint.
- **Partial extraction** — a page that returns some text but not the contact
  block — is covered only incidentally, through the grounding check dropping
  what it cannot verify.
