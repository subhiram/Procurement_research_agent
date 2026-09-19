# Design decisions

Each entry: what was decided, what it costs, and what would change it. The
evidence matters more than the conclusion — a decision whose reasoning is lost
gets reversed by the next person who finds it inconvenient.

---

## Contacts are found by regex and verified against the page

**Decided:** emails and phone numbers are extracted by regex
(`extraction/patterns.py`), so they are literal substrings of the source page by
construction. The model only *chooses* which candidate is the right sales
contact. Everything it returns is then re-checked against the page text in
`grounding.py`; anything unverifiable is dropped and the reason recorded.

**Why:** a hallucinated email does not look wrong. It is well-formed, plausibly
named after the company, and flows straight into a procurement workflow. Nobody
notices until outreach silently goes nowhere — or reaches a real stranger.

**Cost:** a vendor whose contact the regex misses ships with no contact at all.
Accepted: a lead with a source URL is useful; a lead with a fabricated email is
worse than nothing.

**What would change it:** nothing about model quality. The grounding check is
deliberately independent of the pipeline that produced the answer, because the
thing being defended against is the model ignoring its instructions.

---

## Designations are discarded unless they appear in a retrieved source

**Decided:** `material_research` searches first, hands the model the retrieved
text, and then **drops any designation or synonym not present in it** — and any
that is not grade-specific.

**Why:** this was the largest source of wrong output. A model asked for the UNS
equivalent of a trade name supplies one whether or not it knows it, and "UNS
S46500" reads exactly as plausibly as "UNS S45500". The damage spreads:
`vendor_search` builds its queries from these designations, so one invented
number sends the entire search after a different alloy.

**Evidence:** live runs discarded `AMS 5383` (absent from sources), and
`ASTM A564`, `ASTM B637`, `ASTM B348`, `ISO 9723`, `DIN 2.4819` (real, present,
but family standards spanning many grades). `ASTM B348` is the one that matters
— it covers every titanium bar grade, and matching on it previously passed
Grade 7 pages for a Grade 5 enquiry.

**Cost:** a genuine designation the search failed to surface is lost. The notes
field says what was dropped and why, so a thin list reads as "we could not
source these" rather than "this material has no equivalents".

**Not filtered:** `ambiguities`. That field is the model reporting its own
uncertainty, and there is no corpus against which a doubt can be verified.
Filtering it would delete exactly the warning the buyer most needs.

---

## Ranking is deterministic, not model-judged

**Decided:** `vendor_summary` sorts by a fixed key — manufacturer above
distributor above trader, verified contact above none, directories and retail
last — with no model involved.

**Why:** the ordering rule is a stable business preference, not a judgment call
worth spending free-tier tokens on per run. It is also the part a buyer will
question, and "because the rules say so" is an answer where "because the model
thought so" is not.

**Cost:** no nuance. A superb trader ranks below a mediocre manufacturer.

---

## Retail and non-suppliers are kept, ranked last, and labelled

**Decided:** a page that is clearly not a supplier is not dropped. It sinks to
the bottom with a note saying why.

**Why:** a live run returned NIST (a standards body selling reference chips) and
three retail sites among twelve results for a 150 kg enquiry. Dropping them
silently would leave an unexplained gap; a misjudged vendor should be visible
and labelled rather than vanish. Occasionally a small-quantity source is exactly
what someone wants.

---

## The archive replaced the cache

**Decided:** every completed run is written to `runs/<date>-<thread>.json`, and
**nothing in the graph ever reads it back**. Retrieval is an explicit CLI or API
call.

**Why:** the vendor cache made a *reuse* decision the operator could not see — a
later run silently received vendors found weeks earlier. Most of its complexity
existed to make that safe: TTLs, alias distinctiveness rules, embedding-based
material matching, and a contact re-verification protocol. All of it served a
decision nobody had asked for.

**Cost:** repeat searches cost credits again. Accepted: the credits are cheap
compared to a buyer emailing a contact that went stale without anyone noticing.

**Enforced by a test** asserting no node imports `load_run`, `list_runs` or
`search_runs` — the property is easy to lose by accident.

---

## The run budget counts metered searches, not provider credits

**Decided:** one call that costs the account something counts as one, whatever
the provider bills internally. Keyless providers count as zero.

**Why:** provider units are not comparable. The same content search bills 2 on
Tavily, 100 on Exa and 0 on arXiv. A ceiling in those units means "twelve
searches" against one provider and nothing at all against another.

**Evidence:** measured — three Exa queries reported 300 against a 25-credit
budget and exhausted the run before the vendor search had issued half its
queries. Separately, flooring the charge at 1 meant the entire academic pass and
both reference lookups consumed budget they never actually spent.

**The provider's own figure is still recorded** in the trace as
`provider_cost_units`, for spotting an expensive provider. It is just not what
the ceiling is denominated in.

---

## Allocations are spending limits, not transfers

**Decided:** `SessionBudget.sub(n, name)` caps what a node may spend. It does
not move credits out of the run's pool.

**Why:** taking them up front looks equivalent and is not. An allocation that
goes unused — the academic pass on a material with no literature — would keep
them, and the vendor search the run exists to perform would find the pool
drained by nodes that never spent anything.

**Evidence:** four allocations totalling 12 removed 12 of 25 credits from a run
that made five charged searches.

Allocations are **memoised by name**, which is what lets a twelve-way fan-out
share one pool instead of taking twelve.

---

## Structured output is chosen per endpoint

**Decided:** each endpoint declares `structured_output_method` in LLMRoute's
`limits.yaml`, and `LLMRouter.with_structured_output` binds per candidate rather
than once for the whole ladder. Passing `method=` explicitly is refused.

**Why:** the right method is a property of the model, not the request. Measured:
Groq's `gpt-oss` models answer in prose under function calling and Groq rejects
that with a 400; Gemma's tool calling is too weak to rely on. Both need
`json_schema`. But **Groq's `allam-2-7b` rejects `json_schema` outright** and
cannot do tool calling either — so even a per-*provider* setting is wrong.

Endpoints that cannot do structured output at all declare
`supports_structured_output: false` and are filtered out of schema-carrying
requests, rather than discovered by burning an attempt on a certain 400.

---

## A model that cannot produce the shape is skippable, not fatal

**Decided:** a 400 carrying `json_validate_failed` — or saying the model does
not support the response format — steps to the next candidate instead of failing
the call.

**Why:** the router treats a non-transient 4xx as fatal, correctly: a malformed
request should fail once, not fourteen times. But these say *this model* could
not produce the shape, and another will. Found live: a resume died mid-run and
surfaced to the client as an `error` event when the ladder would have served it.

**The exemption is narrow on purpose** — matched on the provider's error *code*
first, since message wording changes without notice. A bad parameter is still
fatal.

---

## Tiers are measured, and the ladder must clear them

**Decided:** tasks name a quality tier, not a provider. Ordering *within* a tier
is left to live quota rather than fixed, since a hardcoded chain is only correct
until the first provider runs dry.

**Bake-off on this project's real schemas:**

| Provider | Intake parse | Contact extraction | Speed |
| --- | --- | --- | --- |
| `groq/gpt-oss-20b` | full (dims + qty + unit) | correct | 0.8 s |
| `gemma4:e4b` (local) | partial (name only) | correct | 19–25 s |
| `llama3.1` / `3.2` | partial (name only) | **all nulls** | 2–8 s |

MLX Ollama builds cannot be used at all: they ignore structured-output schemas
entirely, because the MLX backend does no grammar-constrained decoding.

**`DEFAULT_MAX_ATTEMPTS` must exceed the largest tier.** The cap is applied to
the ordered candidate list *before* availability is checked, so parked endpoints
consume slots. This has bitten twice — at 6 (one short of tier S) and at 8, when
tier B grew to 14 and put the free local backstop six places beyond reach. A
test now fails the suite if a tier grows past it.

---

## Ollama is the keyless backstop, and it is last

**Decided:** a local Ollama endpoint sits at priority 950 — last in its tier —
and declares no `api_key_env`, so it stays routable on a machine with no
credentials at all.

**Why last:** it is free and unmetered but roughly an order of magnitude slower.
It should absorb overflow, not lead.

**Why NVIDIA is not promoted above it in the general order:** NVIDIA NIM has no
daily or token cap, which is genuinely attractive, but it is shared public
infrastructure that can be slow or time out under load — which is why it carries
a 120-second timeout where everything else gets 60. It sits third, ahead of
Ollama, which is what matters: a hosted endpoint is always tried before the
local one.

---

## crawl4ai runs ahead of metered extraction

**Decided:** a local headless Chromium fetches pages first; only what it cannot
get falls through to a paid extract.

**Why:** it costs nothing, and it is the only path here that renders
JavaScript — vendor contact details frequently sit in a nav or footer that does
not exist until the page has run. Measured: following contact pages took
verified contacts from 6 of 12 vendors to **11 of 12**.

**Cost:** Chromium is most of the container image, and some sites (403 on
anti-bot) refuse it. A browser that will not start disables the tier for the
process rather than failing every page slowly.

---

## The trace is derived from framework seams, not hand-written

**Decided:** node spans and LLM calls come from LangChain's callback protocol;
search events come from SearchRoute's `hooks`. No node reports anything itself.

**Why:** instrumentation that each node must remember to call is instrumentation
that will be forgotten in the node added next month. LLMRoute already stamps
every reply with its routing decision; that data was being produced and thrown
away.

**Two things learned the hard way:**

- **Use an allowlist of node names, not a blocklist of framework internals.**
  LangChain raises a chain event for every Runnable it composes, so one real
  node produced five spans named `RunnableParallel<raw>`, `PydanticOutputParser`
  and so on. Naming what we want is stable; enumerating what we do not never is.
- **`ladder_step` is not a fallback signal.** It says which rung served the
  call, and its numbering depends on what was pinned. Every request here is
  tier-pinned, so all of them report step 3 — reading `step > 1` as a fallback
  reported "half of all calls are falling back" when the true figure was none.
  Count the recorded failed attempts instead.
