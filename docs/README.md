# Documentation

Start here.

| | For |
| --- | --- |
| [architecture.md](architecture.md) | How it fits together: the three components, the graph, what each node does and spends, where state lives |
| [design-decisions.md](design-decisions.md) | Why it is shaped this way — each decision with its evidence and what would change it |
| [fallbacks.md](fallbacks.md) | What happens when each part fails, and the test that holds it up |
| [operations.md](operations.md) | Running, configuring, and diagnosing it |
| [../API_CONTRACT.md](../API_CONTRACT.md) | The HTTP surface, for building a frontend against |

## The short version

A raw-material purchase request goes in; a ranked list of vendors with verified
contact details comes out, each traceable to the page it came from.

```
"Custom 465 Dia 2 inch - 200 KG"
  -> clarifying questions, only where genuinely ambiguous
  -> material research, verified against retrieved sources
  -> research literature mined for named suppliers
  -> vendor search, then each vendor's own contact pages
  -> verified contacts, ranked by whether the vendor can fill the order
  -> the whole run archived, with a trace of what it did
```

A run takes 2–3 minutes and returns around twelve vendors, usually ten of them
with a contact.

## The three things worth knowing first

**Nothing is invented.** Contacts are found by regex, so they are literal
substrings of the source page, and then re-verified against it independently.
Material designations are discarded unless they appear in a retrieved source. A
lead that ships with no email is a real answer, not a failure.

**Free-tier budgets are the binding constraint, not latency.** Every design
choice about fan-out, model tiers and search depth follows from that. It runs
with no API keys at all — on a local Ollama and the keyless search providers —
and keys widen the ladder rather than changing any code.

**Runs are archived, never cached.** Nothing in the graph reads a past run back.
That is enforced by a test, because it is the property most easily lost by
accident, and silent reuse was what the previous cache got wrong.

## Where the code is

```
src/procurement_agent/
  graph/           the LangGraph state machine; one module per node
  search/          the search seam and the per-run credit budget
  llm/             the model seam - one function, get_model(task, schema)
  extraction/      regex contact and certification finding
  crawl/           local headless-browser page fetching
  grounding.py     the independent re-check that contacts appear on their page
  designations.py  which material designations actually identify one grade
  archive.py       per-run records
  trace.py         what a run did, collected from framework callbacks
  api/             FastAPI, SSE streaming
LLMRoute/          model routing: ladder, quota ledger, provider wrappers
SearchRoute/       search routing: providers, strategies, quota
```

The modules carry their reasoning in their docstrings — often more of it than
these documents do. `grounding.py`, `designations.py` and `trace.py` are the
ones most worth reading directly.
