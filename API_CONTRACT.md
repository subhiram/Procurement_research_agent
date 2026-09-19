# Procurement Research Agent — API Contract

Everything a frontend needs, without reading the source.

The API turns a free-text raw-material request into a ranked list of vendors
with contact details, each traceable to the page it came from. A run takes
**2–3 minutes** and may pause partway to ask the buyer a question.

- [Read this first](#read-this-first)
- [Auth and CORS](#auth-and-cors)
- [The conversation](#the-conversation)
- [Streaming](#streaming)
- [Endpoints](#endpoints)
- [Schemas](#schemas)
- [Displaying results](#displaying-results)
- [Errors](#errors)
- [What this API does not do](#what-this-api-does-not-do)

---

## Read this first

Three things will cost you an afternoon if you learn them by debugging.

**1. `EventSource` will not work.** The streaming endpoints are `POST`, and
`EventSource` only issues `GET`. It also cannot set request headers at all, so
it could never send `X-API-Key`. Use `fetch()` with a `ReadableStream` reader —
[worked example below](#streaming).

**2. A run takes 2–3 minutes.** Measured: 133 s, 193 s, 226 s. Show progress
from the `node_end` events, not a spinner. If you put a proxy in front of this,
raise its read timeout well past the usual 60 s default or runs will appear to
fail at the one-minute mark.

**3. The stream always ends in exactly one of `interrupt`, `final`, or
`error`.** You can rely on reaching a known state. `interrupt` means the agent
needs an answer before it can continue — it is not a failure.

---

## Auth and CORS

Every endpoint requires a shared key:

```
X-API-Key: <API_KEY>
```

Missing or wrong gives `401`.

CORS is configured with an **explicit origin allowlist** (`CORS_ORIGINS`), not a
wildcard. The defaults cover `localhost` on ports 3000, 5173 and 8080. If your
dev server runs elsewhere, add it there — a blocked origin presents as an
opaque browser CORS error, not a helpful message.

> **Before you deploy this anywhere shared:** a key in frontend JavaScript is
> readable by anyone who opens devtools, and `GET /runs` returns real companies'
> email addresses and phone numbers. For anything beyond localhost, put the API
> behind a proxy that holds `API_KEY` server-side and never ships it to the
> browser. The API is bound to `127.0.0.1` by default so this has to be a
> deliberate choice.

---

## The conversation

```
POST /sessions                      -> { thread_id }
        |
POST /sessions/{id}/messages        -> SSE
        |
        +-- node_end ... node_end
        |
        +-- final          done, render the vendors
        |
        +-- interrupt      questions -> ask the buyer
                |
        POST /sessions/{id}/resume  -> SSE
                |
                +-- node_end ... -> final
```

A session is a `thread_id`. `POST /sessions` only allocates one — nothing runs
until the first message.

**Interrupts survive a restart.** The buyer can answer an hour later, from a
different device. If your client loses the stream, `GET /sessions/{id}` returns
the current state including any `pending_questions`, and you can resume from
there. Streaming is how you watch a run; it is not where the state lives.

---

## Streaming

Both streaming endpoints return `text/event-stream` with frames shaped:

```
event: node_end
data: {"node": "material_research", "status": "researching"}

```

### Events

| Event | When | Payload |
| --- | --- | --- |
| `node_end` | A graph node finished | `{ node, status }` |
| `interrupt` | The agent needs an answer | `{ kind, questions[], thread_id }` |
| `final` | The run completed | the full result, [below](#final) |
| `error` | The run failed | `{ detail, type }` |

`node_end` is your progress bar. The nodes in order are `intake_parser`,
`clarify_spec`, `ask_clarification` (only when a question is asked),
`material_research`, `research_sourcing`, `vendor_search`, then
`contact_extraction` **once per vendor found** — usually twelve, and the bulk of
the wall time — then `vendor_summary` and `save_run`.

### Consuming it

```js
async function streamRun(url, body, { apiKey, onEvent }) {
  const response = await fetch(url, {
    method: "POST",
    headers: {
      "X-API-Key": apiKey,
      "Content-Type": "application/json",
      Accept: "text/event-stream",
    },
    body: JSON.stringify(body),
  });
  if (!response.ok) throw new Error(`${response.status} ${await response.text()}`);

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    // Normalise CRLF first. This server sends \r\n line endings, so splitting
    // on "\n\n" matches nothing and you silently parse zero events - the
    // stream appears to hang and then end.
    buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");

    // Frames are separated by a blank line. Keep the trailing partial frame in
    // the buffer - a chunk boundary lands mid-frame often enough to matter.
    const frames = buffer.split("\n\n");
    buffer = frames.pop() ?? "";

    for (const frame of frames) {
      let event = "message";
      const data = [];
      for (const line of frame.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) data.push(line.slice(5).trim());
      }
      if (data.length) onEvent(event, JSON.parse(data.join("\n")));
    }
  }
}
```

> Keep-alive comment lines (`: ping`) appear in the stream too. The loop above
> ignores them for free, because a frame with no `data:` line is skipped.

```js
```

Used:

```js
const { thread_id } = await (await fetch(`${API}/sessions`, {
  method: "POST", headers: { "X-API-Key": apiKey },
})).json();

await streamRun(`${API}/sessions/${thread_id}/messages`,
  { content: "Inconel 718 round bar, 3 inch diameter, 150 KG, AMS 5662" },
  { apiKey, onEvent: (event, data) => {
      if (event === "node_end")   setProgress(data.node);
      if (event === "interrupt")  askTheBuyer(data.questions);   // then /resume
      if (event === "final")      showVendors(data.vendor_leads);
      if (event === "error")      showError(data.detail);
  }});
```

Resuming takes `{ answers: "..." }` — free text, exactly as the buyer typed it.
There is no need to structure it or map it back to individual questions.

---

## Endpoints

| Method | Path | Returns |
| --- | --- | --- |
| `POST` | `/sessions` | `{ thread_id }` — 201 |
| `POST` | `/sessions/{id}/messages` | SSE. Body `{ content }` |
| `POST` | `/sessions/{id}/resume` | SSE. Body `{ answers }` |
| `GET` | `/sessions/{id}` | current state, for reconnecting |
| `POST` | `/sessions/{id}/emails` | drafts outreach emails — one model call per vendor |
| `GET` | `/runs` | archived runs, newest first. `?q=` filters, `?limit=` caps |
| `GET` | `/runs/{run_id}` | one archived run, complete, including its trace |
| `GET` | `/health` | liveness plus current budget settings |

`POST /sessions/{id}/messages` on an existing thread is treated as a resume, so
a client that reconnects and simply sends text does the right thing.

### `final`

```jsonc
{
  "thread_id": "…",
  "status": "done",
  "material_spec": { … },        // MaterialSpec
  "research": { … },             // MaterialResearch
  "vendor_leads": [ … ],         // VendorLead[], already ranked
  "truncated": false,            // true = search budget ran out, list incomplete
  "search_credits_remaining": 18,
  "archived_to": "runs/2026-09-10-….json",
  "disclaimer": "Contact details are extracted from…"
}
```

### `GET /runs`

Index entries only — `run_id`, `saved_at`, `request`, `summary`, `keywords`,
`vendor_count`, `totals`. Fetch one by id for the whole record: the spec, the
research, every query and search result, every lead, and the `trace` of what the
run actually did (which model served each node, which search provider answered,
what it cost).

---

## Schemas

Generated from the Pydantic models in `src/procurement_agent/graph/state.py`.

### MaterialSpec

| Field | Type | |
| --- | --- | --- |
| `material_name` | string | required. The trade name, e.g. `Custom 465` |
| `grade` | string? | UNS/AMS/ASTM designation if known |
| `form` | string? | bar, rod, sheet, plate, tube, wire, forging, powder |
| `dimensions` | string? | as stated, e.g. `Dia 2 inch` |
| `quantity` | number? | |
| `unit` | string? | KG, LB, PCS, M, FT |
| `condition` | string? | temper / heat treatment, e.g. `H900` |
| `standards` | string[] | standards explicitly called out |

`material_name` deliberately keeps the trade name and puts the formal
designation in `grade` — vendors list by trade name, so generalising it would
wreck the search. Display both.

### MaterialResearch

| Field | Type | |
| --- | --- | --- |
| `canonical_name` | string | required |
| `designations` | string[] | UNS/AMS/werkstoff equivalents, **verified against retrieved sources** |
| `synonyms` | string[] | trade names, also verified |
| `common_forms` | string[] | |
| `ambiguities` | string[] | **show these prominently** — see below |
| `notes` | string? | includes what was discarded as unverified |

**`ambiguities` is the field that matters most.** It means the name could refer
to more than one material, and if it is wrong the entire vendor list is for the
wrong product. Surface it above the results, not in a details pane.

### ClarificationQuestion

| Field | Type | |
| --- | --- | --- |
| `field` | string | which `MaterialSpec` field this resolves |
| `question` | string | show this |
| `why` | string | why it matters — show it, it materially improves answers |

### VendorLead

| Field | Type | |
| --- | --- | --- |
| `company_name` | string | required |
| `website` | string | required |
| `country` | string? | |
| `kind` | enum | `manufacturer` \| `distributor` \| `trader` \| `retail` \| `not_a_supplier` \| `unknown` |
| `contact_name` | string? | |
| `email` | string? | verified present on `contact_source_url` |
| `phone` | string? | likewise |
| `certifications_found` | string[] | filtered to this material plus company approvals |
| `source_url` | string | required. The product page the vendor was found on |
| `contact_source_url` | string? | the page the contact was actually verified against |
| `cited_in_research` | boolean | named as a supplier in a research paper — strong signal, worth a badge |
| `mentions_material` | boolean | false = the page never named the material |
| `confidence_notes` | string[] | **warnings for the buyer** — see below |

### EmailDraft

`vendor_company`, `to_email?`, `subject`, `body`. Drafts only — nothing is sent.

---

## Displaying results

These are domain rules, not styling preferences. Getting them wrong makes the
output misleading rather than merely ugly.

- **`vendor_leads` arrives ranked. Do not re-sort it.** The order encodes
  whether a vendor can actually fill an industrial order: manufacturers above
  distributors above traders, verified contacts above none, directories and
  retail last.
- **`confidence_notes` are warnings meant to be read.** "This page does not name
  Inconel 718 — it may be a neighbouring grade. Confirm before enquiring." That
  is for the buyer, not a debug log. Show them next to the vendor.
- **`retail` and `not_a_supplier` are kept deliberately**, ranked last and
  labelled. They are not noise to filter out — a misjudged vendor should be
  visible rather than silently vanish. Label them; do not hide them.
- **`mentions_material: false`** deserves a visible marker. The vendor may still
  stock it, but the page did not say so.
- **`truncated: true`** means the budget ran out and the list is incomplete. Say
  so, or the buyer will read a short list as "few suppliers exist".
- **Always show `disclaimer`** with any exported or printed contact list.
  Contacts were verified when found, not at the moment of reading, and the
  material may be export-controlled.
- **An empty `email` and `phone` is a real answer**, not an error. A vendor that
  ships with only a `source_url` was found but not verified — link the URL.

---

## Errors

| Status | Meaning |
| --- | --- |
| `401` | missing or invalid `X-API-Key` |
| `404` | no such session or archived run |
| `409` | `/resume` on a session that is not waiting for clarification |
| `503` | the graph is not ready (startup failed) |

Failures *during* a run arrive as an SSE `error` event, not an HTTP status — by
then the response has already begun streaming with a 200. Handle both.

```json
{ "detail": "…", "type": "ProviderError" }
```

---

## What this API does not do

Worth knowing before you design around it.

- **No cancellation.** A run started by mistake spends its budget to completion.
  Do not offer a cancel button; it would be a lie.
- **No per-user isolation.** Every holder of the key sees every archived run.
- **No real pagination.** `GET /runs?limit=` truncates; there is no cursor.
- **No websockets**, and no server push outside a request's own stream.
- **Emails are drafted, never sent.** Present them as copy-out text.
- **No auth beyond the shared key** — no users, roles, or audit trail.
