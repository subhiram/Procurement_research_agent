"""Per-run records on disk.

This replaces the vendor cache, and the difference is the point. The cache made
a *reuse* decision the operator could not see: a later run silently received
vendors found weeks earlier, and the machinery needed to make that safe — TTLs,
alias distinctiveness rules, embedding-based material matching, contact
re-verification — was most of its complexity, all of it in service of a decision
nobody asked for.

An archive makes no decisions. Every completed run is written to one JSON file
and nothing in the graph ever reads it back, so a stored run cannot stand in for
a fresh search. Retrieval is an explicit CLI or API call by someone who wants
the old answer.

The derived fields (`summary`, `keywords`) are computed, not generated. They
could be written by a model; they are not, because a summary of a run is exactly
the kind of thing a model will embellish, and this file is meant to be a record
of what happened.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from procurement_agent.config import Settings, get_settings
from procurement_agent.designations import is_grade_specific, normalise

log = logging.getLogger(__name__)

#: Repeated verbatim into every archived run. These records carry real
#: companies' email addresses and phone numbers, and the file will outlive the
#: terminal session that produced it.
DISCLAIMER = (
    "Contact details were extracted from the linked source pages and verified "
    "against them at the time of the search; always confirm before use. "
    "Material may be subject to export control (ITAR/EAR)."
)

#: Version stamp, so a reader can tell an old record from a new one after the
#: shape changes. Absent from a file written before this existed.
SCHEMA_VERSION = 1


def _dump(value: Any) -> Any:
    """Pydantic model -> dict, anything else unchanged."""
    return value.model_dump(mode="json") if hasattr(value, "model_dump") else value


def build_keywords(state: dict) -> list[str]:
    """Terms this material was actually searched under.

    Drawn from what the run really used — the researched designations and
    synonyms plus the queries that were issued — rather than asked of a model,
    which would invent plausible-looking designations that were never searched.

    Family standards and bare ordinals are filtered out: "ASTM B348" covers
    every titanium bar grade and "Grade 5" means different alloys in different
    industries, so neither identifies what was looked for.
    """
    spec = state.get("material_spec")
    research = state.get("research")

    terms: list[str] = []
    if spec is not None:
        terms.append(spec.material_name)
        if spec.grade:
            terms.append(spec.grade)
    if research is not None:
        terms.append(research.canonical_name)
        terms += list(research.designations or [])
        terms += list(research.synonyms or [])

    seen: set[str] = set()
    keywords: list[str] = []
    for term in terms:
        term = (term or "").strip()
        key = normalise(term)
        if not key or key in seen or not is_grade_specific(term):
            continue
        seen.add(key)
        keywords.append(term)
    return keywords


def build_summary(state: dict) -> str:
    """A few plain sentences describing what this run did and found."""
    spec = state.get("material_spec")
    research = state.get("research")
    leads = state.get("vendor_leads", [])
    reachable = [lead for lead in leads if lead.email or lead.phone]

    lines: list[str] = []
    if spec is not None:
        wanted = spec.search_label()
        if spec.quantity and spec.unit:
            wanted += f", {spec.quantity:g} {spec.unit}"
        lines.append(f"Searched for {wanted}.")
    if research is not None:
        lines.append(
            f"Identified as {research.canonical_name} "
            f"({len(research.designations or [])} designation(s), "
            f"{len(research.synonyms or [])} synonym(s))."
        )
        if research.ambiguities:
            # Surfaced rather than buried: an unresolved ambiguity means the
            # whole vendor list may be for the wrong material.
            lines.append(
                f"UNRESOLVED AMBIGUITY: {'; '.join(research.ambiguities)}"
            )
    lines.append(
        f"Found {len(leads)} vendor(s), {len(reachable)} with a verified contact, "
        f"from {len(state.get('search_queries', []))} search queries."
    )
    if state.get("truncated"):
        lines.append("The search budget ran out, so this list is incomplete.")
    return " ".join(lines)


def run_id(thread_id: str, when: datetime | None = None) -> str:
    """Sortable, human-readable identifier: date first, then the thread."""
    when = when or datetime.now(UTC)
    return f"{when:%Y-%m-%d}-{thread_id}"


def build_record(
    state: dict,
    thread_id: str,
    settings: Settings | None = None,
    trace: dict | None = None,
) -> dict:
    """The archived shape of one run.

    `trace` is what the run actually did on the way to this result - node order,
    the model that served each call and what the routing ladder tried first, the
    search providers that answered and what they cost. Optional so a record can
    still be built for a run that had no tracing attached.
    """
    settings = settings or get_settings()
    now = datetime.now(UTC)

    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id(thread_id, now),
        "thread_id": thread_id,
        "saved_at": now.isoformat(),
        "request": state.get("raw_input", ""),
        "clarifications": state.get("clarification_history", []),
        "summary": build_summary(state),
        "keywords": build_keywords(state),
        "material_spec": _dump(state.get("material_spec")),
        "research": _dump(state.get("research")),
        "search_queries": state.get("search_queries", []),
        "sourcing_companies": state.get("sourcing_companies", []),
        # Deliberately without `raw_content`: those are whole scraped pages and
        # would turn a readable record into megabytes of markup per run.
        "search_results": [
            {
                "company_name": c.company_name,
                "url": c.url,
                "snippet": c.snippet,
                "query": c.query,
                "from_research": c.from_research,
            }
            for c in state.get("vendor_candidates", [])
        ],
        "vendor_leads": [_dump(lead) for lead in state.get("vendor_leads", [])],
        "credits_spent": max(
            0,
            settings.search_credits_per_session
            - state.get("search_credits_remaining", 0),
        ),
        "truncated": bool(state.get("truncated", False)),
        "totals": (trace or {}).get("totals", {}),
        "trace": (trace or {}).get("events", []),
        "disclaimer": DISCLAIMER,
    }


def save_run(
    state: dict,
    thread_id: str,
    settings: Settings | None = None,
    trace: dict | None = None,
) -> Path:
    """Write one run to `runs_dir`, returning the file path."""
    settings = settings or get_settings()
    record = build_record(state, thread_id, settings, trace)

    settings.runs_dir.mkdir(parents=True, exist_ok=True)
    path = settings.runs_dir / f"{record['run_id']}.json"
    # Written whole then moved, so a crash mid-write cannot leave a truncated
    # file that later fails to parse.
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2, ensure_ascii=False), "utf-8")
    temporary.replace(path)
    return path


def list_runs(settings: Settings | None = None) -> list[dict]:
    """Every archived run, newest first, as light index entries.

    Reads each file because the interesting fields are inside it; the archive is
    small by construction (one file per completed run), so this stays cheap
    enough not to need an index that could fall out of sync.
    """
    settings = settings or get_settings()
    if not settings.runs_dir.exists():
        return []

    entries: list[dict] = []
    for path in settings.runs_dir.glob("*.json"):
        try:
            record = json.loads(path.read_text("utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("archive: skipping unreadable run %s: %s", path.name, exc)
            continue
        entries.append(
            {
                "run_id": record.get("run_id", path.stem),
                "saved_at": record.get("saved_at", ""),
                "request": record.get("request", ""),
                "summary": record.get("summary", ""),
                "keywords": record.get("keywords", []),
                "vendor_count": len(record.get("vendor_leads", [])),
                "totals": record.get("totals", {}),
                "path": str(path),
            }
        )
    return sorted(entries, key=lambda e: e["saved_at"], reverse=True)


def load_run(identifier: str, settings: Settings | None = None) -> dict | None:
    """One archived run by id, or None if there is no such file."""
    settings = settings or get_settings()
    path = settings.runs_dir / f"{identifier}.json"
    if not path.exists():
        # Tolerate being given a bare thread id rather than a full run id, since
        # that is what the CLI prints when a session is created.
        matches = sorted(settings.runs_dir.glob(f"*{identifier}.json"))
        if not matches:
            return None
        path = matches[-1]
    try:
        return json.loads(path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("archive: could not read %s: %s", path, exc)
        return None


def search_runs(term: str, settings: Settings | None = None) -> list[dict]:
    """Archived runs whose request, summary or keywords mention `term`."""
    needle = normalise(term)
    if not needle:
        return []
    return [
        entry
        for entry in list_runs(settings)
        if needle in normalise(entry["request"])
        or needle in normalise(entry["summary"])
        or any(needle in normalise(k) for k in entry["keywords"])
    ]
