"""The per-run archive.

Two properties matter and neither is about storage. First, the record has to be
honest — derived from what the run actually did, never from a model, because a
generated summary of a run is exactly the kind of thing that gets embellished.
Second, it must stay a record: nothing in the graph may read it back, or the
cache this replaced has quietly returned under another name.
"""

from __future__ import annotations

import json

import pytest

from procurement_agent.archive import (
    DISCLAIMER,
    build_keywords,
    build_record,
    build_summary,
    list_runs,
    load_run,
    save_run,
    search_runs,
)
from procurement_agent.graph.state import (
    MaterialResearch,
    MaterialSpec,
    VendorCandidate,
    VendorLead,
)


@pytest.fixture
def settings(tmp_path):
    from procurement_agent.config import get_settings

    settings = get_settings()
    original = settings.runs_dir
    settings.runs_dir = tmp_path / "runs"
    yield settings
    settings.runs_dir = original


@pytest.fixture
def state():
    return {
        "raw_input": "Custom 465 Dia 2 inch - 200 KG",
        "material_spec": MaterialSpec(
            material_name="Custom 465", grade="UNS S46500", form="round bar",
            quantity=200, unit="KG",
        ),
        "research": MaterialResearch(
            canonical_name="Custom 465 stainless steel",
            designations=["UNS S46500", "AMS 5936"],
            synonyms=["Custom 465"],
            ambiguities=[],
        ),
        "search_queries": ["Custom 465 supplier", "UNS S46500 supplier stockist"],
        "vendor_candidates": [
            VendorCandidate(
                company_name="Precision Alloys Ltd",
                url="https://precisionalloys.co.uk/custom-465",
                snippet="We stock Custom 465 round bar",
                raw_content="x" * 50_000,
                query="Custom 465 supplier",
            )
        ],
        "vendor_leads": [
            VendorLead(
                company_name="Precision Alloys Ltd",
                website="https://precisionalloys.co.uk",
                email="sales@precisionalloys.co.uk",
                source_url="https://precisionalloys.co.uk/custom-465",
            )
        ],
        "search_credits_remaining": 4,
        "truncated": False,
    }


class TestKeywords:
    def test_collects_the_terms_the_run_actually_used(self, state):
        keywords = build_keywords(state)
        assert "Custom 465" in keywords
        assert "UNS S46500" in keywords
        assert "AMS 5936" in keywords

    def test_drops_family_standards(self, state):
        """"ASTM B348" covers every titanium bar grade. Keeping it would make a
        later search for one alloy match a record about a different one."""
        state["research"].designations = ["UNS S46500", "ASTM B348"]
        assert "ASTM B348" not in build_keywords(state)

    def test_drops_bare_ordinals(self, state):
        """"Grade 5" is Ti-6Al-4V in titanium and something else entirely in
        fasteners."""
        state["research"].designations = ["Grade 5"]
        assert "Grade 5" not in build_keywords(state)

    def test_deduplicates_across_spelling(self, state):
        """"Custom 465", "custom-465" and "CUSTOM 465" are one search term."""
        state["research"].synonyms = ["Custom 465", "custom-465", "CUSTOM 465"]
        keywords = build_keywords(state)
        variants = [k for k in keywords if k.lower().replace("-", " ") == "custom 465"]
        assert len(variants) == 1

    def test_survives_a_run_with_no_research(self, state):
        state["research"] = None
        assert build_keywords(state)


class TestSummary:
    def test_reports_what_was_found(self, state):
        summary = build_summary(state)
        assert "Custom 465" in summary
        assert "1 vendor" in summary

    def test_surfaces_an_unresolved_ambiguity(self, state):
        """An ambiguity means the whole vendor list may be for the wrong
        material, so it cannot be buried in a field nobody reads."""
        state["research"].ambiguities = ["Could be the Carpenter alloy or the Sandvik one"]
        assert "UNRESOLVED AMBIGUITY" in build_summary(state)

    def test_says_when_the_list_is_incomplete(self, state):
        state["truncated"] = True
        assert "incomplete" in build_summary(state)


class TestRecord:
    def test_omits_scraped_page_bodies(self, state, settings):
        """raw_content is whole scraped pages; including it would turn a
        readable record into megabytes of markup per run."""
        record = build_record(state, "thread-1", settings)
        serialised = json.dumps(record)
        assert "x" * 1000 not in serialised
        assert len(serialised) < 20_000

    def test_keeps_the_search_results_themselves(self, state, settings):
        record = build_record(state, "thread-1", settings)
        assert record["search_results"][0]["url"].endswith("/custom-465")

    def test_carries_the_disclaimer(self, state, settings):
        """These files hold real companies' contact details and outlive the
        session that produced them."""
        assert build_record(state, "thread-1", settings)["disclaimer"] == DISCLAIMER

    def test_records_credits_spent(self, state, settings):
        record = build_record(state, "thread-1", settings)
        assert record["credits_spent"] == settings.search_credits_per_session - 4


class TestRoundTrip:
    def test_saves_and_loads_by_run_id(self, state, settings):
        path = save_run(state, "thread-1", settings)
        assert path.exists()

        record = load_run(path.stem, settings)
        assert record["request"] == state["raw_input"]
        assert record["vendor_leads"][0]["email"] == "sales@precisionalloys.co.uk"

    def test_loads_by_bare_thread_id(self, state, settings):
        """The CLI prints a thread id when a session is created, so that is
        what someone will have to hand later."""
        save_run(state, "thread-1", settings)
        assert load_run("thread-1", settings) is not None

    def test_missing_run_returns_none_rather_than_raising(self, settings):
        assert load_run("nope", settings) is None

    def test_leaves_no_partial_file_behind(self, state, settings):
        save_run(state, "thread-1", settings)
        assert not list(settings.runs_dir.glob("*.tmp"))

    def test_lists_runs_newest_first(self, state, settings):
        save_run(state, "thread-a", settings)
        save_run(state, "thread-b", settings)
        entries = list_runs(settings)
        assert len(entries) == 2
        assert entries[0]["saved_at"] >= entries[1]["saved_at"]

    def test_skips_an_unreadable_file_rather_than_failing(self, state, settings):
        save_run(state, "thread-1", settings)
        (settings.runs_dir / "corrupt.json").write_text("{not json")
        assert len(list_runs(settings)) == 1

    def test_listing_an_empty_archive_is_not_an_error(self, settings):
        assert list_runs(settings) == []

    def test_searches_by_keyword_and_request(self, state, settings):
        save_run(state, "thread-1", settings)
        assert search_runs("uns s46500", settings)
        assert search_runs("custom 465", settings)
        assert search_runs("inconel", settings) == []


class TestItIsNotACache:
    def test_the_graph_never_reads_the_archive(self):
        """The point of replacing the cache. If any node imported a read, a
        stored run could silently stand in for a fresh search again.
        """
        import pathlib

        nodes = pathlib.Path("src/procurement_agent/graph/nodes")
        offenders = [
            path.name
            for path in nodes.glob("*.py")
            if any(
                fn in path.read_text()
                for fn in ("load_run", "list_runs", "search_runs")
            )
        ]
        assert offenders == []
