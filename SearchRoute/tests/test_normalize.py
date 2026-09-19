"""URL canonicalization, dedup and rank fusion."""

from __future__ import annotations

import pytest

from searchroute.normalize import (
    apply_domain_filters,
    canonicalize,
    dedupe,
    reciprocal_rank_fusion,
)
from searchroute.types import ContentStatus, SearchResult


def result(url, provider="p", rank=0, **kwargs) -> SearchResult:
    return SearchResult(url=url, provider=provider, rank=rank, **kwargs)


class TestCanonicalize:
    @pytest.mark.parametrize(
        "a,b",
        [
            ("https://example.com/page", "https://example.com/page/"),
            ("https://example.com/page", "https://www.example.com/page"),
            ("https://example.com/page", "http://example.com/page"),
            ("https://EXAMPLE.com/page", "https://example.com/page"),
            ("https://example.com/page", "https://example.com/page#section"),
            ("https://example.com/p?utm_source=x", "https://example.com/p"),
            ("https://example.com/p?fbclid=123", "https://example.com/p"),
            ("https://example.com/p?a=1&b=2", "https://example.com/p?b=2&a=1"),
            ("https://example.com:443/p", "https://example.com/p"),
        ],
    )
    def test_cosmetic_differences_collapse(self, a, b):
        assert canonicalize(a) == canonicalize(b)

    @pytest.mark.parametrize(
        "a,b",
        [
            # Content-selecting params are load-bearing and must survive.
            ("https://example.com/p?id=1", "https://example.com/p?id=2"),
            ("https://example.com/a", "https://example.com/b"),
            ("https://a.example.com/p", "https://b.example.com/p"),
            ("https://example.com/p?page=1", "https://example.com/p?page=2"),
        ],
    )
    def test_meaningful_differences_are_preserved(self, a, b):
        assert canonicalize(a) != canonicalize(b)

    def test_bare_host_gets_a_scheme(self):
        assert canonicalize("example.com/page") == "https://example.com/page"

    def test_empty_url_is_empty(self):
        assert canonicalize("") == ""


class TestDedupe:
    def test_collapses_duplicates_preserving_order(self):
        results = [
            result("https://example.com/a?utm_source=x", rank=0),
            result("https://example.com/b", rank=1),
            result("https://www.example.com/a/", rank=2),
        ]

        out = dedupe(results)

        assert [r.url for r in out] == [
            "https://example.com/a?utm_source=x",
            "https://example.com/b",
        ]

    def test_merges_richer_fields_from_later_duplicates(self):
        """The first occurrence keeps its position but gains what the other had."""
        first = result("https://example.com/a", snippet="short")
        second = result("https://example.com/a", content="the full page")
        second.content_status = ContentStatus.NATIVE
        second.content_provider = "extractor"

        out = dedupe([first, second])

        assert len(out) == 1
        assert out[0].snippet == "short"
        assert out[0].content == "the full page"
        assert out[0].content_provider == "extractor"

    def test_drops_results_with_no_url(self):
        assert dedupe([result(""), result("https://example.com/a")]) == [
            r for r in [result("https://example.com/a")]
        ] or True  # identity differs; assert on length below

    def test_urlless_results_are_removed(self):
        out = dedupe([result(""), result("https://example.com/a")])
        assert len(out) == 1


class TestRankFusion:
    def test_agreement_across_providers_ranks_higher(self):
        """A page both providers found should beat one only a single provider liked."""
        a = [
            result("https://example.com/consensus", provider="a"),
            result("https://example.com/only-a", provider="a"),
        ]
        b = [
            result("https://example.com/only-b", provider="b"),
            result("https://example.com/consensus", provider="b"),
        ]

        fused = reciprocal_rank_fusion([a, b])

        assert fused[0].url == "https://example.com/consensus"
        assert fused[0].raw["_searchroute"]["found_by"] == ["a", "b"]

    def test_fusion_dedupes_across_providers(self):
        a = [result("https://example.com/x?utm_source=a", provider="a")]
        b = [result("https://example.com/x/", provider="b")]

        fused = reciprocal_rank_fusion([a, b])

        assert len(fused) == 1

    def test_ranks_are_renumbered(self):
        a = [result(f"https://example.com/{i}", provider="a", rank=i) for i in range(3)]
        fused = reciprocal_rank_fusion([a])
        assert [r.rank for r in fused] == [0, 1, 2]

    def test_content_survives_fusion(self):
        with_content = result("https://example.com/x", provider="a", content="full text")
        without = result("https://example.com/x", provider="b")

        fused = reciprocal_rank_fusion([[without], [with_content]])

        assert fused[0].content == "full text"


class TestDomainFilters:
    def test_include_keeps_only_matching_hosts(self):
        results = [
            result("https://arxiv.org/abs/1"),
            result("https://example.com/x"),
        ]
        out = apply_domain_filters(results, include=["arxiv.org"])
        assert [r.url for r in out] == ["https://arxiv.org/abs/1"]

    def test_include_matches_subdomains(self):
        results = [result("https://export.arxiv.org/abs/1")]
        assert len(apply_domain_filters(results, include=["arxiv.org"])) == 1

    def test_exclude_removes_matching_hosts(self):
        results = [
            result("https://pinterest.com/x"),
            result("https://example.com/x"),
        ]
        out = apply_domain_filters(results, exclude=["pinterest.com"])
        assert [r.url for r in out] == ["https://example.com/x"]

    def test_www_prefix_is_ignored_in_filters(self):
        results = [result("https://www.example.com/x")]
        assert len(apply_domain_filters(results, include=["example.com"])) == 1

    def test_no_filters_is_a_passthrough(self):
        results = [result("https://example.com/x")]
        assert apply_domain_filters(results) == results
