"""Contact-page discovery and the crawler's success rules.

Both encode behaviour learned from real vendor sites rather than from docs.
"""

from __future__ import annotations

from procurement_agent.crawl.contact_links import (
    find_contact_links,
    guess_contact_urls,
    looks_like_contact_page,
)
from procurement_agent.crawl.fetcher import CrawledPage


def link(href: str, text: str = "") -> dict:
    return {"href": href, "text": text}


class TestLooksLikeContactPage:
    def test_accepts_common_contact_paths(self):
        for url in (
            "https://v.example/contact",
            "https://v.example/contact-us",
            "https://v.example/about/contact",
            "https://v.example/kontakt",
            "https://v.example/request-a-quote",
        ):
            assert looks_like_contact_page(url), url

    def test_rejects_a_product_page_that_merely_starts_with_contact(self):
        """`/contact-lens-alloys` is a product page. Matching on whole path
        segments rather than substrings is what makes this work."""
        assert not looks_like_contact_page("https://v.example/contact-lens-alloys")

    def test_rejects_unrelated_pages(self):
        for url in (
            "https://v.example/products/stainless",
            "https://v.example/privacy",
            "https://v.example/cart",
            "https://v.example/brochure.pdf",
        ):
            assert not looks_like_contact_page(url), url

    def test_accepts_on_anchor_text_when_the_url_is_opaque(self):
        assert looks_like_contact_page("https://v.example/p/42", "Contact us")

    def test_anchor_text_does_not_rescue_a_deep_url(self):
        assert not looks_like_contact_page(
            "https://v.example/a/b/c/d/e", "Contact us"
        )


class TestFindContactLinks:
    def test_finds_and_ranks_best_first(self):
        links = [
            link("/about", "About"),
            link("/contact", "Contact"),
            link("/products/bar", "Bar"),
        ]
        found = find_contact_links(links, "https://v.example/products/custom-465")
        assert found[0] == "https://v.example/contact"

    def test_stays_on_the_same_domain(self):
        """A marketplace profile is not this vendor's contact page, and
        following it would attribute someone else's email to them."""
        links = [link("https://alibaba.com/contact", "Contact")]
        assert find_contact_links(links, "https://v.example/p") == []

    def test_resolves_relative_hrefs(self):
        links = [link("../contact-us", "Contact")]
        found = find_contact_links(links, "https://v.example/products/x")
        assert found == ["https://v.example/contact-us"]

    def test_skips_mailto_and_tel(self):
        links = [link("mailto:a@v.example"), link("tel:+441215550147")]
        assert find_contact_links(links, "https://v.example/p") == []

    def test_deduplicates_trailing_slash_and_fragment(self):
        links = [
            link("/contact", "Contact"),
            link("/contact/", "Contact"),
            link("/contact#form", "Contact"),
        ]
        assert len(find_contact_links(links, "https://v.example/p")) == 1

    def test_excludes_the_page_we_already_have(self):
        links = [link("/contact", "Contact")]
        assert find_contact_links(links, "https://v.example/contact") == []

    def test_respects_the_limit(self):
        links = [link(f"/contact-{i}", "Contact") for i in range(10)]
        links.append(link("/contact", "Contact"))
        assert len(find_contact_links(links, "https://v.example/p", limit=2)) == 2

    def test_prefers_shallow_paths(self):
        links = [link("/a/b/c/contact", "Contact"), link("/contact", "Contact")]
        found = find_contact_links(links, "https://v.example/p")
        assert found[0] == "https://v.example/contact"


class TestGuessContactUrls:
    def test_builds_conventional_urls_from_the_root(self):
        guessed = guess_contact_urls("https://v.example/products/deep/page")
        assert guessed[0] == "https://v.example/contact"


class TestCrawledPage:
    def test_thin_content_is_not_useful(self):
        """A bot-blocked page returned success=True, status 418 and 92 chars,
        so usefulness is judged on content length, never on the success flag."""
        assert not CrawledPage(url="u", content="x" * 92).is_useful

    def test_substantial_content_is_useful(self):
        assert CrawledPage(url="u", content="x" * 5000).is_useful

    def test_a_redirect_status_does_not_mean_failure(self):
        """Two real vendor sites returned 301 with complete content."""
        page = CrawledPage(url="u", content="x" * 5000, status_code=301)
        assert page.is_useful

    def test_an_explicit_failure_is_never_useful(self):
        page = CrawledPage(url="u", content="x" * 5000, failed=True, error="boom")
        assert not page.is_useful
