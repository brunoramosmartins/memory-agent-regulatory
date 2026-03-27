"""Unit tests for the web scraper using HTML fixtures."""

import httpx
import pytest
from bs4 import BeautifulSoup

from src.ingestion.web_scraper import _clean_html, _split_by_headings, scrape_url

# ---------------------------------------------------------------------------
# HTML fixtures
# ---------------------------------------------------------------------------

SIMPLE_HTML = """
<html>
<body>
    <nav>Navigation links</nav>
    <h1>FAQ Title</h1>
    <p>Welcome to our FAQ page.</p>
    <h2>What is PIX?</h2>
    <p>PIX is an instant payment system.</p>
    <h2>Are there fees?</h2>
    <p>No fees for individuals.</p>
    <footer>Footer content</footer>
</body>
</html>
"""

NO_HEADINGS_HTML = """
<html>
<body>
    <p>Just a paragraph of text without any headings.</p>
    <p>Another paragraph here.</p>
</body>
</html>
"""

EMPTY_HTML = """
<html>
<body>
    <nav>Only navigation</nav>
    <script>console.log('js');</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# _clean_html tests
# ---------------------------------------------------------------------------


class TestCleanHtml:
    def test_removes_nav_footer_script(self):
        soup = BeautifulSoup(SIMPLE_HTML, "html.parser")
        cleaned = _clean_html(soup)
        assert cleaned.find("nav") is None
        assert cleaned.find("footer") is None

    def test_preserves_content(self):
        soup = BeautifulSoup(SIMPLE_HTML, "html.parser")
        cleaned = _clean_html(soup)
        assert cleaned.find("h1") is not None
        assert "PIX is an instant payment" in cleaned.get_text()


# ---------------------------------------------------------------------------
# _split_by_headings tests
# ---------------------------------------------------------------------------


class TestSplitByHeadings:
    def test_splits_on_headings(self):
        soup = BeautifulSoup(SIMPLE_HTML, "html.parser")
        _clean_html(soup)
        sections = _split_by_headings(soup)

        assert len(sections) >= 2
        # First section after h1
        headings = [h for h, _ in sections if h]
        assert "FAQ Title" in headings

    def test_no_headings_single_section(self):
        soup = BeautifulSoup(NO_HEADINGS_HTML, "html.parser")
        sections = _split_by_headings(soup)

        assert len(sections) >= 1
        assert any("paragraph" in text for _, text in sections)

    def test_empty_body(self):
        soup = BeautifulSoup(EMPTY_HTML, "html.parser")
        _clean_html(soup)
        sections = _split_by_headings(soup)
        # After stripping nav/script, may be empty
        content = "".join(text for _, text in sections).strip()
        assert content == ""


# ---------------------------------------------------------------------------
# scrape_url tests (with mocked HTTP)
# ---------------------------------------------------------------------------


class TestScrapeUrl:
    def test_scrapes_html(self, monkeypatch):
        def mock_get(url, **kwargs):
            resp = httpx.Response(
                200,
                text=SIMPLE_HTML,
                headers={"content-type": "text/html"},
                request=httpx.Request("GET", url),
            )
            return resp

        monkeypatch.setattr(httpx, "get", mock_get)

        docs = scrape_url("https://example.com/faq")

        assert len(docs) >= 2
        assert docs[0].source_type == "web"
        assert docs[0].source_uri == "https://example.com/faq"
        assert any("PIX" in d.text for d in docs)

    def test_non_html_returns_empty(self, monkeypatch):
        def mock_get(url, **kwargs):
            return httpx.Response(
                200,
                text="binary data",
                headers={"content-type": "application/pdf"},
                request=httpx.Request("GET", url),
            )

        monkeypatch.setattr(httpx, "get", mock_get)

        docs = scrape_url("https://example.com/file.pdf")
        assert docs == []

    def test_http_error_raises(self, monkeypatch):
        def mock_get(url, **kwargs):
            resp = httpx.Response(
                404,
                request=httpx.Request("GET", url),
            )
            resp.raise_for_status()

        monkeypatch.setattr(httpx, "get", mock_get)

        with pytest.raises(httpx.HTTPStatusError):
            scrape_url("https://example.com/404")
