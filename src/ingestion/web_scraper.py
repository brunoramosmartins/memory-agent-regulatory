"""Web scraper for FAQ pages and regulatory websites."""

from __future__ import annotations

import hashlib
import logging
import time

import httpx
from bs4 import BeautifulSoup, Tag

from src.ingestion.models import Document

logger = logging.getLogger(__name__)

# Elements that carry no useful content
_STRIP_TAGS = {"nav", "header", "footer", "script", "style", "aside", "form"}

# Heading tags used for section splitting
_HEADING_TAGS = {"h1", "h2", "h3"}


def _clean_html(soup: BeautifulSoup) -> BeautifulSoup:
    """Remove navigation, scripts, and other non-content elements."""
    for tag in soup.find_all(_STRIP_TAGS):
        tag.decompose()
    return soup


def _split_by_headings(soup: BeautifulSoup) -> list[tuple[str | None, str]]:
    """Split page content into sections based on heading tags.

    Returns a list of (heading_text, body_text) tuples.
    """
    sections: list[tuple[str | None, str]] = []
    current_heading: str | None = None
    current_parts: list[str] = []

    body = soup.find("body") or soup
    for element in body.children:
        if not isinstance(element, Tag):
            if hasattr(element, "get_text"):
                text = element.get_text(strip=True)
            else:
                text = str(element).strip()
            if text:
                current_parts.append(text)
            continue

        if element.name in _HEADING_TAGS:
            # Flush previous section
            if current_parts:
                sections.append((current_heading, "\n".join(current_parts)))
            current_heading = element.get_text(strip=True)
            current_parts = []
        else:
            text = element.get_text(separator="\n", strip=True)
            if text:
                current_parts.append(text)

    # Flush last section
    if current_parts:
        sections.append((current_heading, "\n".join(current_parts)))

    return sections


def scrape_url(
    url: str,
    timeout: float = 30.0,
    delay: float = 0.0,
) -> list[Document]:
    """Fetch and parse a web page into Document objects.

    Splits the page by headings (h1-h3), stripping navigation and boilerplate.

    Args:
        url: URL to scrape.
        timeout: HTTP request timeout in seconds.
        delay: Delay in seconds before the request (rate limiting).

    Returns:
        List of Document objects, one per heading section.

    Raises:
        httpx.HTTPStatusError: On non-2xx responses.
    """
    if delay > 0:
        time.sleep(delay)

    logger.info("Scraping %s", url)

    response = httpx.get(url, timeout=timeout, follow_redirects=True)
    response.raise_for_status()

    content_type = response.headers.get("content-type", "")
    if "text/html" not in content_type:
        logger.warning("Non-HTML content at %s: %s", url, content_type)
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    soup = _clean_html(soup)
    sections = _split_by_headings(soup)

    if not sections:
        # Fallback: treat entire body as one document
        full_text = soup.get_text(separator="\n", strip=True)
        if full_text:
            sections = [(None, full_text)]

    url_hash = hashlib.sha256(url.encode()).hexdigest()[:12]
    documents: list[Document] = []

    for i, (heading, text) in enumerate(sections):
        if not text.strip():
            continue

        documents.append(
            Document(
                text=text,
                source_type="web",
                source_uri=url,
                document_id=f"{url_hash}-s{i}",
                page_number=0,
                section_title=heading,
                metadata={"url": url, "section_index": i},
            )
        )

    logger.info("Extracted %d sections from %s", len(documents), url)
    return documents
