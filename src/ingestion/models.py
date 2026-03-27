"""Document and Chunk models for the ingestion pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Document:
    """A single document extracted by a loader.

    Represents a logical unit of content (e.g. one PDF page, one web section)
    before chunking.
    """

    text: str
    source_type: str  # "pdf", "web"
    source_uri: str  # file path or URL
    document_id: str = ""
    page_number: int = 0
    section_title: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class Chunk:
    """A chunk ready for embedding and indexing in Weaviate.

    Produced by splitting a Document's text with overlap.
    """

    text: str
    chunk_id: str
    document_id: str
    page_number: int
    chunk_index: int
    segment_index: int = 0
    section_title: str | None = None
    article_numbers: list[str] = field(default_factory=list)
    source_file: str = ""
