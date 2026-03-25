"""Lightweight retrieval models (no heavy dependencies)."""

from dataclasses import dataclass


@dataclass
class RetrievalResult:
    """Single retrieval result with metadata."""

    text: str
    chunk_id: str
    document_id: str
    page_number: int
    section_title: str | None
    similarity_score: float | None
    source_file: str | None = None
