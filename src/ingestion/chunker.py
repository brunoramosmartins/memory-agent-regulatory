"""Text chunking with overlap for the ingestion pipeline."""

from __future__ import annotations

import hashlib
import logging

from src.ingestion.models import Chunk, Document

logger = logging.getLogger(__name__)


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[Chunk]:
    """Split documents into overlapping text chunks.

    Uses character-based splitting (not token-based) for simplicity.
    Each chunk preserves source metadata from its parent Document.

    Args:
        documents: List of Document objects to chunk.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Overlap between consecutive chunks.

    Returns:
        List of Chunk objects ready for embedding.
    """
    chunks: list[Chunk] = []

    for doc in documents:
        text = doc.text.strip()
        if not text:
            continue

        doc_chunks = _split_text(text, chunk_size, chunk_overlap)

        for i, chunk_text in enumerate(doc_chunks):
            chunk_id = hashlib.sha256(
                f"{doc.document_id}:{i}:{chunk_text[:50]}".encode()
            ).hexdigest()[:16]

            chunks.append(
                Chunk(
                    text=chunk_text,
                    chunk_id=chunk_id,
                    document_id=doc.document_id,
                    page_number=doc.page_number,
                    chunk_index=i,
                    section_title=doc.section_title,
                    source_file=doc.source_uri,
                )
            )

    logger.info(
        "Chunked %d documents into %d chunks (size=%d, overlap=%d)",
        len(documents),
        len(chunks),
        chunk_size,
        chunk_overlap,
    )
    return chunks


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping windows.

    Tries to break at sentence boundaries (period + space) when possible.
    Guarantees forward progress to avoid infinite loops.
    """
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    parts: list[str] = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            # Try to break at a sentence boundary within the last 20% of the window
            min_break = start + int(chunk_size * 0.8)
            break_point = text.rfind(". ", min_break, end)
            if break_point > start:
                end = break_point + 1  # Include the period

        part = text[start:end].strip()
        if part:
            parts.append(part)

        # Guarantee forward progress: advance at least 1 character
        next_start = end - overlap
        if next_start <= start:
            next_start = start + max(chunk_size // 2, 1)
        start = next_start

    return parts
