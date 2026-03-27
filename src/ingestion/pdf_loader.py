"""PDF document loader using PyMuPDF."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from src.ingestion.models import Document

logger = logging.getLogger(__name__)


def load_pdf(path: str | Path) -> list[Document]:
    """Extract text from a PDF file, one Document per page.

    Args:
        path: Path to the PDF file.

    Returns:
        List of Document objects, one per non-empty page.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a PDF.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Not a PDF file: {path}")

    import fitz  # PyMuPDF

    doc_id = hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:12]
    documents: list[Document] = []

    with fitz.open(path) as pdf:
        logger.info("Loading PDF %s (%d pages)", path.name, len(pdf))

        for page_num, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()
            if not text:
                continue

            documents.append(
                Document(
                    text=text,
                    source_type="pdf",
                    source_uri=str(path),
                    document_id=f"{doc_id}-p{page_num}",
                    page_number=page_num,
                    metadata={"filename": path.name, "total_pages": len(pdf)},
                )
            )

    logger.info("Extracted %d non-empty pages from %s", len(documents), path.name)
    return documents
