"""Multi-source ingestion — PDF, web, and extensible source registry."""

from src.ingestion.chunker import chunk_documents
from src.ingestion.models import Chunk, Document
from src.ingestion.pdf_loader import load_pdf
from src.ingestion.source_registry import SourceRegistry, create_default_registry
from src.ingestion.web_scraper import scrape_url

__all__ = [
    "Chunk",
    "Document",
    "SourceRegistry",
    "chunk_documents",
    "create_default_registry",
    "load_pdf",
    "scrape_url",
]
