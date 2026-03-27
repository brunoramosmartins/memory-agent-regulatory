"""Batch ingestion script — reads sources from config.yaml and ingests all.

Usage:
    python scripts/run_ingestion.py              # ingest all sources
    python scripts/run_ingestion.py --source pdf  # ingest only PDFs
    python scripts/run_ingestion.py --source web   # ingest only web sources
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

# Ensure project root is on sys.path
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.settings import get_settings
from src.ingestion.chunker import chunk_documents
from src.ingestion.source_registry import create_default_registry

logger = logging.getLogger(__name__)


def main(source_filter: str | None = None) -> None:
    """Run the ingestion pipeline for configured sources."""
    settings = get_settings()

    # Read ingestion sources from config
    ingestion_cfg = getattr(settings, "ingestion", None)
    if ingestion_cfg is None:
        logger.error("No 'ingestion' section in config. Nothing to ingest.")
        sys.exit(1)

    sources = ingestion_cfg.sources
    if not sources:
        logger.error("No sources configured in ingestion.sources.")
        sys.exit(1)

    # Apply source filter if provided
    if source_filter:
        sources = [s for s in sources if s.get("type") == source_filter]
        if not sources:
            logger.error("No sources matching type=%s", source_filter)
            sys.exit(1)

    registry = create_default_registry()
    chunk_size = settings.chunking.chunk_size
    chunk_overlap = settings.chunking.chunk_overlap

    total_docs = 0
    total_chunks = 0
    start = time.monotonic()

    for source_cfg in sources:
        source_type = source_cfg.get("type", "unknown")
        logger.info("--- Ingesting source: %s ---", source_cfg)

        try:
            documents = registry.ingest(source_cfg)
            chunks = chunk_documents(
                documents,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            total_docs += len(documents)
            total_chunks += len(chunks)

            logger.info(
                "Source %s: %d documents, %d chunks",
                source_type,
                len(documents),
                len(chunks),
            )

            # TODO (Phase 4+): embed chunks and store in Weaviate
            # For now, log the results for validation
            for chunk in chunks[:3]:
                logger.debug(
                    "  chunk_id=%s doc=%s text=%s...",
                    chunk.chunk_id,
                    chunk.document_id,
                    chunk.text[:80],
                )

        except Exception:
            logger.exception("Failed to ingest source: %s", source_cfg)
            continue

    elapsed = time.monotonic() - start
    logger.info(
        "=== Ingestion complete: %d documents, %d chunks in %.1fs ===",
        total_docs,
        total_chunks,
        elapsed,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run ingestion pipeline")
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Filter by source type (e.g. 'pdf', 'web')",
    )
    args = parser.parse_args()
    main(source_filter=args.source)
