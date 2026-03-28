"""Batch ingestion script — reads sources, chunks, embeds, and indexes in Weaviate.

Usage:
    python scripts/run_ingestion.py              # ingest all sources
    python scripts/run_ingestion.py --source pdf  # ingest only PDFs
    python scripts/run_ingestion.py --source web   # ingest only web sources
    python scripts/run_ingestion.py --dry-run      # extract + chunk without indexing
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.settings import get_settings
from src.ingestion.chunker import chunk_documents
from src.ingestion.source_registry import create_default_registry

logger = logging.getLogger(__name__)


def _embed_and_index(chunks: list, dry_run: bool = False) -> int:
    """Embed chunks with BGE-M3 and index them in Weaviate.

    Args:
        chunks: List of Chunk objects to embed and index.
        dry_run: If True, skip Weaviate indexing.

    Returns:
        Number of chunks indexed.
    """
    if not chunks:
        return 0

    if dry_run:
        logger.info("Dry run: skipping embedding and indexing for %d chunks", len(chunks))
        return 0

    from src.embeddings.embedding_generator import get_embedding_model
    from src.vectorstore.weaviate_client import (
        chunk_to_weaviate_properties,
        get_weaviate_client,
        init_chunk_collection,
    )

    # Initialize embedding model
    logger.info("Loading embedding model...")
    model = get_embedding_model()

    # Embed all chunk texts
    texts = [c.text for c in chunks]
    logger.info("Embedding %d chunks...", len(texts))
    t0 = time.perf_counter()
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    elapsed = time.perf_counter() - t0
    logger.info("Embedding complete in %.1fs (%.0f chunks/s)", elapsed, len(texts) / elapsed)

    # Connect to Weaviate and ensure collection exists
    client = get_weaviate_client()
    init_chunk_collection(client)

    # Batch index
    collection = client.collections.get("Chunk")
    logger.info("Indexing %d chunks in Weaviate...", len(chunks))
    t0 = time.perf_counter()
    indexed = 0

    with collection.batch.dynamic() as batch:
        for chunk, embedding in zip(chunks, embeddings, strict=False):
            props = chunk_to_weaviate_properties(chunk)
            props["alias"] = chunk.metadata.get("alias", "") if hasattr(chunk, "metadata") else ""
            batch.add_object(properties=props, vector=embedding.tolist())
            indexed += 1

    elapsed = time.perf_counter() - t0
    logger.info("Indexed %d chunks in %.1fs", indexed, elapsed)
    return indexed


def main(source_filter: str | None = None, dry_run: bool = False) -> None:
    """Run the ingestion pipeline for configured sources."""
    settings = get_settings()

    ingestion_cfg = getattr(settings, "ingestion", None)
    if ingestion_cfg is None:
        logger.error("No 'ingestion' section in config. Nothing to ingest.")
        sys.exit(1)

    sources = ingestion_cfg.sources
    if not sources:
        logger.error("No sources configured in ingestion.sources.")
        sys.exit(1)

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
    total_indexed = 0
    start = time.monotonic()

    all_chunks = []

    for source_cfg in sources:
        source_type = source_cfg.get("type", "unknown")
        alias = source_cfg.get("alias", "")
        logger.info("--- Ingesting: %s [%s] ---", alias or source_type, source_type)

        try:
            documents = registry.ingest(source_cfg)
            chunks = chunk_documents(
                documents,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            # Attach alias to chunks via source_file field
            for chunk in chunks:
                if alias and not chunk.source_file:
                    chunk.source_file = alias

            total_docs += len(documents)
            total_chunks += len(chunks)
            all_chunks.extend(chunks)

            logger.info(
                "  %s: %d documents -> %d chunks",
                alias or source_type,
                len(documents),
                len(chunks),
            )

        except FileNotFoundError as e:
            logger.warning("  Skipping (file not found): %s", e)
            continue
        except Exception:
            logger.exception("  Failed to ingest: %s", source_cfg)
            continue

    # Embed and index all chunks at once (batched is more efficient)
    if all_chunks:
        total_indexed = _embed_and_index(all_chunks, dry_run=dry_run)

    elapsed = time.monotonic() - start
    logger.info(
        "=== Ingestion complete: %d docs, %d chunks, %d indexed in %.1fs ===",
        total_docs,
        total_chunks,
        total_indexed,
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
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Extract and chunk only, skip embedding/indexing",
    )
    args = parser.parse_args()
    main(source_filter=args.source, dry_run=args.dry_run)
