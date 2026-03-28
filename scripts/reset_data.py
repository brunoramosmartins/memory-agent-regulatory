"""Reset all data — clears PostgreSQL memory tables, Weaviate collections, and Phoenix traces.

Usage:
    python scripts/reset_data.py              # reset everything
    python scripts/reset_data.py --keep-chunks # keep ingested document chunks
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("reset_data")


def _reset_postgresql() -> None:
    """Delete all rows from memory tables."""
    from sqlalchemy import text

    from src.memory.database import get_engine

    engine = get_engine()
    tables = ["response_feedback", "session_summaries", "conversation_turns"]
    with engine.begin() as conn:
        for table in tables:
            try:
                conn.execute(text(f"DELETE FROM {table}"))  # noqa: S608
                logger.info("Cleared table: %s", table)
            except Exception as e:
                logger.warning("Table %s: %s", table, e)


def _reset_weaviate_memory() -> None:
    """Delete SemanticMemory and ProceduralMemory collections (recreate empty)."""
    from src.memory.procedural import PROCEDURAL_COLLECTION, init_procedural_collection
    from src.memory.semantic import SEMANTIC_COLLECTION, init_semantic_collection
    from src.vectorstore.weaviate_client import close_weaviate_client, get_weaviate_client

    client = get_weaviate_client()

    for name in [SEMANTIC_COLLECTION, PROCEDURAL_COLLECTION]:
        if client.collections.exists(name):
            client.collections.delete(name)
            logger.info("Deleted Weaviate collection: %s", name)

    # Recreate empty collections
    init_semantic_collection(client)
    init_procedural_collection(client)

    close_weaviate_client()


def _reset_weaviate_chunks() -> None:
    """Delete the Chunk collection (ingested documents)."""
    from src.vectorstore.weaviate_client import (
        CHUNK_COLLECTION,
        close_weaviate_client,
        get_weaviate_client,
        init_chunk_collection,
    )

    client = get_weaviate_client()

    if client.collections.exists(CHUNK_COLLECTION):
        client.collections.delete(CHUNK_COLLECTION)
        logger.info("Deleted Weaviate collection: %s", CHUNK_COLLECTION)

    init_chunk_collection(client)
    close_weaviate_client()


def _reset_phoenix() -> None:
    """Delete Phoenix local database to clear all traces."""
    import glob
    import os
    import shutil
    import tempfile

    # Phoenix stores data in ~/.phoenix or temp dirs
    phoenix_dirs = [
        os.path.expanduser("~/.phoenix"),
        os.path.join(tempfile.gettempdir(), "phoenix"),
    ]

    # Also look for phoenix.db in temp dirs
    for pattern in [os.path.join(tempfile.gettempdir(), "tmp*", "phoenix.db")]:
        phoenix_dirs.extend(glob.glob(pattern))

    cleared = False
    for path in phoenix_dirs:
        if os.path.exists(path):
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    os.unlink(path)
                logger.info("Removed Phoenix data: %s", path)
                cleared = True
            except Exception as e:
                logger.warning("Could not remove %s: %s", path, e)

    if not cleared:
        logger.info("No Phoenix data found to clear")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reset all project data")
    parser.add_argument(
        "--keep-chunks",
        action="store_true",
        help="Keep ingested document chunks in Weaviate",
    )
    args = parser.parse_args()

    print("=" * 50)
    print("Resetting project data...")
    print("=" * 50)

    logger.info("\n--- PostgreSQL ---")
    try:
        _reset_postgresql()
    except Exception as e:
        logger.error("PostgreSQL reset failed: %s", e)

    logger.info("\n--- Weaviate Memory ---")
    try:
        _reset_weaviate_memory()
    except Exception as e:
        logger.error("Weaviate memory reset failed: %s", e)

    if not args.keep_chunks:
        logger.info("\n--- Weaviate Chunks ---")
        try:
            _reset_weaviate_chunks()
        except Exception as e:
            logger.error("Weaviate chunks reset failed: %s", e)
    else:
        logger.info("Keeping ingested document chunks (--keep-chunks)")

    logger.info("\n--- Phoenix ---")
    try:
        _reset_phoenix()
    except Exception as e:
        logger.error("Phoenix reset failed: %s", e)

    print("\n" + "=" * 50)
    print("Data reset complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
