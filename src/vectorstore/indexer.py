"""Chunk indexing pipeline for Weaviate vector database.

NOTE: This module depends on src.chunking which is not yet implemented.
It is included for structural completeness and will be updated in later phases.
"""

import logging

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 100
EMBEDDING_BATCH_SIZE = 32
