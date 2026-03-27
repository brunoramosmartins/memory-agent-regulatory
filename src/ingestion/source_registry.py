"""Source registry — routes source types to the appropriate loader."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from src.ingestion.models import Document

logger = logging.getLogger(__name__)

# Type alias for a loader function
LoaderFn = Callable[..., list[Document]]


class SourceRegistry:
    """Registry that maps source types to loader functions.

    Usage::

        registry = SourceRegistry()
        registry.register("pdf", load_pdf)
        registry.register("web", scrape_url)

        docs = registry.ingest({"type": "pdf", "path": "/data/doc.pdf"})
    """

    def __init__(self) -> None:
        self._loaders: dict[str, LoaderFn] = {}

    def register(self, source_type: str, loader: LoaderFn) -> None:
        """Register a loader function for a source type."""
        self._loaders[source_type] = loader
        logger.debug("Registered loader for source_type=%s", source_type)

    @property
    def registered_types(self) -> list[str]:
        """Return list of registered source types."""
        return list(self._loaders.keys())

    def ingest(self, source_config: dict[str, Any]) -> list[Document]:
        """Ingest a single source using its registered loader.

        Args:
            source_config: Dict with at least a ``type`` key.
                - For ``pdf``: requires ``path``.
                - For ``web``: requires ``url``, optional ``delay``.

        Returns:
            List of Document objects from the loader.

        Raises:
            ValueError: If ``type`` is missing or not registered.
        """
        source_type = source_config.get("type")
        if not source_type:
            raise ValueError("Source config must have a 'type' key")

        loader = self._loaders.get(source_type)
        if loader is None:
            raise ValueError(
                f"Unknown source type: {source_type!r}. "
                f"Registered: {self.registered_types}"
            )

        # Build kwargs from config (excluding 'type')
        kwargs = {k: v for k, v in source_config.items() if k != "type"}

        logger.info("Ingesting source_type=%s config=%s", source_type, kwargs)
        documents = loader(**kwargs)
        logger.info(
            "Ingested %d documents from source_type=%s",
            len(documents),
            source_type,
        )
        return documents


def create_default_registry() -> SourceRegistry:
    """Create a registry with built-in loaders for pdf and web."""
    from src.ingestion.pdf_loader import load_pdf
    from src.ingestion.web_scraper import scrape_url

    registry = SourceRegistry()
    registry.register("pdf", load_pdf)
    registry.register("web", scrape_url)
    return registry
