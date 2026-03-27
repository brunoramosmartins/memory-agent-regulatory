"""Unit tests for the source registry."""

import pytest

from src.ingestion.models import Document
from src.ingestion.source_registry import SourceRegistry, create_default_registry


def _fake_loader(**kwargs) -> list[Document]:
    return [
        Document(
            text=f"Content from {kwargs}",
            source_type="fake",
            source_uri="fake://test",
        )
    ]


class TestSourceRegistry:
    def test_register_and_ingest(self):
        registry = SourceRegistry()
        registry.register("fake", _fake_loader)

        docs = registry.ingest({"type": "fake", "param": "value"})
        assert len(docs) == 1
        assert "value" in docs[0].text

    def test_registered_types(self):
        registry = SourceRegistry()
        registry.register("pdf", _fake_loader)
        registry.register("web", _fake_loader)
        assert sorted(registry.registered_types) == ["pdf", "web"]

    def test_unknown_type_raises(self):
        registry = SourceRegistry()
        with pytest.raises(ValueError, match="Unknown source type"):
            registry.ingest({"type": "unknown"})

    def test_missing_type_raises(self):
        registry = SourceRegistry()
        with pytest.raises(ValueError, match="must have a 'type' key"):
            registry.ingest({"path": "/some/file"})

    def test_error_message_lists_registered(self):
        registry = SourceRegistry()
        registry.register("pdf", _fake_loader)
        with pytest.raises(ValueError, match="pdf"):
            registry.ingest({"type": "csv"})


class TestCreateDefaultRegistry:
    def test_has_pdf_and_web(self):
        registry = create_default_registry()
        assert "pdf" in registry.registered_types
        assert "web" in registry.registered_types
