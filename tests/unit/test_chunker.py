"""Unit tests for the text chunker."""

from src.ingestion.chunker import _split_text, chunk_documents
from src.ingestion.models import Document


class TestSplitText:
    def test_short_text_single_chunk(self):
        result = _split_text("Hello world", chunk_size=100, overlap=10)
        assert result == ["Hello world"]

    def test_splits_with_overlap(self):
        text = "A" * 200
        parts = _split_text(text, chunk_size=100, overlap=20)
        assert len(parts) >= 2
        # Each chunk should be <= chunk_size
        for p in parts:
            assert len(p) <= 100

    def test_prefers_sentence_boundary(self):
        text = "First sentence. Second sentence. Third sentence here."
        parts = _split_text(text, chunk_size=35, overlap=5)
        # Should break at ". " when possible
        assert any(p.endswith(".") for p in parts[:-1])

    def test_empty_text(self):
        result = _split_text("", chunk_size=100, overlap=10)
        assert result == []


class TestChunkDocuments:
    def test_chunks_single_document(self):
        docs = [
            Document(
                text="Some text content for chunking purposes.",
                source_type="pdf",
                source_uri="/test.pdf",
                document_id="doc1",
                page_number=1,
            )
        ]
        chunks = chunk_documents(docs, chunk_size=500, chunk_overlap=50)
        assert len(chunks) == 1
        assert chunks[0].document_id == "doc1"
        assert chunks[0].page_number == 1
        assert chunks[0].chunk_index == 0
        assert chunks[0].chunk_id != ""

    def test_large_document_multiple_chunks(self):
        docs = [
            Document(
                text="Word " * 200,
                source_type="pdf",
                source_uri="/big.pdf",
                document_id="doc2",
                page_number=3,
            )
        ]
        chunks = chunk_documents(docs, chunk_size=100, chunk_overlap=20)
        assert len(chunks) > 1
        # All chunks share same document_id and page
        for c in chunks:
            assert c.document_id == "doc2"
            assert c.page_number == 3

    def test_empty_document_skipped(self):
        docs = [
            Document(text="", source_type="pdf", source_uri="/empty.pdf", document_id="e"),
            Document(text="  ", source_type="pdf", source_uri="/blank.pdf", document_id="b"),
        ]
        chunks = chunk_documents(docs)
        assert len(chunks) == 0

    def test_preserves_section_title(self):
        docs = [
            Document(
                text="Content here",
                source_type="web",
                source_uri="https://example.com",
                document_id="w1",
                section_title="FAQ Section",
            )
        ]
        chunks = chunk_documents(docs)
        assert chunks[0].section_title == "FAQ Section"

    def test_unique_chunk_ids(self):
        docs = [
            Document(text="A" * 300, source_type="pdf", source_uri="/a.pdf", document_id="d1"),
            Document(text="B" * 300, source_type="pdf", source_uri="/b.pdf", document_id="d2"),
        ]
        chunks = chunk_documents(docs, chunk_size=100, chunk_overlap=10)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))
