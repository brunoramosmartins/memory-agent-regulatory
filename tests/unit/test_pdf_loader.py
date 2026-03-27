"""Unit tests for the PDF loader."""

import pytest

from src.ingestion.pdf_loader import load_pdf


class TestLoadPdf:
    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_pdf(tmp_path / "nonexistent.pdf")

    def test_not_a_pdf(self, tmp_path):
        txt = tmp_path / "file.txt"
        txt.write_text("hello")
        with pytest.raises(ValueError, match="Not a PDF"):
            load_pdf(txt)

    def test_loads_pdf_pages(self, tmp_path):
        """Create a minimal PDF with PyMuPDF and verify extraction."""
        import fitz

        pdf_path = tmp_path / "test.pdf"
        doc = fitz.open()

        # Page 1
        page = doc.new_page()
        page.insert_text((72, 72), "Page one content")

        # Page 2
        page2 = doc.new_page()
        page2.insert_text((72, 72), "Page two content")

        doc.save(str(pdf_path))
        doc.close()

        results = load_pdf(pdf_path)

        assert len(results) == 2
        assert results[0].source_type == "pdf"
        assert results[0].page_number == 1
        assert "Page one" in results[0].text
        assert results[1].page_number == 2
        assert results[0].document_id != ""
        assert results[0].metadata["filename"] == "test.pdf"

    def test_skips_empty_pages(self, tmp_path):
        """Empty pages should be excluded."""
        import fitz

        pdf_path = tmp_path / "sparse.pdf"
        doc = fitz.open()
        doc.new_page()  # empty
        page = doc.new_page()
        page.insert_text((72, 72), "Only content")
        doc.new_page()  # empty
        doc.save(str(pdf_path))
        doc.close()

        results = load_pdf(pdf_path)
        assert len(results) == 1
        assert results[0].page_number == 2
