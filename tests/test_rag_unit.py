"""Unit tests for RAG components — no external dependencies required."""

import pytest


class TestDocumentExtractor:
    """Tests for PDF/DOCX text extraction."""

    def test_extract_text_txt_success(self):
        """Plain text files return content directly."""
        from app.services.document_extractor import extract_text

        content = b"I have been feeling hopeless for weeks."
        result = extract_text(content, "journal.txt")

        assert result is not None
        assert result.text == "I have been feeling hopeless for weeks."
        assert result.method == "text"

    def test_extract_text_md_success(self):
        """Markdown files return content directly."""
        from app.services.document_extractor import extract_text

        content = b"## Depression Overview\n\nDepression is a common condition."
        result = extract_text(content, "overview.md")

        assert result is not None
        assert "Depression" in result.text
        assert result.method == "text"

    def test_extract_text_corrupted_pdf(self):
        """Corrupted PDF returns None (graceful failure)."""
        from app.services.document_extractor import extract_text

        result = extract_text(b"not a real pdf", "broken.pdf")
        assert result is None

    def test_extract_text_empty_content(self):
        """Empty content returns None."""
        from app.services.document_extractor import extract_text

        result = extract_text(b"", "empty.pdf")
        assert result is None

    def test_extract_text_unsupported_format(self):
        """Unsupported format returns None."""
        from app.services.document_extractor import extract_text

        result = extract_text(b"some content", "data.xlsx")
        assert result is None

    def test_extract_text_csv_success(self):
        """CSV files return content as text."""
        from app.services.document_extractor import extract_text

        content = b"date,mood,score\n2026-04-01,sad,7\n2026-04-02,better,4"
        result = extract_text(content, "mood_log.csv")

        assert result is not None
        assert "mood" in result.text
        assert result.method == "text"
