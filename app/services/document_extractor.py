"""
Multi-format document text extraction.

Primary engine: Docling (layout-aware, handles tables, complex PDFs, DOCX).
Fallback engine: pdfplumber (fast, text-native PDFs).

Design principles:
- Never raises exceptions — every public function returns None on failure.
- Logs warnings on degraded paths so operators can track quality.
- Docling is an optional dependency; if not installed, pdfplumber is used
  transparently for PDFs.
- All file I/O is in-memory except Docling's temp-file requirement for PDFs
  (Docling's converter currently requires a path, not a stream).
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Supported extensions → handler routing
_PDF_EXTS = {".pdf"}
_DOCX_EXTS = {".docx"}
_TEXT_EXTS = {".txt", ".md", ".csv"}


@dataclass
class ExtractionResult:
    """Outcome of a document text extraction attempt."""

    text: str
    method: str  # "docling" | "pdfplumber" | "text"
    page_count: int | None = None
    has_tables: bool = False


# ── Public API ────────────────────────────────────────────────────────────────


def extract_text(raw_bytes: bytes, filename: str) -> ExtractionResult | None:
    """Extract text from a document given its raw bytes and filename.

    Routes by file extension:
      .pdf          → _extract_pdf  (Docling → pdfplumber fallback)
      .docx         → _extract_docx (Docling)
      .txt / .md / .csv → _extract_text_file (UTF-8 decode)

    Returns None for empty input, unsupported formats, or unrecoverable errors.
    """
    if not raw_bytes:
        return None

    ext = os.path.splitext(filename.lower())[1]

    if ext in _PDF_EXTS:
        return _extract_pdf(raw_bytes, filename)
    if ext in _DOCX_EXTS:
        return _extract_docx(raw_bytes, filename)
    if ext in _TEXT_EXTS:
        return _extract_text_file(raw_bytes)

    logger.warning("document_extractor: unsupported file type '%s' for file '%s'", ext, filename)
    return None


# ── Private helpers ───────────────────────────────────────────────────────────


def _extract_pdf(raw_bytes: bytes, filename: str) -> ExtractionResult | None:
    """Try Docling first; fall back to pdfplumber on any failure."""
    result = _docling_pdf(raw_bytes, filename)
    if result is not None:
        return result

    logger.info("document_extractor: Docling unavailable or failed for '%s', trying pdfplumber", filename)
    return _pdfplumber_pdf(raw_bytes, filename)


def _docling_pdf(raw_bytes: bytes, filename: str) -> ExtractionResult | None:
    """Attempt PDF extraction via Docling.

    Docling requires a file path, so we write bytes to a NamedTemporaryFile,
    convert, and clean up regardless of outcome.
    """
    try:
        from docling.document_converter import DocumentConverter  # type: ignore[import]
    except ImportError:
        logger.debug("document_extractor: Docling not installed, skipping")
        return None

    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(raw_bytes)
            tmp_path = tmp.name

        converter = DocumentConverter()
        conversion = converter.convert(tmp_path)
        doc = conversion.document

        text = doc.export_to_markdown()
        if not text or not text.strip():
            logger.warning("document_extractor: Docling returned empty text for '%s'", filename)
            return None

        # Detect table presence: Docling markdown uses pipe-table syntax
        has_tables = "|" in text

        page_count: int | None = None
        try:
            page_count = len(doc.pages) if hasattr(doc, "pages") else None
        except Exception:
            pass

        return ExtractionResult(
            text=text.strip(),
            method="docling",
            page_count=page_count,
            has_tables=has_tables,
        )

    except Exception as exc:
        logger.warning("document_extractor: Docling extraction failed for '%s': %s", filename, exc)
        return None

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def _pdfplumber_pdf(raw_bytes: bytes, filename: str) -> ExtractionResult | None:
    """Extract text from a PDF using pdfplumber."""
    try:
        import io

        import pdfplumber  # type: ignore[import]

        with pdfplumber.open(io.BytesIO(raw_bytes)) as pdf:
            pages_text: list[str] = []
            has_tables = False

            for page in pdf.pages:
                text = (page.extract_text() or "").strip()
                if text:
                    pages_text.append(text)
                if not has_tables and page.extract_tables():
                    has_tables = True

            if not pages_text:
                logger.warning(
                    "document_extractor: pdfplumber found no extractable text in '%s'", filename
                )
                return None

            return ExtractionResult(
                text="\n\n---\n\n".join(pages_text),
                method="pdfplumber",
                page_count=len(pdf.pages),
                has_tables=has_tables,
            )

    except Exception as exc:
        logger.warning("document_extractor: pdfplumber extraction failed for '%s': %s", filename, exc)
        return None


def _extract_docx(raw_bytes: bytes, filename: str) -> ExtractionResult | None:
    """Extract text from a DOCX file via Docling."""
    try:
        from docling.document_converter import DocumentConverter  # type: ignore[import]
    except ImportError:
        logger.warning("document_extractor: Docling not installed, cannot extract DOCX '%s'", filename)
        return None

    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
            tmp.write(raw_bytes)
            tmp_path = tmp.name

        converter = DocumentConverter()
        conversion = converter.convert(tmp_path)
        doc = conversion.document

        text = doc.export_to_markdown()
        if not text or not text.strip():
            logger.warning("document_extractor: Docling returned empty text for DOCX '%s'", filename)
            return None

        has_tables = "|" in text

        page_count: int | None = None
        try:
            page_count = len(doc.pages) if hasattr(doc, "pages") else None
        except Exception:
            pass

        return ExtractionResult(
            text=text.strip(),
            method="docling",
            page_count=page_count,
            has_tables=has_tables,
        )

    except Exception as exc:
        logger.warning("document_extractor: Docling DOCX extraction failed for '%s': %s", filename, exc)
        return None

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def _extract_text_file(raw_bytes: bytes) -> ExtractionResult | None:
    """Decode a plain-text file (TXT, MD, CSV) as UTF-8."""
    try:
        text = raw_bytes.decode("utf-8", errors="replace")
        return ExtractionResult(
            text=text,
            method="text",
            page_count=None,
            has_tables=False,
        )
    except Exception as exc:
        logger.warning("document_extractor: text file decode failed: %s", exc)
        return None
