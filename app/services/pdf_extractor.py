"""
PDF text extraction with OCR fallback.

Text-native PDFs: parsed directly by pdfplumber (fast, millisecond-range).

Scanned / image-only PDFs: pdfplumber returns nothing, so we run OCRmyPDF
(Tesseract under the hood) to add a text layer in-memory, then re-parse.
This keeps the happy path fast and only pays the OCR cost when the PDF
genuinely needs it.

All work is done in-memory; nothing ever touches disk.

Designed to fail soft: any upstream error (encrypted, corrupted, truly
unreadable) surfaces as a PDFExtractionError with a patient-friendly
message the caller can show directly.
"""

from __future__ import annotations

import asyncio
import io
import logging

import pdfplumber

logger = logging.getLogger(__name__)

# Per-file ceiling — anything bigger than this on the wire is rejected
# before we ever touch pdfplumber, since malformed/huge PDFs can be slow.
MAX_PDF_BYTES = 10 * 1024 * 1024  # 10 MB

# Per-page hard-cap after extraction, to prevent runaway stored content.
MAX_TOTAL_CHARS = 200_000

# Minimum character count to consider a direct extraction "real".
# Below this, we assume the PDF is scan-only and try OCR.
MIN_TEXT_CHARS_BEFORE_OCR_FALLBACK = 40

# OCR wall-clock budget — anything slower than this returns the original error.
OCR_TIMEOUT_SECONDS = 45


class PDFExtractionError(Exception):
    """Raised when a PDF cannot be read at all (encrypted, malformed, OCR failed)."""


def _plumber_extract(data: bytes) -> str:
    """Try pdfplumber text extraction. Returns '' if nothing extractable."""
    try:
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            pages_text: list[str] = []
            total_chars = 0
            for page in pdf.pages:
                text = page.extract_text() or ""
                text = text.strip()
                if not text:
                    continue
                pages_text.append(text)
                total_chars += len(text)
                if total_chars >= MAX_TOTAL_CHARS:
                    break
    except Exception as e:
        logger.debug(f"pdfplumber direct extract failed: {e}")
        return ""

    if not pages_text:
        return ""
    return "\n\n---\n\n".join(pages_text).strip()[:MAX_TOTAL_CHARS]


def _ocr_sync(data: bytes) -> bytes:
    """Blocking OCR pass. Returns the OCR'd PDF bytes with a text layer.

    Raises PDFExtractionError on unrecoverable OCR failures.
    """
    try:
        import ocrmypdf
    except ImportError as e:
        logger.warning(f"ocrmypdf not installed, cannot OCR: {e}")
        raise PDFExtractionError("This PDF appears to be a scan and OCR is not available in this environment.") from e

    in_buf = io.BytesIO(data)
    out_buf = io.BytesIO()
    try:
        # ocrmypdf accepts positional input/output (file objects or paths).
        # Force OCR because we only get here when text-layer extraction
        # returned nothing — we already know there's no useful text.
        ocrmypdf.ocr(
            in_buf,
            out_buf,
            language="eng",
            force_ocr=True,
            progress_bar=False,
            # Skip image pre-processing steps that need expensive tools when
            # they aren't available; keeps the container slim without failing.
            clean=False,
            deskew=False,
            optimize=0,
        )
    except Exception as e:
        logger.warning(f"OCR pass failed: {e}")
        raise PDFExtractionError(
            "We tried to read this scanned PDF but couldn't make sense of it. "
            "Please upload a clearer scan, or paste the content as text."
        ) from e

    return out_buf.getvalue()


async def _ocr_with_timeout(data: bytes) -> bytes:
    """Run the blocking OCR in a thread with a wall-clock budget."""
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_ocr_sync, data),
            timeout=OCR_TIMEOUT_SECONDS,
        )
    except TimeoutError:
        raise PDFExtractionError(
            f"OCR took longer than {OCR_TIMEOUT_SECONDS}s. For large scans, "
            "please split into smaller PDFs or paste the content as text."
        )


async def extract_text_from_pdf_bytes(data: bytes) -> str:
    """Extract text from a PDF's raw bytes. Tries direct parse, then OCR.

    Raises:
        PDFExtractionError: PDF is unreadable, encrypted, or OCR also failed.
    """
    if len(data) == 0:
        raise PDFExtractionError("Empty file.")
    if len(data) > MAX_PDF_BYTES:
        raise PDFExtractionError(f"PDF exceeds the {MAX_PDF_BYTES // (1024 * 1024)} MB limit.")

    # Fast path: text-native PDF
    text = _plumber_extract(data)
    if len(text) >= MIN_TEXT_CHARS_BEFORE_OCR_FALLBACK:
        return text

    logger.info("PDF looks scan-only; running OCR fallback")

    # Slow path: OCR it, then re-extract
    try:
        ocred_bytes = await _ocr_with_timeout(data)
    except PDFExtractionError:
        # If we already got *some* text from the direct pass, return that
        # rather than losing it to a flaky OCR attempt.
        if text:
            return text
        raise

    text = _plumber_extract(ocred_bytes)
    if not text:
        # Preserve any partial direct-extract text we may have had
        if text:
            return text
        raise PDFExtractionError("No text was found in this PDF, even after OCR. Please type the content manually.")

    return text
