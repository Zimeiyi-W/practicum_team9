"""
tools/pdf_extract_tool.py — PDF extraction and schema-aligned field mapping

Purpose
-------
Provide deterministic helpers for:
  1. Extracting full text from native-text or scanned PDFs
  2. Preserving page-level output and extraction metadata
  3. Mapping extracted text into a partial RawProjectRecord

This module is designed to align document ingestion with the project's
canonical schema and ETL transforms.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

from schema import RawProjectRecord

logger = logging.getLogger(__name__)

_MIN_CHARS_PER_PAGE_FOR_NATIVE_TEXT = 200


def _safe_import_pypdf2():
    import PyPDF2
    return PyPDF2


def _safe_import_fitz():
    import fitz
    return fitz


def _safe_import_pytesseract():
    import pytesseract
    return pytesseract


def _normalize_whitespace(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_native_text(pdf_path: str | Path) -> dict[str, Any]:
    PyPDF2 = _safe_import_pypdf2()
    path = Path(pdf_path)

    page_text: list[dict[str, Any]] = []
    total_chars = 0

    with path.open("rb") as f:
        reader = PyPDF2.PdfReader(f)
        for i, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception as e:
                logger.warning("Native extraction failed on page %s: %s", i, e)
                text = ""

            text = _normalize_whitespace(text)
            total_chars += len(text)
            page_text.append({"page": i, "text": text, "char_count": len(text)})

    page_count = len(page_text)
    avg_chars = total_chars / max(page_count, 1)

    return {
        "document_text": "\n\n".join(
            f"--- PAGE {p['page']} ---\n{p['text']}" for p in page_text
        ).strip(),
        "page_text": page_text,
        "page_count": page_count,
        "total_chars": total_chars,
        "avg_chars_per_page": avg_chars,
        "extraction_method": "native_pdf",
        "ocr_used": False,
        "warnings": [],
        "status": "success",
    }


def _extract_ocr_text(pdf_path: str | Path) -> dict[str, Any]:
    from PIL import Image
    import io
    fitz = _safe_import_fitz()
    pytesseract = _safe_import_pytesseract()
    path = Path(pdf_path)

    doc = fitz.open(path)
    page_text: list[dict[str, Any]] = []
    total_chars = 0
    warnings: list[str] = []

    for i in range(len(doc)):
        page_num = i + 1
        try:
            page = doc.load_page(i)
            pix = page.get_pixmap(dpi=200)

            img = Image.open(io.BytesIO(pix.tobytes("png")))
            text = pytesseract.image_to_string(img) or ""
            text = _normalize_whitespace(text)

        except Exception as e:
            logger.warning("OCR failed on page %s: %s", page_num, e)
            warnings.append(f"OCR failed on page {page_num}: {e}")
            text = ""

        total_chars += len(text)
        page_text.append({"page": page_num, "text": text, "char_count": len(text)})

    page_count = len(page_text)
    avg_chars = total_chars / max(page_count, 1)

    return {
        "document_text": "\n\n".join(
            f"--- PAGE {p['page']} ---\n{p['text']}" for p in page_text
        ).strip(),
        "page_text": page_text,
        "page_count": page_count,
        "total_chars": total_chars,
        "avg_chars_per_page": avg_chars,
        "extraction_method": "ocr_fallback",
        "ocr_used": True,
        "warnings": warnings,
        "status": "partial_success" if warnings else "success",
    }


def extract_pdf_text(
    pdf_path: str | Path,
    *,
    native_text_threshold: int = _MIN_CHARS_PER_PAGE_FOR_NATIVE_TEXT,
) -> dict[str, Any]:
    """
    Extract full text from a PDF with native-text first, OCR fallback second.
    """
    path = Path(pdf_path)
    if not path.exists():
        return {
            "document_text": "",
            "page_text": [],
            "page_count": 0,
            "total_chars": 0,
            "avg_chars_per_page": 0.0,
            "extraction_method": "none",
            "ocr_used": False,
            "warnings": [f"File not found: {path}"],
            "status": "failed",
        }

    native_result: dict[str, Any] | None = None

    try:
        native_result = _extract_native_text(path)
        if native_result["avg_chars_per_page"] >= native_text_threshold:
            return native_result
        native_result["warnings"].append(
            f"Native extraction fell below threshold ({native_result['avg_chars_per_page']:.1f} chars/page); OCR fallback used."
        )
    except Exception as e:
        logger.warning("Native extraction failed entirely: %s", e)
        native_result = {
            "document_text": "",
            "page_text": [],
            "page_count": 0,
            "total_chars": 0,
            "avg_chars_per_page": 0.0,
            "extraction_method": "native_pdf",
            "ocr_used": False,
            "warnings": [f"Native extraction failed: {e}"],
            "status": "partial_success",
        }

    try:
        ocr_result = _extract_ocr_text(path)
        ocr_result["warnings"] = native_result.get("warnings", []) + ocr_result.get("warnings", [])
        return ocr_result
    except Exception as e:
        logger.exception("OCR fallback failed")
        return {
            "document_text": native_result.get("document_text", ""),
            "page_text": native_result.get("page_text", []),
            "page_count": native_result.get("page_count", 0),
            "total_chars": native_result.get("total_chars", 0),
            "avg_chars_per_page": native_result.get("avg_chars_per_page", 0.0),
            "extraction_method": native_result.get("extraction_method", "native_pdf"),
            "ocr_used": native_result.get("ocr_used", False),
            "warnings": native_result.get("warnings", []) + [f"OCR fallback failed: {e}"],
            "status": "partial_success" if native_result.get("document_text") else "failed",
        }


def _search_first(pattern: str, text: str, flags: int = re.IGNORECASE) -> Optional[str]:
    match = re.search(pattern, text, flags)
    return match.group(1).strip() if match else None


def _infer_state(text: str) -> Optional[str]:
    value = _search_first(r"\bstate\s*[:\-]\s*([A-Z]{2})\b", text)
    return value.upper() if value else None


def _infer_city(text: str) -> Optional[str]:
    return _search_first(r"\bcity\s*[:\-]\s*([A-Za-z .'-]+)", text)


def _infer_county(text: str) -> Optional[str]:
    return _search_first(r"\bcounty\s*[:\-]\s*([A-Za-z .'-]+)", text)


def _infer_project_sq_ft(text: str) -> Optional[float]:
    value = _search_first(
        r"\b(?:square footage|square feet|sq\.?\s*ft\.?|sf)\s*[:\-]?\s*([\d,]+(?:\.\d+)?)",
        text,
    )
    if not value:
        return None
    try:
        return float(value.replace(",", ""))
    except ValueError:
        return None


def _infer_budget_range(text: str) -> Optional[str]:
    known = [
        "Less than 1M",
        "$1M-$3M",
        "$3M-$6M",
        "$6M-$10M",
        "$10M-$20M",
        "$20M+",
    ]
    lower = text.lower()
    for item in known:
        if item.lower() in lower:
            return item
    return None


def _infer_phase_description(text: str) -> Optional[str]:
    return _search_first(
        r"\b(?:phase|design phase|project phase)\s*[:\-]\s*([A-Za-z /&-]+)",
        text,
    )


def _infer_project_type(text: str) -> Optional[str]:
    return _search_first(r"\bproject type\s*[:\-]\s*([A-Za-z0-9 /&,'().-]+)", text)


def _infer_project_category(text: str) -> Optional[str]:
    return _search_first(r"\bproject category\s*[:\-]\s*([A-Za-z0-9 /&,'().-]+)", text)


def _infer_complexity_category(text: str) -> Optional[str]:
    value = _search_first(r"\b(?:complexity|ciqs complexity)\s*[:\-]\s*(Category\s*[1-4])", text)
    return value if value else None


def extract_project_record(
    extraction_result: dict[str, Any],
    *,
    project_id: str = "pdf_extraction",
) -> dict[str, Any]:
    """
    Map extracted PDF text into a partial RawProjectRecord.

    Returns a schema-native payload for downstream ETL and model transforms.
    """
    text = extraction_result.get("document_text", "") or ""
    warnings = list(extraction_result.get("warnings", []))

    record = RawProjectRecord(
        project_id=project_id,
        project_city=_infer_city(text),
        project_state=_infer_state(text),
        county_name=_infer_county(text),
        project_sq_ft=_infer_project_sq_ft(text),
        official_budget_range=_infer_budget_range(text),
        phase_description=_infer_phase_description(text),
        project_type=_infer_project_type(text),
        project_category=_infer_project_category(text),
        ciqs_complexity_category=_infer_complexity_category(text),
        project_description=text[:5000] if text else None,
    )

    missing_required_for_simple = [
        field_name
        for field_name in [
            "project_type",
            "project_category",
            "project_state",
        ]
        if getattr(record, field_name, None) in (None, "")
    ]

    missing_recommended_for_simple = [
        field_name
        for field_name in [
            "county_name",
            "official_budget_range",
            "ciqs_complexity_category",
            "project_sq_ft",
        ]
        if getattr(record, field_name, None) in (None, "")
    ]

    if not text:
        warnings.append("No extracted text available for schema mapping.")

    return {
        "project_record": asdict(record),
        "missing_required_for_simple": missing_required_for_simple,
        "missing_recommended_for_simple": missing_recommended_for_simple,
        "warnings": warnings,
        "status": extraction_result.get("status", "failed"),
        "extraction_method": extraction_result.get("extraction_method", "none"),
        "ocr_used": extraction_result.get("ocr_used", False),
    }
