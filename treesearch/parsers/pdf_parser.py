# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Document parser for TreeSearch (PyMuPDF backend).

Uses PyMuPDF (pymupdf) to extract text from PDF/XPS/EPUB/FB2/CBZ/CBR,
then delegates to text_to_tree for structure detection.

Simple pipeline: PyMuPDF page text extraction → text_to_tree (heading detection).
"""
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Backend detection flag
_HAS_PYMUPDF = None


def _check_backends():
    """Lazy-check that PyMuPDF is available."""
    global _HAS_PYMUPDF
    if _HAS_PYMUPDF is None:
        try:
            import pymupdf  # noqa: F401
            _HAS_PYMUPDF = True
        except ImportError:
            _HAS_PYMUPDF = False
    if not _HAS_PYMUPDF:
        raise ImportError(
            "Document parsing (PDF/EPUB/XPS/FB2/CBZ/CBR) requires PyMuPDF. "
            "Install with: pip install pymupdf"
        )


# All file extensions that PyMuPDF can open natively
PYMUPDF_EXTENSIONS = {
    ".pdf", ".xps", ".oxps", ".epub", ".fb2", ".cbz", ".cbr",
}


def extract_pdf_text(file_path: str) -> str:
    """Extract text from a document file using PyMuPDF.

    Supports: PDF, XPS, OpenXPS, EPUB, FB2, CBZ, CBR.

    Args:
        file_path: path to the document file.

    Returns page-aware text with [PAGE N] markers.
    Returns empty string on failure.
    """
    _check_backends()
    try:
        import pymupdf
        doc = pymupdf.open(file_path)
        parts = []
        for i, page in enumerate(doc):
            text = page.get_text().strip()
            if text:
                parts.append(f"\n[PAGE {i + 1}]\n{text}")
        doc.close()
        return "\n".join(parts)
    except Exception as e:
        logger.error("Error extracting text from %s: %s", file_path, e)
        return ""


async def pdf_to_tree(
    file_path: str = "",
    *,
    pdf_path: str = "",
    model: Optional[str] = None,
    if_add_node_summary: bool = True,
    summary_chars_threshold: int = 600,
    if_add_doc_description: bool = False,
    if_add_node_text: bool = False,
    if_add_node_id: bool = True,
    **kwargs,
) -> dict:
    """Build a tree index from a document file using PyMuPDF.

    Supports: PDF, XPS, OpenXPS, EPUB, FB2, CBZ, CBR.

    Simple pipeline: extract page text with [PAGE N] markers → text_to_tree
    for pattern-based heading detection and tree building.

    Args:
        file_path: path to document file (preferred parameter name).
        pdf_path: deprecated alias for file_path, kept for backward compatibility.

    Returns:
        {'doc_name': str, 'structure': list, 'source_path': str}
    """
    # Backward compatibility: accept pdf_path as alias
    fp = file_path or pdf_path
    if not fp:
        raise ValueError("file_path (or pdf_path) is required")

    _check_backends()
    doc_name = os.path.splitext(os.path.basename(fp))[0]
    logger.debug("Parsing document: %s", fp)

    # Extract text with [PAGE N] markers
    text = extract_pdf_text(fp)

    if not text.strip():
        logger.warning("No text extracted from document: %s", fp)
        from ..tree import assign_node_ids
        structure = [{"title": doc_name, "node_id": "0", "text": "", "nodes": []}]
        if if_add_node_id:
            assign_node_ids(structure)
        return {
            "doc_name": doc_name,
            "structure": structure,
            "source_path": os.path.abspath(fp),
        }

    # Delegate to text_to_tree for heading detection and tree building
    from ..indexer import text_to_tree
    result = await text_to_tree(
        text_content=text,
        model=model,
        if_add_node_summary=if_add_node_summary,
        summary_chars_threshold=summary_chars_threshold,
        if_add_doc_description=if_add_doc_description,
        if_add_node_text=if_add_node_text,
        if_add_node_id=if_add_node_id,
        **kwargs,
    )
    result["doc_name"] = doc_name
    result["source_path"] = os.path.abspath(fp)
    return result
