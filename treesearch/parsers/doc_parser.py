# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: DOC (Word 97-2003) parser for TreeSearch.

Extracts text from legacy .doc files using available system tools:
- macOS: textutil (built-in)
- Linux/other: antiword or catdoc

Falls back to text_to_tree for structure detection after extraction.
"""
import logging
import os
import shutil
import subprocess
import tempfile
from typing import Optional

logger = logging.getLogger(__name__)


def _extract_doc_text_textutil(doc_path: str) -> str:
    """Extract text from .doc using macOS textutil command."""
    try:
        result = subprocess.run(
            ["textutil", "-convert", "txt", "-stdout", doc_path],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return ""


def _extract_doc_text_antiword(doc_path: str) -> str:
    """Extract text from .doc using antiword command."""
    try:
        result = subprocess.run(
            ["antiword", doc_path],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return ""


def _extract_doc_text_catdoc(doc_path: str) -> str:
    """Extract text from .doc using catdoc command."""
    try:
        result = subprocess.run(
            ["catdoc", doc_path],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return ""


def _extract_doc_text_libreoffice(doc_path: str) -> str:
    """Extract text by converting .doc to .docx via LibreOffice, then reading with python-docx."""
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice:
        return ""

    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            subprocess.run(
                [soffice, "--headless", "--convert-to", "docx", "--outdir", tmp_dir, doc_path],
                capture_output=True, timeout=120,
            )
            base = os.path.splitext(os.path.basename(doc_path))[0]
            docx_path = os.path.join(tmp_dir, base + ".docx")
            if os.path.exists(docx_path):
                from docx import Document as DocxDocument
                doc = DocxDocument(docx_path)
                return "\n".join(p.text for p in doc.paragraphs)
        except (FileNotFoundError, subprocess.TimeoutExpired, ImportError, Exception) as e:
            logger.debug("LibreOffice conversion failed: %s", e)
    return ""


def extract_doc_text(doc_path: str) -> str:
    """Extract text from a .doc (Word 97-2003) file.

    Tries multiple backends in order:
    1. textutil (macOS built-in)
    2. antiword
    3. catdoc
    4. LibreOffice headless conversion

    Returns extracted text or empty string on failure.
    """
    for extractor in (
        _extract_doc_text_textutil,
        _extract_doc_text_antiword,
        _extract_doc_text_catdoc,
        _extract_doc_text_libreoffice,
    ):
        text = extractor(doc_path)
        if text.strip():
            logger.debug("Extracted .doc text via %s: %d chars", extractor.__name__, len(text))
            return text

    logger.warning(
        "No backend available to extract .doc text from %s. "
        "Install one of: textutil (macOS), antiword, catdoc, or LibreOffice.",
        doc_path,
    )
    return ""


async def doc_to_tree(
    doc_path: str,
    *,
    model: Optional[str] = None,
    if_add_node_summary: bool = True,
    summary_chars_threshold: int = 600,
    if_add_doc_description: bool = False,
    if_add_node_text: bool = False,
    if_add_node_id: bool = True,
    **kwargs,
) -> dict:
    """Build a tree index from a .doc (Word 97-2003) file.

    Extracts text using system tools, then applies text_to_tree
    for structure detection.

    Returns:
        {'doc_name': str, 'structure': list, 'source_path': str}
    """
    doc_name = os.path.splitext(os.path.basename(doc_path))[0]
    logger.debug("Parsing DOC: %s", doc_path)

    text = extract_doc_text(doc_path)
    if not text.strip():
        logger.warning("No text extracted from .doc: %s", doc_path)
        from ..tree import assign_node_ids
        structure = [{"title": doc_name, "node_id": "0", "text": "", "nodes": []}]
        if if_add_node_id:
            assign_node_ids(structure)
        return {
            "doc_name": doc_name,
            "structure": structure,
            "source_path": os.path.abspath(doc_path),
        }

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
    result["source_path"] = os.path.abspath(doc_path)
    return result
