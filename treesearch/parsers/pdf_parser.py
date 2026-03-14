# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: PDF parser for TreeSearch.

Requires optional dependency: ``pip install pageindex``
Extracts text from PDF and builds tree structure based on heading detection.
"""
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


async def pdf_to_tree(
    pdf_path: str,
    *,
    model: Optional[str] = None,
    if_add_node_summary: bool = True,
    summary_token_threshold: int = 200,
    if_add_doc_description: bool = False,
    if_add_node_text: bool = False,
    if_add_node_id: bool = True,
    **kwargs,
) -> dict:
    """Build a tree index from a PDF file.

    Uses ``pageindex`` to extract text, then delegates to ``text_to_tree``
    for structure detection.

    Returns:
        {'doc_name': str, 'structure': list, 'source_path': str}
    """
    try:
        import PyPDF2
    except ImportError:
        raise ImportError(
            "PDF support requires 'PyPDF2'. Install with: pip install PyPDF2"
        )

    doc_name = os.path.splitext(os.path.basename(pdf_path))[0]
    logger.info("Extracting text from PDF: %s", pdf_path)

    try:
        pdf_reader = PyPDF2.PdfReader(pdf_path)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    except Exception as e:
        logger.error("Error extracting text from PDF %s: %s", pdf_path, e)
        text = ""

    if not text or not text.strip():
        logger.warning("No text extracted from PDF: %s", pdf_path)
        from ..tree import assign_node_ids
        structure = [{"title": doc_name, "node_id": "0", "text": "", "nodes": []}]
        if if_add_node_id:
            assign_node_ids(structure)
        return {
            "doc_name": doc_name,
            "structure": structure,
            "source_path": os.path.abspath(pdf_path),
        }

    from ..indexer import text_to_tree
    result = await text_to_tree(
        text_content=text,
        model=model,
        if_add_node_summary=if_add_node_summary,
        summary_token_threshold=summary_token_threshold,
        if_add_doc_description=if_add_doc_description,
        if_add_node_text=if_add_node_text,
        if_add_node_id=if_add_node_id,
        **kwargs,
    )
    result["doc_name"] = doc_name
    result["source_path"] = os.path.abspath(pdf_path)
    return result
