# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: HTML parser for TreeSearch.

Requires optional dependency: ``pip install beautifulsoup4``
Extracts heading structure and text from HTML using BeautifulSoup.
"""
import logging
import os
import re
from typing import Optional

logger = logging.getLogger(__name__)


def _extract_html_structure(html_content: str) -> tuple[list[dict], str]:
    """Extract headings and body text from HTML content.

    Returns:
        (headings, plain_text) where headings is a list of
        {'title': str, 'line_num': int, 'level': int}
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        raise ImportError(
            "HTML parser requires 'beautifulsoup4'. Install with: pip install beautifulsoup4"
        )

    soup = BeautifulSoup(html_content, "html.parser")

    # Remove script and style elements
    for tag in soup(["script", "style"]):
        tag.decompose()

    # Extract plain text
    plain_text = soup.get_text(separator="\n")
    # Collapse blank lines
    plain_text = re.sub(r"\n{3,}", "\n\n", plain_text).strip()
    lines = plain_text.split("\n")

    # Extract headings (h1-h6)
    headings = []
    heading_tags = soup.find_all(re.compile(r"^h[1-6]$"))
    for tag in heading_tags:
        level = int(tag.name[1])
        title = tag.get_text(strip=True)
        if not title:
            continue
        # Find approximate line number by searching in plain text
        line_num = 1
        title_lower = title.lower().strip()
        for i, line in enumerate(lines):
            if title_lower in line.lower().strip():
                line_num = i + 1
                break
        headings.append({
            "title": title,
            "line_num": line_num,
            "level": level,
        })

    return headings, plain_text


async def html_to_tree(
    html_path: str,
    *,
    model: Optional[str] = None,
    if_add_node_summary: bool = True,
    summary_chars_threshold: int = 600,
    if_add_doc_description: bool = False,
    if_add_node_text: bool = False,
    if_add_node_id: bool = True,
    **kwargs,
) -> dict:
    """Build a tree index from an HTML file.

    Uses BeautifulSoup to extract headings (h1-h6) and text content.
    Falls back to ``text_to_tree`` if no headings are found.

    Returns:
        {'doc_name': str, 'structure': list, 'source_path': str}
    """
    doc_name = os.path.splitext(os.path.basename(html_path))[0]
    logger.debug("Parsing HTML: %s", html_path)

    with open(html_path, "r", encoding="utf-8", errors="replace") as f:
        html_content = f.read()

    headings, plain_text = _extract_html_structure(html_content)

    if not headings:
        from ..indexer import text_to_tree
        result = await text_to_tree(
            text_content=plain_text,
            model=model,
            if_add_node_summary=if_add_node_summary,
            summary_chars_threshold=summary_chars_threshold,
            if_add_doc_description=if_add_doc_description,
            if_add_node_text=if_add_node_text,
            if_add_node_id=if_add_node_id,
            **kwargs,
        )
        result["doc_name"] = doc_name
        result["source_path"] = os.path.abspath(html_path)
        return result

    # Build nodes from headings
    from ..indexer import _build_tree, generate_summaries, generate_doc_description
    from ..tree import assign_node_ids, format_structure

    lines = plain_text.split("\n")
    nodes = []
    for i, hd in enumerate(headings):
        start = hd["line_num"] - 1
        end = headings[i + 1]["line_num"] - 1 if i + 1 < len(headings) else len(lines)
        text = "\n".join(lines[start:end]).strip()
        nodes.append({
            "title": hd["title"],
            "line_num": hd["line_num"],
            "line_start": hd["line_num"],
            "line_end": end,
            "level": hd["level"],
            "text": text,
        })

    tree = _build_tree(nodes)

    if if_add_node_id:
        assign_node_ids(tree)

    base_order = ["title", "node_id", "summary", "prefix_summary"]
    text_fields = ["text"] if if_add_node_text or if_add_node_summary else []
    tail_fields = ["line_start", "line_end", "nodes"]
    order = base_order + text_fields + tail_fields
    tree = format_structure(tree, order=order)

    if if_add_node_summary:
        tree = generate_summaries(tree, threshold=summary_chars_threshold)
        if not if_add_node_text:
            order_no_text = [f for f in order if f != "text"]
            tree = format_structure(tree, order=order_no_text)

    result = {"doc_name": doc_name, "structure": tree, "source_path": os.path.abspath(html_path)}

    if if_add_doc_description:
        result["doc_description"] = generate_doc_description(tree)

    return result
