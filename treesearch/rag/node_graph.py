# -*- coding: utf-8 -*-
"""Convert TreeSearch documents into structure-preserving graph passages."""

from treesearch.tree import Document, flatten_tree
from treesearch.rag.models import GraphNodePassage, StructuralEdge


def document_to_node_passages(document: Document) -> list[GraphNodePassage]:
    """Convert every TreeSearch node into a graph passage with grounding metadata."""
    passages: list[GraphNodePassage] = []
    source_path = str(document.metadata.get("source_path", ""))

    for node in flatten_tree(document.structure):
        node_id = str(node.get("node_id", ""))
        if not node_id:
            continue

        path_titles = []
        for path_node_id in document.get_path_to_root(node_id):
            path_node = document.get_node_by_id(path_node_id)
            if path_node:
                path_titles.append(str(path_node.get("title", "")))

        passages.append(
            GraphNodePassage(
                node_id=node_id,
                doc_id=document.doc_id,
                doc_name=document.doc_name,
                source_path=source_path,
                source_type=document.source_type,
                title=str(node.get("title", "")),
                text=str(node.get("text", "")),
                path_titles=tuple(path_titles),
                line_start=node.get("line_start"),
                line_end=node.get("line_end"),
                parent_node_id=document.get_parent_id(node_id),
                child_node_ids=tuple(document.get_children_ids(node_id)),
                sibling_node_ids=tuple(document.get_sibling_ids(node_id)),
            )
        )

    return passages


def document_to_structural_edges(document: Document) -> list[StructuralEdge]:
    """Create bidirectional structural edges over TreeSearch node IDs."""
    edges: list[StructuralEdge] = []
    for node in flatten_tree(document.structure):
        node_id = str(node.get("node_id", ""))
        if not node_id:
            continue

        parent_id = document.get_parent_id(node_id)
        if parent_id:
            edges.append(StructuralEdge(doc_id=document.doc_id, src_node_id=parent_id, dst_node_id=node_id, edge_type="child"))
            edges.append(StructuralEdge(doc_id=document.doc_id, src_node_id=node_id, dst_node_id=parent_id, edge_type="parent"))

        for sibling_id in document.get_sibling_ids(node_id):
            edges.append(StructuralEdge(doc_id=document.doc_id, src_node_id=node_id, dst_node_id=sibling_id, edge_type="sibling"))

    return edges
