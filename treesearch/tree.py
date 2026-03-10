# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Tree structure data models and operations.
"""
import copy
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Document:
    """A document with its tree structure index."""
    doc_id: str
    doc_name: str
    structure: list  # tree structure (list of root nodes)
    doc_description: str = ""
    metadata: dict = field(default_factory=dict)
    source_type: str = ""  # e.g. "markdown", "code", "text", "json", "csv"

    def __post_init__(self):
        self._node_map: dict[str, dict] = {}
        self._rebuild_node_map()

    def _rebuild_node_map(self) -> None:
        """Build O(1) node_id -> node dict mapping from tree structure."""
        self._node_map.clear()
        for node in flatten_tree(self.structure):
            nid = node.get("node_id")
            if nid:
                self._node_map[nid] = node

    def get_tree_without_text(self) -> list:
        """Return tree with text fields removed (for LLM prompts)."""
        return remove_fields(copy.deepcopy(self.structure), fields=["text"])

    def get_node_by_id(self, node_id: str) -> Optional[dict]:
        """Find a node by its node_id. O(1) via cached node map."""
        return self._node_map.get(node_id)


# ---------------------------------------------------------------------------
# Tree traversal
# ---------------------------------------------------------------------------

def flatten_tree(structure) -> list[dict]:
    """Flatten a tree structure into a list of node dicts."""
    nodes = []
    if isinstance(structure, dict):
        nodes.append(structure)
        if "nodes" in structure:
            nodes.extend(flatten_tree(structure["nodes"]))
    elif isinstance(structure, list):
        for item in structure:
            nodes.extend(flatten_tree(item))
    return nodes


def get_leaf_nodes(structure) -> list[dict]:
    """Return all leaf nodes (no children)."""
    leaves = []
    if isinstance(structure, dict):
        if not structure.get("nodes"):
            leaf = copy.deepcopy(structure)
            leaf.pop("nodes", None)
            return [leaf]
        leaves.extend(get_leaf_nodes(structure["nodes"]))
    elif isinstance(structure, list):
        for item in structure:
            leaves.extend(get_leaf_nodes(item))
    return leaves


def find_node(structure, node_id: str) -> Optional[dict]:
    """Find a node by node_id in the tree."""
    if isinstance(structure, dict):
        if structure.get("node_id") == node_id:
            return structure
        for child in structure.get("nodes", []):
            result = find_node(child, node_id)
            if result:
                return result
    elif isinstance(structure, list):
        for item in structure:
            result = find_node(item, node_id)
            if result:
                return result
    return None


# ---------------------------------------------------------------------------
# Node ID assignment
# ---------------------------------------------------------------------------

def assign_node_ids(data, node_id: int = 0) -> int:
    """Recursively assign node_id to each node. Returns next available id.

    Uses variable-length encoding: no fixed zero-padding, supports any tree size.
    """
    if isinstance(data, dict):
        data["node_id"] = str(node_id)
        node_id += 1
        if "nodes" in data:
            node_id = assign_node_ids(data["nodes"], node_id)
    elif isinstance(data, list):
        for item in data:
            node_id = assign_node_ids(item, node_id)
    return node_id


# ---------------------------------------------------------------------------
# Field operations
# ---------------------------------------------------------------------------

def remove_fields(data, fields: list[str] = None) -> any:
    """Recursively remove specified fields from a tree structure."""
    if fields is None:
        fields = ["text"]
    if isinstance(data, dict):
        return {k: remove_fields(v, fields) for k, v in data.items() if k not in fields}
    elif isinstance(data, list):
        return [remove_fields(item, fields) for item in data]
    return data


def reorder_dict(data: dict, key_order: list[str]) -> dict:
    """Reorder dict keys according to key_order."""
    if not key_order:
        return data
    return {k: data[k] for k in key_order if k in data}


def format_structure(structure, order: list[str] = None):
    """Recursively format tree nodes with specified key order."""
    if not order:
        return structure
    if isinstance(structure, dict):
        if "nodes" in structure:
            structure["nodes"] = format_structure(structure["nodes"], order)
        if not structure.get("nodes"):
            structure.pop("nodes", None)
        structure = reorder_dict(structure, order)
    elif isinstance(structure, list):
        structure = [format_structure(item, order) for item in structure]
    return structure


# ---------------------------------------------------------------------------
# Persistence (SQLite DB via FTS5Index)
# ---------------------------------------------------------------------------

INDEX_VERSION = "1.0"


def save_index(index: dict, db_path: str, doc_id: str = "") -> None:
    """Save tree structure index into a SQLite database.

    Args:
        index: dict with 'doc_name', 'structure', and optional metadata.
        db_path: path to the .db file.
        doc_id: explicit doc_id. If empty, derived from doc_name.
    """
    from .fts import FTS5Index
    doc_name = index.get("doc_name", "untitled")
    if not doc_id:
        doc_id = doc_name
    doc = Document(
        doc_id=doc_id,
        doc_name=doc_name,
        structure=index.get("structure", []),
        doc_description=index.get("doc_description", ""),
        metadata={"source_path": index.get("source_path", "")},
        source_type=index.get("source_type", ""),
    )
    fts = FTS5Index(db_path=db_path)
    fts.save_document(doc)
    fts.index_document(doc)
    fts.close()
    logger.info("Index saved to DB: %s (doc_id=%s)", db_path, doc_id)


def load_index(db_path: str, doc_id: str = "") -> Document:
    """Load a single Document from a SQLite database.

    Args:
        db_path: path to the .db file.
        doc_id: the document to load. If empty, loads the first document found.

    Returns:
        Document object.
    """
    from .fts import FTS5Index
    fts = FTS5Index(db_path=db_path)
    if doc_id:
        doc = fts.load_document(doc_id)
    else:
        docs = fts.load_all_documents()
        doc = docs[0] if docs else None
    fts.close()
    if doc is None:
        raise FileNotFoundError(f"No document found in DB: {db_path} (doc_id={doc_id!r})")
    return doc


def load_documents(db_path: str) -> list[Document]:
    """Load all Documents from a SQLite database.

    Args:
        db_path: path to the .db file.

    Returns:
        List of Document objects.
    """
    from .fts import FTS5Index
    fts = FTS5Index(db_path=db_path)
    docs = fts.load_all_documents()
    fts.close()
    return docs


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def print_toc(tree, indent: int = 0) -> None:
    """Print tree as a table of contents."""
    if isinstance(tree, list):
        for node in tree:
            print("  " * indent + node.get("title", ""))
            if node.get("nodes"):
                print_toc(node["nodes"], indent + 1)
    elif isinstance(tree, dict):
        print("  " * indent + tree.get("title", ""))
        if tree.get("nodes"):
            print_toc(tree["nodes"], indent + 1)


def print_tree_json(data, max_len: int = 40, indent: int = 2) -> None:
    """Print tree as JSON with long strings truncated."""
    def _truncate(obj):
        if isinstance(obj, dict):
            return {k: _truncate(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_truncate(item) for item in obj]
        elif isinstance(obj, str) and len(obj) > max_len:
            return obj[:max_len] + "..."
        return obj

    print(json.dumps(_truncate(data), indent=indent, ensure_ascii=False))
