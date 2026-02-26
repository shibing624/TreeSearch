# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Tree structure data models and operations.
"""
import copy
import glob as globmod
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
    """Recursively assign zero-padded node_id to each node. Returns next available id."""
    if isinstance(data, dict):
        data["node_id"] = str(node_id).zfill(4)
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
# Persistence
# ---------------------------------------------------------------------------

INDEX_VERSION = "1.0"

# Global document cache: path -> Document
_doc_cache: dict[str, Document] = {}


def save_index(index: dict, path: str) -> None:
    """Save tree structure index to a JSON file with version metadata."""
    index["index_version"] = INDEX_VERSION
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
    logger.info("Index saved to: %s", path)


def load_index(path: str) -> Document:
    """Load a single index JSON file and return a Document object.

    Results are cached: loading the same path again returns the cached Document.
    """
    abs_path = os.path.abspath(path)
    if abs_path in _doc_cache:
        return _doc_cache[abs_path]

    with open(abs_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    file_version = data.get("index_version", "unknown")
    if file_version != INDEX_VERSION:
        logger.warning("Index version mismatch: file=%s, expected=%s, path=%s",
                       file_version, INDEX_VERSION, abs_path)

    basename = os.path.basename(abs_path)
    doc_id = os.path.splitext(basename)[0].replace("_structure", "")
    doc = Document(
        doc_id=doc_id,
        doc_name=data.get("doc_name", doc_id),
        structure=data.get("structure", []),
        doc_description=data.get("doc_description", ""),
        metadata={"source_path": data.get("source_path", "")},
    )
    _doc_cache[abs_path] = doc
    return doc


def load_documents(index_dir: str) -> list[Document]:
    """Load all index JSON files from a directory and return a list of Documents.

    Skips meta files (starting with '_'). Uses cache internally.
    """
    pattern = os.path.join(index_dir, "*.json")
    files = sorted(
        f for f in globmod.glob(pattern)
        if not os.path.basename(f).startswith("_")
    )
    return [load_index(f) for f in files]


def clear_doc_cache() -> None:
    """Clear the global document cache."""
    _doc_cache.clear()


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
