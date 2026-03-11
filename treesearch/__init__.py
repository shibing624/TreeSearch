# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: TreeSearch - Structure-aware document retrieval via tree-structured indexing.

No vector embeddings. No chunk splitting. FTS5/BM25 keyword matching over document trees, with optional LLM reasoning for enhanced accuracy.

Quick Start::

    from treesearch import TreeSearch

    # Lazy indexing -- auto-builds index on first search
    ts = TreeSearch("docs/*.md", "src/*.py", model="gpt-4o")
    results = ts.search("How to configure voice calls?")

FTS5 Compatibility (CentOS/older systems):

    pip install pysqlite3-binary

    The library will automatically use pysqlite3 (with FTS5) if available.
"""
__version__ = "0.5.6"

# ============================================================================
# FTS5 Compatibility: Use pysqlite3 on systems without FTS5 support
# ============================================================================
# On CentOS/older systems, SQLite may be compiled without FTS5 extension.
# pysqlite3-binary provides a modern SQLite with FTS5 enabled.
#
# Installation: pip install pysqlite3-binary
#
# If neither pysqlite3 nor system FTS5 is available, the library falls back
# to plain-table LIKE-based search (reduced ranking quality, but functional).
import sys

def _ensure_fts5_support():
    """Ensure SQLite has FTS5 support, use pysqlite3 as fallback."""
    try:
        import sqlite3
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE VIRTUAL TABLE _fts5_test USING fts5(content)")
        conn.close()
        return True
    except Exception:
        return False

if not _ensure_fts5_support():
    try:
        from pysqlite3 import dbapi2 as _sqlite3
        sys.modules["sqlite3"] = _sqlite3
        if "treesearch.fts" in sys.modules:
            sys.modules["treesearch.fts"].sqlite3 = _sqlite3
    except ImportError:
        import warnings
        warnings.warn(
            "SQLite FTS5 not available. Full-text search will use LIKE fallback "
            "(slower, no BM25 ranking). For best performance, install pysqlite3-binary:\n"
            "    pip install pysqlite3-binary",
            RuntimeWarning,
            stacklevel=2,
        )

# -- Primary API: TreeSearch is the only class most users need --
from treesearch.treesearch import TreeSearch

# -- Advanced / Power-user API --
# Index & Document
from treesearch.tree import Document, load_index, load_documents, save_index
from treesearch.indexer import build_index, md_to_tree, text_to_tree, code_to_tree, json_to_tree, csv_to_tree

# Search
from treesearch.search import search, search_sync
from treesearch.search import BestFirstTreeSearch, route_documents, PreFilter, GrepFilter

# Configuration
from treesearch.config import TreeSearchConfig, get_config, set_config, reset_config

# FTS5
from treesearch.fts import FTS5Index, get_fts_index, set_fts_index, reset_fts_index

# BM25
from treesearch.rank_bm25 import NodeBM25Index, NodeTFIDFIndex, BM25Okapi, tokenize

# Tree utilities
from treesearch.tree import (
    INDEX_VERSION,
    assign_node_ids,
    flatten_tree,
    find_node,
    get_leaf_nodes,
    remove_fields,
    format_structure,
    print_toc,
    print_tree_json,
)

# Parser registry
from treesearch.parsers import ParserRegistry, get_parser

# LLM utilities (lazy import to avoid loading openai/tiktoken at startup)
def __getattr__(name):
    _llm_names = {"achat", "chat", "count_tokens", "extract_json"}
    if name in _llm_names:
        from treesearch.llm import achat, chat, count_tokens, extract_json
        globals().update({"achat": achat, "chat": chat, "count_tokens": count_tokens, "extract_json": extract_json})
        return globals()[name]
    raise AttributeError(f"module 'treesearch' has no attribute {name!r}")

__all__ = [
    # Primary API
    "TreeSearch",
    # Advanced API
    "build_index", "md_to_tree", "text_to_tree", "code_to_tree", "json_to_tree", "csv_to_tree",
    "Document", "load_index", "load_documents", "save_index",
    "search", "search_sync",
    "BestFirstTreeSearch", "route_documents", "PreFilter", "GrepFilter",
    "TreeSearchConfig", "get_config", "set_config", "reset_config",
    "FTS5Index", "get_fts_index", "set_fts_index", "reset_fts_index",
    "NodeBM25Index", "NodeTFIDFIndex", "BM25Okapi", "tokenize",
    "INDEX_VERSION", "assign_node_ids", "flatten_tree", "find_node",
    "get_leaf_nodes", "remove_fields", "format_structure", "print_toc", "print_tree_json",
    "ParserRegistry", "get_parser",
    "achat", "chat", "count_tokens", "extract_json",
]
